"""
churn_prediction/predict.py

저장된 model.pkl + feature_transformer.pkl 을 불러와 새 고객 데이터에
이탈확률만 매긴다. 학습은 전혀 일어나지 않으므로 매우 빠르다.

⚠️ 이 스크립트에는 "주기"가 없다 - train.py(재학습)는 segment_discovery/app.py
(분석A/B/Q)와 같은 묶음으로, 충분한 표본이 쌓여야 의미가 있어 가끔 함께
실행하지만, 추론은 그런 제약이 없으므로 필요할 때마다(매일, 매주, 즉석
조회 등) 독립적으로 자유롭게 실행할 수 있다. "최신 데이터 반영"이라는
역할은 재학습이 아니라 바로 이 추론 단계가 담당한다.
호출하는 쪽(사람, 스케줄러, 추후 webapp)이 언제 부르든 그 순간의
입력 데이터로 즉시 답을 준다.

[실무 구조] 모델/변환규칙은 항상 outputs/latest/ 에서 읽는다 (train.py 가
가장 최근 버전을 그곳에 복사해둠). 과거 특정 버전을 쓰고 싶으면 --model,
--transformer 로 outputs/versions/{timestamp}/ 안의 파일을 직접 지정할 것.

입력 데이터는 원본과 동일한 컬럼 구조(customerID, tenure, Contract 등)를
가진 CSV면 된다 - 신규 가입자든, 기존 고객의 갱신된 정보든 상관없다.
전체 데이터의 일부 범위(예: tenure 0~5개월만, 또는 특정 segment만)를 미리
필터링해서 넣어도 문제없이 동작한다 - 한 행씩 독립적으로 변환/추론하므로
입력 범위와 무관하게 항상 안전하다.

⚠️ "기간(월) 단위로 범위를 나눠 본다"는 게 의미를 가지려면, 날짜/수집시점
컬럼이 아니라 tenure(가입 후 경과월)를 기준으로 나눠야 한다. 이 데이터는
절대 가입일/수집일이 없는 스냅샷 데이터라(기획_메모.md 2.2 참조), "2026년
5월에 가입한 고객" 같은 절대 시점 필터링은 할 수 없고, "tenure 0~5개월인
고객들"처럼 경과월 구간으로만 범위를 나눌 수 있다. 별도의 날짜 컬럼을
추가할 필요는 없다 - 이미 있는 tenure 컬럼으로 충분하다.

[신규 데이터가 아직 없을 때] --data 를 지정하지 않고 그냥 실행(또는 VSCode
실행버튼)하면, data/new_customers.csv 를 먼저 찾는다. 그 파일이 없으면
분석A/B/Q가 썼던 원본 데이터로 자동 대체해서 "데모 모드"로 실행한다 -
신규 데이터가 아니라 동작 확인용이라는 점을 출력에 명확히 표시한다.

사용법:
    cd churn_prediction
    python predict.py --data ../data/new_customers.csv
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_engineering import FeatureTransformer  # noqa: E402

LATEST_DIR = Path(__file__).parent / "outputs" / "latest"
DEFAULT_MODEL_PATH = LATEST_DIR / "model.pkl"
DEFAULT_TRANSFORMER_PATH = LATEST_DIR / "feature_transformer.pkl"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "outputs" / "latest_predictions.csv"

DEFAULT_NEW_DATA_PATH = config.DEFAULT_NEW_DATA_PATH
DEMO_FALLBACK_DATA_PATH = config.DEFAULT_DATA_PATH


def predict(
    df_raw: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    transformer_path: str | Path = DEFAULT_TRANSFORMER_PATH,
) -> pd.DataFrame:
    """
    새 고객 데이터(df_raw, 원본 컬럼 그대로)에 이탈확률을 매겨 반환.
    학습이 전혀 없으므로 호출할 때마다 즉시 끝난다 - 언제 불러도 상관없다.
    """
    model = joblib.load(model_path)
    transformer = FeatureTransformer.load(transformer_path)

    # 메인 모델(3단계)은 feature_cols_3(segment_* 포함) 기준으로 학습되었으므로
    # 추론도 반드시 같은 피처 집합을 사용해야 한다.
    transformed = transformer.transform(df_raw)
    X = transformed["X_tree_3"]

    result = transformed["df_with_labels"].copy()
    result["이탈확률"] = model.predict_proba(X)[:, 1]
    return result


def resolve_input_path(requested_path: str, was_explicitly_set: bool) -> tuple[Path, bool]:
    """
    추론 대상 경로를 결정한다.

    Returns
    -------
    path : 실제로 사용할 경로
    is_demo : True면 신규 데이터가 없어 원본 데이터로 대체한 "데모 모드"
    """
    path = Path(requested_path)
    if path.exists():
        return path, False

    if was_explicitly_set:
        # 사용자가 --data 로 명시한 경로인데 없으면, 조용히 대체하지 않고 바로 알려준다.
        raise FileNotFoundError(
            f"지정한 추론 대상 파일이 없습니다: {path}\n"
            f"경로를 다시 확인해주세요."
        )

    if DEMO_FALLBACK_DATA_PATH.exists():
        return DEMO_FALLBACK_DATA_PATH, True

    raise FileNotFoundError(
        f"추론 대상 파일이 없고, 데모용 원본 데이터({DEMO_FALLBACK_DATA_PATH})도 없습니다.\n"
        f"새 고객 데이터를 {DEFAULT_NEW_DATA_PATH} 에 두거나, --data 로 경로를 지정하세요."
    )


def main(csv_path: str, model_path: str, transformer_path: str, output_path: str, is_demo: bool):
    if is_demo:
        print(
            "[데모 모드] data/new_customers.csv 가 없어 분석A/B/Q가 쓴 원본 데이터로 "
            "동작을 확인합니다.\n"
            "            실제 신규 고객 데이터가 준비되면 data/new_customers.csv 에 "
            "두고 다시 실행하세요.\n"
        )
    if not Path(model_path).exists() or not Path(transformer_path).exists():
        print(
            f"[안내] 학습된 모델이 없습니다: {model_path}\n"
            f"       먼저 train.py 를 실행해 모델을 만들어두세요."
        )
        raise SystemExit(1)

    print(f"[1/2] 추론 대상 데이터 로드: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"      {len(df_raw)}건")

    print("[2/2] 저장된 모델/변환규칙으로 즉시 추론 (재학습 없음)")
    result = predict(df_raw, model_path, transformer_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"      이탈확률 평균: {result['이탈확률'].mean():.4f}")
    print(f"      저장 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="저장된 모델로 즉시 추론 (학습 없음)")
    parser.add_argument(
        "--data", default=None,
        help=f"추론 대상 고객 CSV 경로 (기본값: {DEFAULT_NEW_DATA_PATH}, "
             f"없으면 원본 데이터로 데모 실행)",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH),
        help="model.pkl 경로 (기본값: outputs/latest/model.pkl)",
    )
    parser.add_argument(
        "--transformer", default=str(DEFAULT_TRANSFORMER_PATH),
        help="feature_transformer.pkl 경로 (기본값: outputs/latest/feature_transformer.pkl)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="결과 저장 경로")
    args = parser.parse_args()

    was_explicitly_set = args.data is not None
    requested = args.data if was_explicitly_set else str(DEFAULT_NEW_DATA_PATH)

    try:
        resolved_path, is_demo = resolve_input_path(requested, was_explicitly_set)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        raise SystemExit(1)

    print(f"[안내] 추론 대상: {resolved_path}\n")
    main(str(resolved_path), args.model, args.transformer, args.output, is_demo)
