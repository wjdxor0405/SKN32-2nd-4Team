"""
segment_discovery/app.py

분석A → 분석B → 서브트랙Q 를 순서대로 실행하고, 그 결과를
segment_rules.json 으로 저장한다.

이 결과 파일은 churn_prediction 패키지가 가져다 쓰는 유일한 인터페이스다.
(churn_prediction 은 이 패키지의 내부 코드를 알 필요가 없다)

실행 주기: 표본이 충분히 쌓였을 때(또는 패턴 변화가 의심될 때) - churn_prediction의
train.py(재학습)와 같은 묶음으로 함께 실행되는 게 일관적이다. train.py가 쓰는
segment_rules.json 자체가 이 스크립트의 산출물이므로, 분석A/B/Q가 갱신될 때
모델도 같은 누적 데이터로 재학습해야 한다. (predict.py는 이 묶음과 무관하게
독립적으로 자유로운 주기 - 기획_메모.md 운영 설계 참조)

사용법:
    cd segment_discovery
    python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis_a import run_analysis_a, make_segment_column  # noqa: E402
from analysis_b import extract_risk_attribute_values, run_analysis_b  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402
from subtrack_q import run_subtrack_q  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "outputs" / "segment_rules.json"


def main(csv_path: str, output_path: Path = OUTPUT_PATH) -> dict:
    print("[1/4] 전처리 (df_train만 사용 - 누수 방지)")
    df_train, df_test = run_preprocessing(csv_path)
    print(f"      train={len(df_train)}건, test={len(df_test)}건")

    print("[2/4] 분석 A — ①②③ 검증 통과까지 반복")
    result_a = run_analysis_a(df_train)
    print(f"      경계: {result_a['boundaries']} (반복 {result_a['n_iterations']}회)")
    print(f"      RF 보조검증 트리개수(데이터가 직접 찾음): {result_a['rf_n_estimators_used']}개")
    print(f"      부트스트랩 반복횟수(순차적 조기중단으로 데이터가 직접 찾음): {result_a['n_bootstrap_used']}회")
    print(f"      순열검정 반복횟수(순차적 조기중단으로 데이터가 직접 찾음): {result_a['n_permutations_used']}회")
    print(f"      세그먼트단독 AUC: {result_a['segment_only_auc']:.4f}, "
          f"p={result_a['p_value']:.4f}, "
          f"신뢰구간=[{result_a['ci_low']:.4f}, {result_a['ci_high']:.4f}]")
    if "warning" in result_a:
        print(f"      ⚠️ {result_a['warning']}")

    df_train = df_train.copy()
    df_train["segment"] = make_segment_column(df_train, result_a["boundaries"])

    print("[3/4] 분석 B — 세그먼트별 Ⓐ→Ⓑ→Ⓒ")
    result_b = run_analysis_b(df_train)
    print(result_b.to_string(index=False))

    print("[4/4] 서브 트랙 Q — risk_count(메인) + K-means(보조)")
    # ⚠️ [정정] 예전에는 risk_attribute_values를 사람이 직접 하드코딩했는데,
    # 그중 PaymentMethod·TechSupport는 분석B의 top_attributes에 단 한 번도
    # 등장한 적이 없는 속성이었다 - 메모 4.1-B("분석B의 검증을 그대로 물려받음,
    # 새로 임의 선정하지 않음")라는 원칙을 코드가 어기고 있던 정합성 버그.
    # extract_risk_attribute_values가 분석B 결과(result_b)에서 범주형
    # 위험속성의 위험값만 자동으로 추출한다 - 연속형(MonthlyCharges)은 "위험값"
    # 개념이 없어 제외됨(analysis_b.extract_risk_attribute_values 참조).
    risk_attribute_values = extract_risk_attribute_values(df_train, result_b)
    result_q = run_subtrack_q(df_train, risk_attribute_values)
    print(f"      risk_count단독 AUC: {result_q['risk_count_only_auc']:.4f}, "
          f"p={result_q['p_value']:.4f} (n_permutations_used={result_q['n_permutations_used']})")
    print(f"      최고위험(risk_count={result_q['top_risk_count_value']}) "
          f"신뢰구간=[{result_q['ci_low']:.4f}, {result_q['ci_high']:.4f}] "
          f"(n_bootstrap_used={result_q['n_bootstrap_used']})")

    rules = {
        "analysis_a": {
            "boundaries": result_a["boundaries"],
            "alpha": result_a["alpha"],
            "rf_n_estimators_used": result_a["rf_n_estimators_used"],
            "n_bootstrap_used": result_a["n_bootstrap_used"],
            "n_permutations_used": result_a["n_permutations_used"],
            "segment_only_auc": result_a["segment_only_auc"],
            "p_value": result_a["p_value"],
            "ci_low": result_a["ci_low"],
            "ci_high": result_a["ci_high"],
            "n_iterations": result_a["n_iterations"],
        },
        "analysis_b": result_b.to_dict(orient="records"),
        "subtrack_q": {
            "risk_attribute_values": risk_attribute_values,
            "risk_count_only_auc": result_q["risk_count_only_auc"],
            "p_value": result_q["p_value"],
            "n_permutations_used": result_q["n_permutations_used"],
            "top_risk_count_value": result_q["top_risk_count_value"],
            "ci_low": result_q["ci_low"],
            "ci_high": result_q["ci_high"],
            "n_bootstrap_used": result_q["n_bootstrap_used"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {output_path}")
    return rules


DEFAULT_DATA_PATH = config.DEFAULT_DATA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분석A/B/서브트랙Q 실행")
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH),
        help=f"원본 CSV 경로 (기본값: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="결과 json 저장 경로")
    args = parser.parse_args()
    # VSCode 실행버튼처럼 인자 없이 실행될 때를 위한 안내
    print(f"[안내] 데이터 경로: {args.data}\n")
    main(args.data, Path(args.output))
