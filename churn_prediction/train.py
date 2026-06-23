"""
churn_prediction/train.py

segment_discovery 가 만든 segment_rules.json 을 읽어 1·2·3단계 예측모델을
학습/평가하고, 모델과 FeatureTransformer를 저장한다.

⚠️ 이 스크립트는 "재학습"용이다 - 새 고객 데이터(신규 가입/이탈 확정)가
누적되었을 때만 실행한다. 단순히 기존 고객의 최신 위험도 점수만 보고 싶다면
predict.py(추론 전용, 학습 없이 즉시 실행)를 쓸 것 - 재학습보다 훨씬 빠르다.

이 패키지는 segment_discovery 의 내부 코드를 알 필요가 없다 - 오직
segment_rules.json 파일(인터페이스)만 의존한다.

실행 주기: segment_discovery/app.py 와 같은 묶음(재학습 묶음) - 표본이 충분히
쌓였을 때(또는 패턴 변화가 의심될 때) 둘을 함께 실행하는 게 일관적이다. 표본
부족으로 일부 범주가 통째로 빠지는 위험이 있으므로, "신규 데이터만"으로
재학습하지 말고 항상 누적 전체 데이터로 학습할 것. (predict.py 는 이 묶음과
무관하게 독립적으로 자유로운 주기 - 학습이 없어 데이터 양과 무관하게 안정적)

[실무 구조] 결과는 outputs/versions/{timestamp}/ 에 버전별로 보관되고,
outputs/latest/ 에는 그 중 가장 최근 버전이 그대로 복사된다.
predict.py 는 항상 outputs/latest/ 만 본다 - 과거 버전이 필요하면
outputs/versions/ 에서 직접 꺼내 쓸 것. 매 실행마다 outputs/run_history.csv
에 한 줄씩(시각, 입력 데이터, 행수, 성능지표 요약)이 누적되어, "언제 누가
어떤 데이터로 재학습했는지" 추적할 수 있다.

사용법:
    cd churn_prediction
    python train.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
                     --rules ../segment_discovery/outputs/segment_rules.json
"""
import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from logging_setup import get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))

from evaluate import (  # noqa: E402
    check_segment_label_incremental_contribution,
    compare_stage_metrics,
    compute_shap_importance,
    compute_xgboost_feature_importance,
    compute_xgboost_root_node_distribution,
    estimate_fn_cost,
    summarize_segment_feature_ranking,
)
from feature_engineering import load_segment_rules, prepare_model_inputs  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402
from train_models import train_all_stages  # noqa: E402

logger = get_logger("train")

OUTPUT_DIR = Path(__file__).parent / "outputs"
VERSIONS_DIR = OUTPUT_DIR / "versions"
LATEST_DIR = OUTPUT_DIR / "latest"
RUN_HISTORY_PATH = OUTPUT_DIR / "run_history.csv"

DEFAULT_DATA_PATH = config.DEFAULT_DATA_PATH
DEFAULT_RULES_PATH = config.SEGMENT_RULES_PATH


def append_run_history(row: dict) -> None:
    """매 실행마다 한 줄씩 누적 기록 - 언제 누가 어떤 데이터로 재학습했는지 추적용"""
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RUN_HISTORY_PATH.exists()
    with open(RUN_HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(csv_path: str, rules_path: str):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = VERSIONS_DIR / run_timestamp

    print(f"[실행 ID] {run_timestamp}")
    print("[1/6] 전처리 (Train/Test 분할)")
    df_train, df_test = run_preprocessing(csv_path)
    print(f"      train={len(df_train)}건, test={len(df_test)}건")

    print("[2/6] segment_rules.json 로드 및 피처 변환규칙 fit (feature_cols_12 / _3 분기)")
    rules = load_segment_rules(rules_path)
    inputs, transformer = prepare_model_inputs(df_train, df_test, rules)
    print(f"      feature_cols_12: {len(inputs['feature_cols_12'])}개 (segment_* 제외)")
    print(f"      feature_cols_3 : {len(inputs['feature_cols_3'])}개 (segment_* 포함)")

    print("[3/6] 1·2·3단계 + 보조 MLP 학습")
    stage_results, xgb_search_result = train_all_stages(inputs)
    print(f"      XGBoost 하이퍼파라미터 자동탐색 결과: {xgb_search_result.best_params} "
          f"(2단계에서 GridSearchCV로 탐색, 3단계가 그대로 물려받음)")
    for key, result in stage_results.items():
        print(f"      {result.name} 학습 완료 (피처 {len(result.feature_cols)}개)")

    print("[4/6] 평가지표 비교 (각 모델별 권장 임계값으로 평가 - 0.5 고정 아님)")
    metrics_df = compare_stage_metrics(stage_results, inputs)
    print(metrics_df.to_string())

    contribution = check_segment_label_incremental_contribution(
        metrics_df.loc["2단계_XGBoost"].to_dict(),
        metrics_df.loc["3단계_XGBoost_세그먼트포함"].to_dict(),
    )
    print(f"\n      세그먼트 라벨 증분기여: F1 {contribution['f1_increment']:+.4f}, "
          f"AUC {contribution['auc_increment']:+.4f}")
    print(f"      ※ {contribution['note']}")

    # ⚠️ [실측 확인] "segment_*가 통계적으로 검증된 랜드마크 피처라 모델의
    # 최상위 분기 기준으로 선택될 것"이라는 가설을 feature importance(gain)·
    # SHAP·실제 트리 루트노드 분포 세 가지 독립적인 방법으로 직접 검증한다.
    # 결과가 일치하면(=segment 영향력이 낮다는 결론이 재현되면), 이는 측정
    # 방식에 따른 우연이 아니라는 교차검증이 된다 - 위 contribution(F1/AUC
    # 증분)과 같은 결론을 다른 각도에서 다시 확인하는 단계.
    print("\n      [세그먼트 피처 영향력 교차검증 — gain / SHAP / 트리 루트노드]")
    stage3_model = stage_results["stage3"].model
    gain_importance_full = compute_xgboost_feature_importance(
        stage3_model, top_n=len(stage3_model.feature_names_in_)
    )
    shap_importance_full = compute_shap_importance(
        stage3_model, inputs["X_test_tree_3"], top_n=len(stage3_model.feature_names_in_)
    )
    segment_ranking = summarize_segment_feature_ranking(
        gain_importance_full, shap_importance_full, list(stage3_model.feature_names_in_),
    )
    print(segment_ranking.to_string(index=False))

    root_node_dist = compute_xgboost_root_node_distribution(stage3_model)
    segment_root_count = sum(
        count for feat, count in root_node_dist.items() if feat.startswith("segment_")
    )
    print(f"      전체 {root_node_dist.sum()}개 트리 중 segment_*가 루트 노드로 "
          f"선택된 횟수: {segment_root_count}개")
    print("      ※ 영향력이 낮게 나와도 \"segment가 무의미하다\"는 뜻이 아님 - "
          "분석A(세그먼트단독AUC+순열검정)가 메인 증거, 위 결과는 예측모델 "
          "차원의 보조적 실용성 지표일 뿐")

    # ⚠️ [수정] 예전에는 여기서 find_recall_drop_threshold를 또 호출해서
    # 메타데이터에만 기록하고, compare_stage_metrics는 별개로 항상 0.5를 썼다.
    # 이제 compare_stage_metrics가 이미 메인 모델(3단계)의 권장 임계값으로
    # 평가했으므로, 그 결과를 그대로 가져온다 - 중복 계산 제거 + 평가에 실제 반영.
    threshold = metrics_df.loc["3단계_XGBoost_세그먼트포함", "threshold"]
    print(f"\n      메인 모델(3단계) 권장 분류 임계값(Recall 급락 직전): {threshold:.4f}")

    fn_cost = estimate_fn_cost(inputs["df_train_raw"])
    print(f"      FN 비용 근사 추정: {fn_cost:.0f} (참고용, FP비용은 추정 불가)")

    print(f"\n[5/6] 버전 저장: outputs/versions/{run_timestamp}/")
    version_dir.mkdir(parents=True, exist_ok=True)

    # 세그먼트 피처 영향력 교차검증 결과를 버전 디렉토리에 함께 저장
    # (보고서/슬라이드 작성 시 재계산 없이 바로 인용 가능하도록)
    segment_ranking.to_csv(version_dir / "segment_feature_ranking.csv", index=False)

    # 메인 모델(3단계) + 변환규칙(FeatureTransformer) 저장
    # ⚠️ FeatureTransformer가 없으면 predict.py가 학습 때와 다른 방식으로
    # 인코딩하게 되어 모델이 기대하는 피처 형식과 어긋날 수 있다.
    joblib.dump(stage_results["stage3"].model, version_dir / "model.pkl")
    transformer.save(version_dir / "feature_transformer.pkl")

    df_test_result = inputs["df_test_raw"].copy()
    df_test_result["이탈확률"] = stage_results["stage3"].model.predict_proba(
        inputs["X_test_tree_3"]
    )[:, 1]
    df_test_result.to_csv(version_dir / "predictions.csv", index=False)
    metrics_df.to_csv(version_dir / "stage_metrics.csv")

    main_metrics = metrics_df.loc["3단계_XGBoost_세그먼트포함"]
    metadata = {
        "run_timestamp": run_timestamp,
        "data_path": str(Path(csv_path).resolve()),
        "rules_path": str(Path(rules_path).resolve()),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "xgboost_best_params": xgb_search_result.best_params,
        "main_model_f2": round(float(main_metrics["f2"]), 4),
        "main_model_auc": round(float(main_metrics["model_auc"]), 4),
        "recommended_threshold": round(threshold, 4),
        "fn_cost_estimate": round(fn_cost, 0),
    }
    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[6/6] latest 갱신: outputs/latest/ <- versions/{run_timestamp}/")
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(version_dir, LATEST_DIR)

    append_run_history({
        "run_timestamp": run_timestamp,
        "data_path": metadata["data_path"],
        "n_train": metadata["n_train"],
        "n_test": metadata["n_test"],
        "main_model_f2": metadata["main_model_f2"],
        "main_model_auc": metadata["main_model_auc"],
        "f1_increment_vs_stage2": round(contribution["f1_increment"], 4),
    })

    print("      model.pkl, feature_transformer.pkl, predictions.csv, "
          "stage_metrics.csv, metadata.json")
    print(f"      -> outputs/versions/{run_timestamp}/  (이번 버전 보관)")
    print("      -> outputs/latest/                    (predict.py 가 보는 위치)")
    print("      -> outputs/run_history.csv             (실행 기록 누적)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="예측모델 1·2·3단계 재학습")
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH),
        help=f"원본 CSV 경로 (재학습용 데이터, 기본값: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--rules", default=str(DEFAULT_RULES_PATH),
        help=f"segment_discovery/outputs/segment_rules.json 경로 (기본값: {DEFAULT_RULES_PATH})",
    )
    args = parser.parse_args()
    print(f"[안내] 데이터: {args.data}\n       규칙: {args.rules}\n")
    logger.info(f"재학습 시작 - data={args.data}, rules={args.rules}")
    try:
        main(args.data, args.rules)
        logger.info("재학습 성공")
    except Exception:
        logger.exception("재학습 실패")  # 콘솔에는 보통의 traceback이, 파일에는 같은 내용이 남음
        raise
