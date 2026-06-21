"""
webapp/utils.py

대시보드 전 페이지가 공유하는 로딩/전처리 유틸리티.

⚠️ 이 모듈은 segment_discovery / churn_prediction 의 "산출물 파일"만 읽는다
(코드를 import해서 재계산하지 않음) - README의 "인터페이스는 코드가 아니라
결과 파일" 원칙을 webapp에서도 그대로 따른다.

읽는 파일:
- segment_discovery/outputs/segment_rules.json  (경계, 위험속성, 분석A/B 결과)
- churn_prediction/outputs/latest/model.pkl, feature_transformer.pkl
- data/WA_FnUseC_TelcoCustomerChurn.csv          (거시 차트용 원본 데이터)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "churn_prediction" / "src"))

import config  # noqa: E402
from data_loader import clean_raw_data  # noqa: E402
from feature_engineering import FeatureTransformer  # noqa: E402

SEGMENT_RULES_PATH = config.SEGMENT_RULES_PATH
LATEST_DIR = PROJECT_ROOT / "churn_prediction" / "outputs" / "latest"
MODEL_PATH = LATEST_DIR / "model.pkl"
TRANSFORMER_PATH = LATEST_DIR / "feature_transformer.pkl"
METADATA_PATH = LATEST_DIR / "metadata.json"
DATA_PATH = config.DEFAULT_DATA_PATH

# 입력 폼/검증에서 공통으로 쓰는 선택지 (shared/columns.py 의 CATEGORICAL_COLS와 1:1 대응)
FORM_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

REQUIRED_RAW_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# ---------------------------------------------------------------------------
# 도메인 지식 레이어 (⚠️ 모델 산출물이 아니라 사람이 정한 비즈니스 규칙)
#
# 아래 두 딕셔너리는 segment_rules.json/model.pkl에서 나온 게 아니라, 분석B가
# 찾아낸 "위험속성"(risk_attribute_values) 5개에 대해 사람이 매핑한 대응
# 프로모션/설명 문구다. 데이터가 통계적으로 검증한 것은 "이 속성이 이탈과
# 관련 있다"까지이고, "그래서 이 프로모션을 줘야 한다"는 마케팅 도메인
# 지식이므로 둘을 분리해서 대시보드에 명시적으로 라벨링한다.
# ---------------------------------------------------------------------------

FEATURE_LABELS_KO = {
    "gender": "성별", "SeniorCitizen": "고령(65세+) 여부", "Partner": "배우자 유무",
    "Dependents": "부양가족 유무", "tenure": "가입 후 경과월", "PhoneService": "전화 서비스",
    "PaperlessBilling": "전자청구서", "MonthlyCharges": "월 요금", "TotalCharges": "누적 청구액",
    "MultipleLines": "복수 회선", "InternetService": "인터넷 서비스", "OnlineSecurity": "온라인 보안",
    "OnlineBackup": "온라인 백업", "DeviceProtection": "기기 보호", "TechSupport": "기술 지원",
    "StreamingTV": "스트리밍 TV", "StreamingMovies": "스트리밍 영화", "Contract": "계약 형태",
    "PaymentMethod": "결제 수단", "segment": "가입기간 세그먼트",
}

# 위험속성(분석B/서브트랙Q의 risk_attribute_values, segment_rules.json 기준)이
# 해당 값일 때, 마케터가 바로 꺼낼 수 있는 프로모션 카드.
#
# ⚠️ 할인폭/혜택 같은 구체적 숫자는 placeholder([ ] 표기)로 둔다. 행동
# 카테고리(자동이체 전환 유도, 체험권 제공, 캐시백 등)는 위험속성과의
# 관련성에 근거해 구체적으로 제시하지만, 숫자 자체는 원가 구조·마진율에
# 대한 근거가 없어 실제 운영 시 담당팀이 채워야 한다 (PENDING_DECISIONS.md
# "프로모션 매핑" 확정 항목 참고).
PROMOTION_PLAYBOOK = {
    ("Contract", "Month-to-month"): {
        "원인설명": "월단위 계약이라 이탈 장벽이 낮습니다",
        "프로모션카드": "1년 약정 전환 시 [할인폭] 요금 할인 제안",
    },
    ("PaymentMethod", "Electronic check"): {
        "원인설명": "전자수표 결제는 자동이체 대비 결제 마찰·해지 전환이 잦은 결제수단입니다",
        "프로모션카드": "자동이체(카드/계좌) 전환 시 월 [할인액] 할인 쿠폰",
    },
    ("InternetService", "Fiber optic"): {
        "원인설명": "광인터넷 고가 요금제 사용자로, 가격 민감도가 높은 군입니다",
        "프로모션카드": "광인터넷 전용 [캐시백 금액] 요금 캐시백 또는 결합상품 제안",
    },
    ("OnlineSecurity", "No"): {
        "원인설명": "온라인 보안 부가서비스 미가입 상태입니다",
        "프로모션카드": "보안 부가서비스 [기간] 무료 체험권 제공",
    },
    ("TechSupport", "No"): {
        "원인설명": "기술 지원 부가서비스 미가입 상태입니다",
        "프로모션카드": "기술 지원 패키지 [기간] 무료 체험권 제공",
    },
}


def humanize_feature(raw_col: str) -> str:
    """원-핫 인코딩된 컬럼명(예: 'Contract_Month-to-month')을 사람이 읽는 한국어로 변환"""
    if raw_col in FEATURE_LABELS_KO:
        return FEATURE_LABELS_KO[raw_col]
    for base_col, ko_label in FEATURE_LABELS_KO.items():
        prefix = f"{base_col}_"
        if raw_col.startswith(prefix):
            value = raw_col[len(prefix):]
            return f"{ko_label}: {value}"
    if raw_col.startswith("segment_"):
        return f"가입기간 세그먼트 {raw_col.split('_')[-1]}"
    return raw_col


# ---------------------------------------------------------------------------
# 산출물 존재 여부 점검 (없으면 화면에 안내만 하고 멈춤)
# ---------------------------------------------------------------------------

def assert_artifacts_ready() -> None:
    missing = []
    if not SEGMENT_RULES_PATH.exists():
        missing.append(f"- 세그먼트 규칙: `{SEGMENT_RULES_PATH}`\n  → `cd segment_discovery && python app.py` 먼저 실행")
    if not MODEL_PATH.exists() or not TRANSFORMER_PATH.exists():
        missing.append(f"- 학습된 모델: `{LATEST_DIR}`\n  → `cd churn_prediction && python train.py` 먼저 실행")
    if not DATA_PATH.exists():
        missing.append(f"- 원본 데이터: `{DATA_PATH}`")

    if missing:
        st.error(
            "대시보드를 표시하려면 먼저 파이프라인 산출물이 필요합니다.\n\n"
            + "\n\n".join(missing)
        )
        st.stop()


# ---------------------------------------------------------------------------
# 캐시된 로더 (Streamlit 세션 내내 한 번만 로드)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_segment_rules() -> dict:
    with open(SEGMENT_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_transformer() -> FeatureTransformer:
    return FeatureTransformer.load(TRANSFORMER_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = clean_raw_data(df)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


# ---------------------------------------------------------------------------
# 세그먼트 라벨링 (predict.py / feature_engineering.py 와 동일 규칙)
# ---------------------------------------------------------------------------

def assign_segment(df: pd.DataFrame, boundaries: list[float]) -> pd.DataFrame:
    df = df.copy()
    upper = max(df["tenure"].max(), (boundaries[-1] if boundaries else 0)) + 1
    bins = [-1] + list(boundaries) + [upper]
    labels = list(range(len(bins) - 1))
    df["segment"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)
    return df


def segment_label_text(seg_id: int, boundaries: list[float], max_tenure: int) -> str:
    """세그먼트 번호 -> 'tenure 0~10개월'같은 사람이 읽는 구간 텍스트"""
    edges = [0] + [int(round(b)) for b in boundaries] + [max_tenure]
    lo = edges[seg_id]
    hi = edges[seg_id + 1]
    if seg_id == 0:
        return f"{lo}~{hi}개월"
    return f"{lo + 1}~{hi}개월"


def risk_count_series(df: pd.DataFrame, risk_attribute_values: dict[str, str]) -> pd.Series:
    masks = pd.DataFrame(index=df.index)
    for col, risky_value in risk_attribute_values.items():
        masks[col] = (df[col] == risky_value).astype(int)
    return masks.sum(axis=1)


# ---------------------------------------------------------------------------
# 추론 (predict.py 의 predict() 와 동일 로직 - 모델/트랜스포머만 재사용)
# ---------------------------------------------------------------------------

def predict_churn(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 컬럼 형태(customerID, tenure, Contract, ... )의 DataFrame을 받아
    '이탈확률' 컬럼이 추가된 결과를 반환한다. 1행이든 여러 행이든 동일하게 동작.
    """
    model = load_model()
    transformer = load_transformer()
    transformed = transformer.transform(df_raw)
    X = transformed["X_tree_3"]
    result = transformed["df_with_labels"].copy()
    result["이탈확률"] = model.predict_proba(X)[:, 1]
    return result


def validate_uploaded_df(df: pd.DataFrame) -> list[str]:
    """업로드된 CSV의 필수 컬럼/값 점검. 문제 메시지 리스트(없으면 빈 리스트) 반환."""
    errors = []
    missing_cols = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"필수 컬럼 누락: {', '.join(missing_cols)}")
        return errors  # 컬럼이 없으면 이후 검증 무의미

    if df["customerID"].duplicated().any():
        dup_n = int(df["customerID"].duplicated().sum())
        errors.append(f"customerID 중복 {dup_n}건 (각 행은 고유해야 합니다)")

    if not pd.api.types.is_numeric_dtype(df["tenure"]):
        errors.append("tenure 컬럼은 숫자(개월 수)여야 합니다")
    elif (df["tenure"] < 0).any():
        errors.append("tenure에 음수 값이 있습니다")

    if not pd.api.types.is_numeric_dtype(df["MonthlyCharges"]):
        errors.append("MonthlyCharges 컬럼은 숫자여야 합니다")

    for col, allowed in FORM_OPTIONS.items():
        if col not in df.columns:
            continue
        if col in ("SeniorCitizen",):
            continue  # 0/1 숫자는 read_csv로 자동 처리됨
        bad_values = set(df[col].dropna().unique()) - set(allowed)
        if bad_values:
            errors.append(f"{col}에 허용되지 않은 값: {sorted(bad_values)}")

    return errors


def risk_tier(prob: float) -> tuple[str, str]:
    """이탈확률 -> (등급 라벨, 색상 hex). 임계값은 metadata의 recommended_threshold 참고."""
    if prob >= 0.6:
        return "고위험", "#D64545"
    if prob >= 0.32:
        return "중위험", "#E0A03C"
    return "저위험", "#3C8C5C"


# ---------------------------------------------------------------------------
# 비용 효율성: Expected Loss = MonthlyCharges × 이탈확률
#
# ⚠️ "1개월치 예상 손실" 근사치다. 잔여 생애주기(tenure 추정)를 곱하는 LTV
# 버전도 가능하지만, 잔여 개월수를 추정할 근거 데이터가 없어 과대해석을
# 피하기 위해 가장 단순하고 방어 가능한 1개월 단위로 고정한다.
# evaluate.py의 estimate_fn_cost(전체 모델 단위 FN비용 추정)와는 다른
# 개념(개인별 우선순위화용)이므로 용어/위치를 분리해서 보여준다.
#
# 화면에 표시할 때는 항상 "1개월 근사치이며 LTV 반영 시 더 클 수 있다"는
# 캡션을 함께 노출한다(02/03 페이지). 정확한 LTV 계산(생존분석 기반)은
# 지금 범위 밖의 향후 과제 — PENDING_DECISIONS.md "Expected Loss" 확정
# 항목 참고.
# ---------------------------------------------------------------------------

def add_expected_loss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["예상손실액"] = df["이탈확률"] * df["MonthlyCharges"]
    return df


# ---------------------------------------------------------------------------
# 실행 가능성: 위험속성 기반 프로모션 카드 추천 (도메인 규칙, PROMOTION_PLAYBOOK 참고)
# ---------------------------------------------------------------------------

def recommend_promotions(row: pd.Series, risk_attribute_values: dict[str, str]) -> list[dict]:
    """
    한 고객 행(원본 컬럼 보유)에 대해, 해당하는 위험속성마다 프로모션 카드를 반환.
    risk_count가 0이면 빈 리스트.
    """
    cards = []
    for col, risky_value in risk_attribute_values.items():
        if col in row and row[col] == risky_value:
            playbook = PROMOTION_PLAYBOOK.get((col, risky_value))
            if playbook:
                cards.append({"속성": col, "값": risky_value, **playbook})
    return cards


# ---------------------------------------------------------------------------
# SHAP 기반 개별 예측 설명 (메인 모델이 XGBoost 이므로 TreeExplainer 사용)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_shap_explainer():
    import shap
    return shap.TreeExplainer(load_model())


def explain_prediction(df_raw: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    df_raw(원본 컬럼 형태)를 변환 후 SHAP 값을 계산해, 이탈확률을 가장 크게
    밀어올리는/낮추는 변수 상위 top_n개를 반환한다. (다건이면 행별 평균 |SHAP|)
    단건 입력(개인 예측 페이지)에 적합 - 다건 입력의 행별 설명은
    explain_prediction_per_row()를 사용할 것.
    """
    transformer = load_transformer()
    transformed = transformer.transform(df_raw)
    X = transformed["X_tree_3"]

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(X)

    mean_abs = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)
    mean_signed = pd.Series(shap_values.mean(axis=0), index=X.columns)

    out = pd.DataFrame({
        "변수": [humanize_feature(c) for c in mean_abs.index],
        "영향도(|SHAP| 평균)": mean_abs.values,
        "방향": ["이탈 위험 ↑" if v > 0 else "이탈 위험 ↓" for v in mean_signed.values],
    }).sort_values("영향도(|SHAP| 평균)", ascending=False).head(top_n)
    return out.reset_index(drop=True)


def explain_prediction_per_row(df_raw: pd.DataFrame, top_n: int = 1) -> list[str]:
    """
    다건 입력(일괄 예측 페이지)용 - 행마다 이탈확률을 가장 크게 밀어올린
    변수 top_n개를 사람이 읽는 한국어 문자열 리스트로 반환 (행 순서 보존).
    예: ["계약 형태: Month-to-month", ...]
    """
    transformer = load_transformer()
    transformed = transformer.transform(df_raw)
    X = transformed["X_tree_3"]

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(X)  # shape: (n_rows, n_features)

    results = []
    for i in range(shap_values.shape[0]):
        row_shap = pd.Series(shap_values[i], index=X.columns)
        top_positive = row_shap.sort_values(ascending=False).head(top_n)
        labels = [humanize_feature(c) for c in top_positive.index]
        results.append(" · ".join(labels))
    return results
