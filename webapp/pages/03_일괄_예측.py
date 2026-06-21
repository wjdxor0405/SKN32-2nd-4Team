"""
webapp/pages/03_일괄_예측.py

여러 명의 고객 데이터를 CSV로 한 번에 업로드해 이탈확률을 일괄 예측하고,
위험군 분포·세그먼트별 통계 등 분석 결과를 함께 보여준다.

predict.py와 동일하게 "한 행씩 독립적으로 변환/추론"하므로 업로드 행 수와
무관하게 안전하게 동작한다 (utils.predict_churn 재사용).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    REQUIRED_RAW_COLS,
    add_expected_loss,
    assert_artifacts_ready,
    explain_prediction_per_row,
    load_metadata,
    load_segment_rules,
    predict_churn,
    risk_tier,
    segment_label_text,
    validate_uploaded_df,
)

st.set_page_config(page_title="일괄 예측 | 이탈 예측", page_icon="📋", layout="wide")
assert_artifacts_ready()

st.title("📋 여러 고객 일괄 이탈율 예측")
st.caption("CSV 파일을 업로드하면 고객별 이탈확률과 함께 요약 분석을 보여줍니다.")

metadata = load_metadata()
threshold = metadata.get("recommended_threshold", 0.5)
rules = load_segment_rules()
boundaries = rules["analysis_a"]["boundaries"]

with st.expander("CSV 형식 안내", expanded=False):
    st.markdown(
        "다음 컬럼을 모두 포함해야 합니다 (원본 Telco 데이터와 동일한 컬럼 구조):"
    )
    st.code(", ".join(REQUIRED_RAW_COLS), language=None)
    sample = pd.DataFrame([{
        "customerID": "C-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5, "TotalCharges": 241.5,
    }])
    st.dataframe(sample, width='stretch', hide_index=True)
    st.download_button(
        "샘플 CSV 양식 다운로드", sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="sample_customers.csv", mime="text/csv",
    )

uploaded = st.file_uploader("고객 데이터 CSV 업로드", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    errors = validate_uploaded_df(df_raw)
    if errors:
        st.error("업로드한 파일에 문제가 있습니다:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.stop()

    st.success(f"{len(df_raw):,}명의 데이터를 확인했습니다. 예측을 진행합니다.")

    SHAP_ROW_LIMIT = 50_000  # 이 이상이면 핵심원인 계산을 생략 (안전장치, 실측 기반 결정은 PENDING_DECISIONS.md 참고)

    with st.spinner("일괄 예측 중..."):
        result = predict_churn(df_raw)
        result = add_expected_loss(result)

        skip_shap = len(df_raw) > SHAP_ROW_LIMIT
        if not skip_shap:
            result["핵심원인"] = explain_prediction_per_row(df_raw, top_n=1)
        else:
            result["핵심원인"] = None

    if skip_shap:
        st.warning(
            f"⚠️ 데이터가 너무 커({len(df_raw):,}건 > {SHAP_ROW_LIMIT:,}건) "
            "핵심원인 계산을 생략했습니다. 이탈확률·예상손실액 등 나머지 결과는 정상 제공됩니다."
        )

    result["위험등급"] = result["이탈확률"].apply(lambda p: risk_tier(p)[0])
    result["세그먼트_구간"] = result["segment"].apply(
        lambda s: segment_label_text(int(s), boundaries, int(result["tenure"].max()))
    )

    st.divider()
    st.markdown("### 요약")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("입력 고객 수", f"{len(result):,}명")
    k2.metric("평균 이탈확률", f"{result['이탈확률'].mean():.1%}")
    n_high = int((result["이탈확률"] >= threshold).sum())
    k3.metric(f"고위험 고객 (≥{threshold:.2f})", f"{n_high:,}명", f"{n_high/len(result):.1%}")
    k4.metric("총 예상 손실액 (1개월)", f"${result['예상손실액'].sum():,.0f}")

    st.divider()
    st.markdown("### 💰 한정된 예산의 비용 효율성 — Expected Loss 우선순위화")
    st.caption(
        "이탈확률 단순 정렬은 '월요금이 적은 저가 고객'을 먼저 보여줄 수 있습니다. "
        "예상손실액(이탈확률 × 월요금) 기준으로 보면, 확률은 다소 낮아도 월요금이 높아 "
        "잃었을 때 타격이 큰 고객이 위로 올라옵니다."
    )
    st.caption("📌 예상손실액은 1개월 근사치 · 약정 잔여기간(LTV) 반영 시 더 클 수 있음")

    budget_n = st.slider(
        "예산이 허락하는 리텐션 캠페인 대상 인원 (상위 N명)",
        min_value=5, max_value=min(200, len(result)), value=min(20, len(result)), step=5,
    )

    rank_basis = st.radio(
        "우선순위 기준", ["예상손실액 (권장)", "이탈확률만"], horizontal=True,
    )
    sort_col = "예상손실액" if rank_basis.startswith("예상손실액") else "이탈확률"
    top_n_df = result.sort_values(sort_col, ascending=False).head(budget_n)

    bc1, bc2 = st.columns(2)
    bc1.metric(
        f"상위 {budget_n}명 방어 가능 예상손실액",
        f"${top_n_df['예상손실액'].sum():,.0f}",
        help="이 고객들에게 리텐션 액션이 성공한다고 가정했을 때 방어 가능한 1개월 매출 손실 총합",
    )
    bc2.metric(
        f"상위 {budget_n}명 평균 월요금",
        f"${top_n_df['MonthlyCharges'].mean():,.0f}",
        help="이탈확률만으로 정렬했다면 평균 월요금이 더 낮은 고객군이 선택됐을 수 있습니다",
    )

    with st.expander(f"우선순위 비교: '{rank_basis}' 기준 상위 {budget_n}명"):
        compare_cols = ["customerID", "tenure", "Contract", "MonthlyCharges", "이탈확률", "예상손실액"]
        st.dataframe(
            top_n_df[compare_cols].style.format(
                {"이탈확률": "{:.1%}", "MonthlyCharges": "${:.1f}", "예상손실액": "${:.1f}"}
            ),
            width="stretch", hide_index=True,
        )

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### 이탈확률 분포")
        fig_hist = px.histogram(
            result, x="이탈확률", nbins=20, color_discrete_sequence=["#4C78A8"],
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#D64545",
                            annotation_text="권장 임계값")
        fig_hist.update_layout(xaxis_tickformat=".0%", height=380, margin=dict(t=20))
        st.plotly_chart(fig_hist, width='stretch')

    with chart_col2:
        st.markdown("#### 위험등급 비율")
        tier_counts = result["위험등급"].value_counts().reindex(["고위험", "중위험", "저위험"]).fillna(0)
        fig_pie = px.pie(
            names=tier_counts.index, values=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map={"고위험": "#D64545", "중위험": "#E0A03C", "저위험": "#3C8C5C"},
            hole=0.45,
        )
        fig_pie.update_layout(height=380, margin=dict(t=20))
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown("#### 세그먼트별 평균 이탈확률")
    seg_summary = (
        result.groupby("세그먼트_구간", sort=False)["이탈확률"]
        .agg(평균이탈확률="mean", 고객수="count")
        .reset_index()
    )
    fig_bar = px.bar(
        seg_summary, x="세그먼트_구간", y="평균이탈확률", text="고객수",
        color="평균이탈확률", color_continuous_scale="OrRd",
    )
    fig_bar.update_traces(texttemplate="%{text}명", textposition="outside")
    fig_bar.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(t=20))
    st.plotly_chart(fig_bar, width='stretch')

    st.markdown("#### Contract(계약 형태)별 평균 이탈확률")
    contract_summary = (
        result.groupby("Contract")["이탈확률"].mean().sort_values(ascending=False).reset_index()
    )
    fig_contract = px.bar(
        contract_summary, x="Contract", y="이탈확률", color="이탈확률", color_continuous_scale="OrRd",
    )
    fig_contract.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(t=20))
    st.plotly_chart(fig_contract, width='stretch')

    st.divider()
    st.markdown("### 고객별 상세 결과")

    tier_filter = st.multiselect(
        "위험등급 필터", options=["고위험", "중위험", "저위험"],
        default=["고위험", "중위험", "저위험"],
    )
    sort_basis = st.radio(
        "정렬 기준", ["이탈확률 높은 순", "예상손실액 높은 순"], horizontal=True, key="detail_sort",
    )
    sort_col2 = "이탈확률" if sort_basis.startswith("이탈확률") else "예상손실액"

    display_cols = [
        "customerID", "tenure", "세그먼트_구간", "Contract", "MonthlyCharges",
        "risk_count", "이탈확률", "예상손실액", "핵심원인", "위험등급",
    ]
    filtered = result[result["위험등급"].isin(tier_filter)]
    filtered = filtered.sort_values(sort_col2, ascending=False)

    st.dataframe(
        filtered[display_cols].style.format(
            {"이탈확률": "{:.1%}", "MonthlyCharges": "${:.1f}", "예상손실액": "${:.1f}"}
        ),
        width='stretch', hide_index=True, height=420,
    )
    if skip_shap:
        st.caption("핵심원인 컬럼은 데이터 규모로 인해 비어 있습니다 (위 안내 참고).")

    st.download_button(
        "전체 예측 결과 CSV 다운로드",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="churn_predictions.csv",
        mime="text/csv",
        width='stretch',
    )
else:
    st.info("CSV 파일을 업로드하면 예측 결과와 분석이 표시됩니다.")
