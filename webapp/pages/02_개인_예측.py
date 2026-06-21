"""
webapp/pages/02_개인_예측.py

한 사람의 정보를 입력하면 저장된 모델(churn_prediction/outputs/latest/)로
즉시 이탈확률을 예측한다. predict.py와 동일한 추론 로직(FeatureTransformer.transform
+ model.predict_proba)을 utils.predict_churn()을 통해 재사용한다.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    FORM_OPTIONS,
    add_expected_loss,
    assert_artifacts_ready,
    explain_prediction,
    load_metadata,
    load_segment_rules,
    predict_churn,
    recommend_promotions,
    risk_tier,
    segment_label_text,
)

st.set_page_config(page_title="개인 예측 | 이탈 예측", page_icon="🧑", layout="wide")
assert_artifacts_ready()

st.title("🧑 개인 고객 이탈 예측")
st.caption("고객 한 명의 정보를 입력하면 저장된 모델로 이탈확률을 즉시 계산합니다.")

metadata = load_metadata()
threshold = metadata.get("recommended_threshold", 0.5)

with st.form("single_customer_form"):
    st.markdown("#### 기본 정보")
    c1, c2, c3 = st.columns(3)
    customer_id = c1.text_input("고객 ID", value="NEW-0001")
    gender = c2.selectbox("성별", FORM_OPTIONS["gender"])
    senior = c3.selectbox("65세 이상 여부", FORM_OPTIONS["SeniorCitizen"], format_func=lambda x: "예" if x == 1 else "아니오")

    c4, c5, c6 = st.columns(3)
    partner = c4.selectbox("배우자 여부", FORM_OPTIONS["Partner"])
    dependents = c5.selectbox("부양가족 여부", FORM_OPTIONS["Dependents"])
    tenure = c6.number_input("가입 후 경과월 (tenure)", min_value=0, max_value=100, value=5)

    st.markdown("#### 서비스 가입 현황")
    c7, c8, c9 = st.columns(3)
    phone_service = c7.selectbox("전화 서비스", FORM_OPTIONS["PhoneService"])
    multiple_lines = c8.selectbox("복수 회선", FORM_OPTIONS["MultipleLines"])
    internet_service = c9.selectbox("인터넷 서비스", FORM_OPTIONS["InternetService"])

    c10, c11, c12 = st.columns(3)
    online_security = c10.selectbox("온라인 보안", FORM_OPTIONS["OnlineSecurity"])
    online_backup = c11.selectbox("온라인 백업", FORM_OPTIONS["OnlineBackup"])
    device_protection = c12.selectbox("기기 보호", FORM_OPTIONS["DeviceProtection"])

    c13, c14, c15 = st.columns(3)
    tech_support = c13.selectbox("기술 지원", FORM_OPTIONS["TechSupport"])
    streaming_tv = c14.selectbox("스트리밍 TV", FORM_OPTIONS["StreamingTV"])
    streaming_movies = c15.selectbox("스트리밍 영화", FORM_OPTIONS["StreamingMovies"])

    st.markdown("#### 계약 및 결제")
    c16, c17, c18 = st.columns(3)
    contract = c16.selectbox("계약 형태", FORM_OPTIONS["Contract"])
    paperless = c17.selectbox("전자청구서 여부", FORM_OPTIONS["PaperlessBilling"])
    payment_method = c18.selectbox("결제 수단", FORM_OPTIONS["PaymentMethod"])

    c19, c20 = st.columns(2)
    monthly_charges = c19.number_input("월 요금 ($)", min_value=0.0, value=70.0, step=1.0)
    total_charges = c20.number_input(
        "누적 청구액 ($, 비워두면 tenure×월요금으로 추정 — 신규 고객일수록 실제값과 차이가 클 수 있음)",
        min_value=0.0, value=float(tenure * monthly_charges), step=1.0,
    )

    submitted = st.form_submit_button("이탈 확률 예측하기", width='stretch', type="primary")

if submitted:
    row = {
        "customerID": customer_id or "NEW-0001",
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }
    df_input = pd.DataFrame([row])

    with st.spinner("예측 중..."):
        result = predict_churn(df_input)
        result = add_expected_loss(result)
        explanation = explain_prediction(df_input)

    rules = load_segment_rules()
    risk_attrs = rules["subtrack_q"]["risk_attribute_values"]
    promo_cards = recommend_promotions(df_input.iloc[0], risk_attrs)

    prob = float(result["이탈확률"].iloc[0])
    segment = int(result["segment"].iloc[0])
    risk_count = int(result["risk_count"].iloc[0])
    expected_loss = float(result["예상손실액"].iloc[0])
    seg_text = segment_label_text(segment, rules["analysis_a"]["boundaries"], max(tenure, 72))
    tier_label, tier_color = risk_tier(prob)

    st.divider()

    # ---- 마케터용 한 줄 해설지 ------------------------------------------
    primary_cause = explanation["변수"].iloc[0] if len(explanation) else "추가 분석 필요"
    st.markdown(
        f"""
        <div style="padding:18px 20px;border-radius:10px;background:#F4F1EA;
                    border-left:5px solid {tier_color};margin-bottom:8px;">
            <span style="font-size:16px;">
            이 고객은 <b>가입 {tenure}개월 차(세그먼트 {segment}, {seg_text})</b>이고
            위험 신호 <b>{risk_count}개</b>를 보유하고 있습니다.
            핵심 원인은 <b>{primary_cause}</b>입니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 예측 결과")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background:{tier_color}1A;
                        border:2px solid {tier_color};text-align:center;">
                <div style="font-size:13px;color:#555;">예상 이탈 확률</div>
                <div style="font-size:40px;font-weight:700;color:{tier_color};">{prob:.1%}</div>
                <div style="font-size:16px;font-weight:600;color:{tier_color};">{tier_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"권장 분류 임계값: {threshold:.2f}")
    with rc2:
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background:#FFF3E0;
                        border:2px solid #E0A03C;text-align:center;">
                <div style="font-size:13px;color:#555;">예상 손실액 (1개월)</div>
                <div style="font-size:40px;font-weight:700;color:#B5732A;">${expected_loss:,.0f}</div>
                <div style="font-size:13px;color:#777;">이탈확률 × 월요금</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("이탈확률이 낮아도 월요금이 높으면 손실액 기준으로는 우선순위가 높을 수 있습니다.")
        st.caption("📌 1개월 근사치 · 약정 잔여기간(LTV) 반영 시 더 클 수 있음")
    with rc3:
        st.metric("소속 세그먼트", f"세그먼트 {segment}", seg_text)
        st.metric("위험 속성 개수 (risk_count)", f"{risk_count} / 5")

    st.divider()

    pc1, pc2 = st.columns([3, 2])
    with pc1:
        st.markdown("#### 🎯 권장 프로모션 카드")
        st.caption(
            "⚠️ 아래 카드는 모델이 찾은 위험속성에 대해 사람이 정한 대응 규칙(도메인 지식)입니다 — "
            "통계 검증된 인과관계가 아니라 실행을 위한 제안입니다."
        )
        if promo_cards:
            for card in promo_cards:
                st.markdown(
                    f"""
                    <div style="padding:14px 16px;border-radius:8px;background:#FFFFFF;
                                border:1px solid #DDD;margin-bottom:10px;">
                        <div style="font-size:13px;color:#888;">{card['속성']} = {card['값']}</div>
                        <div style="font-size:14px;color:#444;margin:4px 0;">{card['원인설명']}</div>
                        <div style="font-size:15px;font-weight:600;color:#2E6B9E;">→ {card['프로모션카드']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("이 고객은 정의된 5개 위험속성에 해당하지 않습니다. 별도 위험 신호가 없는 안정적인 프로필입니다.")

    with pc2:
        st.markdown("#### 이탈확률에 영향을 준 요인 (SHAP)")
        st.dataframe(explanation, width='stretch', hide_index=True)

    if prob >= threshold:
        st.warning(
            f"⚠️ 권장 임계값({threshold:.2f})을 초과해 **이탈 가능성이 높음**으로 분류됩니다. "
            "위 프로모션 카드 중 하나를 즉시 검토하세요."
        )
    else:
        st.success("✅ 이 고객은 현재 이탈 가능성이 낮은 편으로 예측됩니다.")
else:
    st.info("위 양식을 채우고 **이탈 확률 예측하기** 버튼을 누르면 결과가 표시됩니다.")
