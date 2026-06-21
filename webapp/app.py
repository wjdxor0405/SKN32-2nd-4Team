"""
webapp/app.py

Streamlit 대시보드 진입점.

segment_discovery/outputs/segment_rules.json 과
churn_prediction/outputs/latest/{model.pkl, feature_transformer.pkl, metadata.json}
을 읽어다 시각화한다 (코드 재실행이 아닌 산출물 파일을 읽는 구조 - README 참조).

페이지 구성 (pages/ 폴더, 사이드바에서 이동):
- 01_세그먼트_분석.py : tenure-이탈율 거시 차트, 세그먼트 경계선, 세그먼트별 상세
- 02_개인_예측.py     : 한 명의 정보를 입력해 이탈확률 확인
- 03_일괄_예측.py     : 여러 명을 CSV로 업로드해 일괄 이탈확률 + 분석

실행 방법:
    pip install -r requirements.txt
    streamlit run webapp/app.py

⚠️ 모델이 아직 없다면 먼저 아래를 실행해야 한다:
    cd segment_discovery && python app.py
    cd churn_prediction && python train.py
"""
import streamlit as st

from utils import (
    assert_artifacts_ready,
    load_metadata,
    load_raw_data,
    load_segment_rules,
)

st.set_page_config(
    page_title="고객 이탈 예측 대시보드",
    page_icon="📉",
    layout="wide",
)

st.title("📉 가입 고객 이탈 예측 대시보드")
st.caption("segment_discovery(세그먼트 발견) + churn_prediction(예측모델) 산출물을 기반으로 합니다.")

assert_artifacts_ready()

rules = load_segment_rules()
metadata = load_metadata()
df = load_raw_data()

st.markdown("### 한눈에 보기")
col1, col2, col3, col4 = st.columns(4)

boundaries = rules["analysis_a"]["boundaries"]
n_segments = len(boundaries) + 1
overall_churn = df["ChurnFlag"].mean()

col1.metric("전체 고객 수", f"{len(df):,}명")
col2.metric("전체 이탈율", f"{overall_churn:.1%}")
col3.metric("발견된 세그먼트 수", f"{n_segments}개")
if metadata:
    col4.metric("메인 모델 AUC", f"{metadata.get('main_model_auc', 0):.3f}")
else:
    col4.metric("메인 모델 AUC", "—")

st.divider()

left, right = st.columns([3, 2])
with left:
    st.markdown("#### 이 대시보드로 할 수 있는 것")
    st.markdown(
        """
1. **세그먼트 분석** — 가입 후 경과월(tenure)에 따라 이탈율이 어떻게 달라지는지,
   그리고 데이터가 직접 찾아낸 위험 구간 경계선을 확인합니다.
2. **개인 예측** — 신규 가입 후보 또는 기존 고객 한 명의 정보를 입력해
   이탈 확률과 주요 위험 요인을 확인합니다.
3. **일괄 예측** — 여러 고객 데이터를 CSV로 한 번에 업로드해 이탈율을 예측하고,
   위험군 분포·세그먼트별 통계 등 분석 결과를 함께 봅니다.
        """
    )
with right:
    st.markdown("#### 모델 메타데이터")
    if metadata:
        st.json(metadata, expanded=False)
    else:
        st.info("metadata.json이 없습니다. train.py를 먼저 실행하세요.")

st.divider()
st.markdown("#### 세그먼트 경계 (tenure 기준, 개월)")
st.write(
    " · ".join(f"`{b:.1f}개월`" for b in boundaries)
    + f"  →  세그먼트 {n_segments}개로 분할"
)
st.caption(
    "이 경계는 segment_discovery의 가지치기 회귀나무가 데이터로부터 직접 찾았으며, "
    "순열검정·부트스트랩 신뢰구간 검증을 통과한 결과입니다. "
    "자세한 내용은 사이드바의 **세그먼트 분석** 페이지에서 확인하세요."
)

st.sidebar.success("왼쪽 메뉴에서 페이지를 선택하세요.")
