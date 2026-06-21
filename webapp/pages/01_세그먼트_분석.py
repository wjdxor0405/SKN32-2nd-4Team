"""
webapp/pages/01_세그먼트_분석.py

거시적 관점: 가입 후 경과월(tenure)을 x축, 이탈율을 y축으로 한 라인 차트.
segment_discovery 가 찾은 경계(점선)를 표시하고, 각 세그먼트를 클릭해서
선택하면 해당 세그먼트의 상세 정보(분석A/B 결과)를 보여준다.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    assert_artifacts_ready,
    assign_segment,
    load_raw_data,
    load_segment_rules,
    segment_label_text,
)

st.set_page_config(page_title="세그먼트 분석 | 이탈 예측", page_icon="📈", layout="wide")
assert_artifacts_ready()

st.title("📈 거시적 관점: tenure별 이탈율과 세그먼트 경계")
st.caption(
    "가입 후 경과월(tenure)이 늘어날수록 이탈율이 어떻게 바뀌는지 보여줍니다. "
    "점선은 데이터가 직접 찾아낸 세그먼트 경계입니다."
)
st.caption(
    "📌 현재 보유 고객 기준 가입 경과월별 이탈율 (특정 시점 스냅샷, 코호트 추이 아님)"
)

rules = load_segment_rules()
df = load_raw_data()
boundaries = rules["analysis_a"]["boundaries"]
analysis_b = {row["segment"]: row for row in rules["analysis_b"]}
max_tenure = int(df["tenure"].max())

# ---------------------------------------------------------------------------
# 월별 이탈율 집계 (분석A가 실제로 학습에 쓴 것과 같은 집계 단위)
# ---------------------------------------------------------------------------
monthly = (
    df.groupby("tenure")["ChurnFlag"]
    .agg(이탈율="mean", 고객수="count")
    .reset_index()
)

df_seg = assign_segment(df, boundaries)
n_segments = len(boundaries) + 1
seg_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]


def make_figure(selected_segment: int | None) -> go.Figure:
    fig = go.Figure()

    # 세그먼트 배경 음영 (선택된 세그먼트는 강조)
    edges = [0] + [round(b, 1) for b in boundaries] + [max_tenure]
    for seg_id in range(n_segments):
        x0, x1 = edges[seg_id], edges[seg_id + 1]
        is_selected = selected_segment == seg_id
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=seg_colors[seg_id % len(seg_colors)],
            opacity=0.22 if is_selected else 0.08,
            line_width=0,
            layer="below",
        )

    # 실제 월별 이탈율 라인
    fig.add_trace(
        go.Scatter(
            x=monthly["tenure"],
            y=monthly["이탈율"],
            mode="lines+markers",
            name="월별 이탈율",
            line=dict(color="#333333", width=2),
            marker=dict(size=5),
            customdata=monthly["고객수"],
            hovertemplate=(
                "tenure=%{x}개월<br>이탈율=%{y:.1%}<br>고객수=%{customdata}명<extra></extra>"
            ),
        )
    )

    # 세그먼트 경계 점선 + 라벨
    for b in boundaries:
        fig.add_vline(
            x=b, line_width=2, line_dash="dash", line_color="#888888",
        )
        fig.add_annotation(
            x=b, y=1.0, yref="paper", showarrow=False,
            text=f"{b:.1f}개월", yshift=10,
            font=dict(size=11, color="#888888"),
        )

    # 세그먼트 클릭 영역(투명 박스, 클릭 선택용) + 세그먼트 라벨
    for seg_id in range(n_segments):
        x0, x1 = edges[seg_id], edges[seg_id + 1]
        seg_churn = df_seg[df_seg["segment"] == seg_id]["ChurnFlag"].mean()
        fig.add_trace(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[monthly["이탈율"].max() * 1.08],
                mode="markers",
                marker=dict(size=26, color=seg_colors[seg_id % len(seg_colors)], opacity=0.85),
                name=f"세그먼트 {seg_id}",
                text=[f"세그먼트 {seg_id}<br>평균 이탈율 {seg_churn:.1%}<br>(클릭해서 상세보기)"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=520,
        xaxis_title="가입 후 경과월 (tenure)",
        yaxis_title="이탈율",
        yaxis_tickformat=".0%",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        margin=dict(t=60, b=40),
    )
    return fig


if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None

fig = make_figure(st.session_state.selected_segment)
event = st.plotly_chart(
    fig, width='stretch', on_select="rerun", key="tenure_churn_chart",
    selection_mode="points",
)

# 클릭된 포인트로부터 세그먼트 선택 (큰 동그라미 trace만 세그먼트 마커이므로
# curveNumber로 구분: trace 0=라인, 이후 trace가 세그먼트 마커들)
clicked_segment = None
if event and event.get("selection", {}).get("points"):
    point = event["selection"]["points"][0]
    curve_number = point.get("curve_number", point.get("curveNumber"))
    if curve_number is not None and curve_number >= 1:
        clicked_segment = curve_number - 1  # trace 0은 라인차트

if clicked_segment is not None:
    st.session_state.selected_segment = clicked_segment

# ---------------------------------------------------------------------------
# 세그먼트 선택 UI (차트 클릭이 익숙하지 않을 수 있으므로 버튼도 함께 제공)
# ---------------------------------------------------------------------------
st.markdown("##### 세그먼트 선택")
btn_cols = st.columns(n_segments + 1)
for seg_id in range(n_segments):
    label = f"세그먼트 {seg_id} ({segment_label_text(seg_id, boundaries, max_tenure)})"
    if btn_cols[seg_id].button(label, key=f"seg_btn_{seg_id}", width='stretch'):
        st.session_state.selected_segment = seg_id
        st.rerun()
if btn_cols[n_segments].button("선택 해제", width='stretch'):
    st.session_state.selected_segment = None
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 선택된 세그먼트 상세 정보
# ---------------------------------------------------------------------------
selected = st.session_state.selected_segment
if selected is None:
    st.info("위 차트에서 동그라미(세그먼트 마커)를 클릭하거나, 위 버튼을 눌러 세그먼트를 선택하세요.")
else:
    seg_df = df_seg[df_seg["segment"] == selected]
    b_info = analysis_b.get(selected)

    st.markdown(f"### 세그먼트 {selected} 상세 — tenure {segment_label_text(selected, boundaries, max_tenure)}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("고객 수", f"{len(seg_df):,}명")
    m2.metric("이탈율", f"{seg_df['ChurnFlag'].mean():.1%}")
    m3.metric("평균 월요금", f"${seg_df['MonthlyCharges'].mean():.1f}")
    if b_info:
        m4.metric("속성기반 AUC", f"{b_info['attribute_auc']:.3f}")

    if b_info:
        st.markdown("**이 세그먼트에서 이탈을 가장 잘 설명하는 속성** (분석B 결과)")
        attrs = b_info["top_attributes"]
        st.write(" · ".join(f"`{a}`" for a in attrs))
        p_val = b_info["p_value"]
        ci_low, ci_high = b_info["ci_low"], b_info["ci_high"]
        sig_text = "통계적으로 유의함 (p < 0.05)" if p_val < 0.05 else f"유의하지 않을 수 있음 (p={p_val:.3f})"
        st.caption(
            f"순열검정 p-value = {p_val:.4f} → {sig_text}  |  "
            f"부트스트랩 95% 신뢰구간: [{ci_low:.3f}, {ci_high:.3f}]"
        )
    else:
        st.warning("이 세그먼트는 표본 수가 적어 분석B(위험속성) 검증을 통과하지 못했습니다.")

    with st.expander("이 세그먼트 고객 미리보기 (상위 20명)"):
        st.dataframe(
            seg_df[
                ["customerID", "tenure", "Contract", "MonthlyCharges", "InternetService", "Churn"]
            ].head(20),
            width='stretch',
            hide_index=True,
        )

st.divider()
with st.expander("분석A 통계 검증 결과 (경계 자체의 신뢰도)"):
    a = rules["analysis_a"]
    c1, c2, c3 = st.columns(3)
    c1.metric("세그먼트단독 AUC", f"{a['segment_only_auc']:.3f}")
    c2.metric("순열검정 p-value", f"{a['p_value']:.4f}")
    c3.metric("부트스트랩 95% CI", f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    st.caption(
        "tenure 구간(세그먼트) 라벨 하나만으로 이탈을 분류했을 때의 AUC와, "
        "그 결과가 우연이 아님을 확인하는 순열검정·부트스트랩 신뢰구간입니다. "
        f"RF 보조검증 일치 여부: {'일치' if str(a.get('rf_agreement')) == 'True' else '불일치'}"
    )

st.info(
    "💡 가입일·이탈일이 누적되면 실제 코호트 추이 분석(예: 2025년 1월 가입자가 "
    "N개월 후 몇 % 남았는가)으로 확장 가능합니다."
)
