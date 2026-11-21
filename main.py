#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
극지식물 개체별 생육 차이 분석 대시보드
- Streamlit Cloud 대응 / 한글 폰트 자동 설치 / 인코딩 우회
- 학교마다 다른 열 이름을 최대한 자동 매핑
"""
import warnings, io, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────
# ① 한글 폰트 (Streamlit Cloud / 로컬 모두)
# ─────────────────────────────────────────────
def install_korean_font():
    # Linux(Streamlit Cloud)에서 Nanum 폰트 설치
    if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        return "NanumGothic"
    try:
        import subprocess
        subprocess.run(
            ["apt-get", "install", "-y", "fonts-nanum"],
            capture_output=True, check=False
        )
    except Exception:
        pass
    # Matplotlib 폰트 캐시 초기화 후 다시 조회
    fm._load_fontmanager(try_read_cache=False)
    for candidate in ["NanumGothic", "NanumBarunGothic", "NanumSquare"]:
        matches = [f for f in fm.fontManager.ttflist if candidate in f.name]
        if matches:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    # 마지막 수단: Malgun Gothic(win) 또는 기본
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"

FONT_NAME = install_korean_font()

# ─────────────────────────────────────────────
# ② 열 이름 자동 매핑 유틸 (학교별 차이 대응)
# ─────────────────────────────────────────────
def find_col(df, candidates):
    """candidates 목록 중 df에 존재하는 첫 번째 열 이름을 반환"""
    for c in candidates:
        if c in df.columns:
            return c
        # 공백·단위 붙은 변형도 시도 (예: '온도 ' vs '온도')
        stripped = [cc.strip() for cc in df.columns]
        idx = [i for i, s in enumerate(stripped) if s == c.strip()]
        if idx:
            return df.columns[idx[0]]
    return None

ENV_COL_MAP = {
    "time":      ["time", "시각", "시간", "날짜", "time", "일자"],
    "temperature": ["temperature", "temp", "온도", "기온", "공기온도"],
    "humidity":  ["humidity", "hum", "습도", "상대습도"],
    "ph":        ["ph", "PH", "수소이온농도", "산도"],
    "ec":        ["ec", "EC", "전기전도도", "ec_ms_cm"],
    "water_temp":["water_temp", "water temperature", "수온", "물온도"],
    "co2":       ["co2", "CO2", "이산화탄소", "co2_ppm"],
}

GROWTH_COL_MAP = {
    "id":            ["개체번호", " 개체번호", "ID", "id", "식물번호", "개체"],
    "leaf_count":    ["잎 수(장)", "잎수(장)", "잎 수", "leaf_count", "leaf"],
    "shoot_length":  ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "shoot_length"],
    "root_length":   ["지하부길이(mm)", "지하부 길이(mm)", "지하부길이", "지하부 길이", "root_length"],
    "fresh_weight":  ["생중량(g)", "생중량", "fresh_weight", "생체량"],
    "shoot_mass":    ["지상부질량(g)", "지상부 질량(g)", "지상부질량", "shoot_mass", "지상부질량"],
    "root_mass":     ["지하부질량(g)", "지하부 질량(g)", "지하부질량", "root_mass", "지하부질량"],
}

def read_csv_safe(path):
    """인코딩 오류가 나도 여러 인코딩을 시도"""
    for enc in ["utf-8-sig", "cp949", "utf-8", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    # 전부 실패 시 pandas 기본(mac crc 등)으로 읽고 오류를 경고
    df = pd.read_csv(path)
    st.warning("CSV 인코딩을 자동 감지했습니다. 데이터가 깨져 보이면 파일을 utf-8-sig으로 다시 저장해 주세요.")
    return df

def map_growth_cols(df_raw):
    """school마다 다른 생육 열 이름을 표준 키로 매핑"""
    out = {}
    for std_key, candidates in GROWTH_COL_MAP.items():
        col = find_col(df_raw, candidates)
        if col is not None:
            out[std_key] = col
    return out

def map_env_cols(df_raw):
    out = {}
    for std_key, candidates in ENV_COL_MAP.items():
        col = find_col(df_raw, candidates)
        if col is not None:
            out[std_key] = col
    return out

# ─────────────────────────────────────────────
# ③ 통계 설명 (고등학생 수준)
# ─────────────────────────────────────────────
STAT_HELP = {
    "avg": "평균 — 모든 값을 더해 개수로 나눈 '대표값'이에요. 극단값에 영향을 받아요.",
    "std": "표준편차 — 값들이 평균 주위에 얼마나 흩어져 있는지 나타내요. 클수록 개체 간 차이가 크다는 뜻이에요.",
    "cv": "변동계수(CV) — 표준편차를 평균으로 나눈 값. 단위가 다른 것도 상대 비교할 수 있어요.",
    "corr": "상관계수(r) — 두 변수가 함께 움직이는 정도(-1~1). 0에 가까우면 관계 없음, ±1에 가까울수록 강한 관계예요.",
    "pearson": "피어슨 상관계수 — 두 수치형 변수의 선형 관계를 측정해요. 위 상관 그림과 같은 값이에요.",
    "count": "개체 수 — 분석에 포함된 식물 개체의 총 개수예요.",
    "min": "최솟값 — 데이터에서 가장 작은 값, 반대는 최댓값이에요.",
    "max": "최댓값 — 데이터에서 가장 큰 값이에요.",
}

def stat_explain(key):
    return STAT_HELP.get(key, "")

# ─────────────────────────────────────────────
# ④ 페이지 구성
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 대시보드",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🌱 극지식물 생육 대시보드")
st.sidebar.markdown("""
**사용법**
1. 왼쪽 아래 **파일 업로드**에 두 CSV 파일을 올려 주세요.
2. **환경 데이터**와 **생육 결과 데이터**를 각각 올리면 대시보드가 자동 생성됩니다.
3. 학교마다 열 이름이 달라도 가능한 한 자동으로 맞춰줍니다.
""")

uploaded_env = st.sidebar.file_uploader(
    "🌿 환경 데이터 (CSV)", type=["csv"], help="시간별 온도·습도·pH·EC·CO₂ 등"
)
uploaded_growth = st.sidebar.file_uploader(
    "📏 생육 결과 데이터 (CSV)", type=["csv"], help="개체번호·잎 수·길이·질량 등"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "학교별로 열 이름이 다를 수 있어요. "
    "예) 환경: `time,temperature,humidity,ph,ec,water_temp,co2` "
    "또는 `categories,온도,습도,EC,PH` 등"
)
st.sidebar.caption(
    "생육: `개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)` 등"
)

# ─────────────────────────────────────────────
# ⑤ 메인 영역
# ─────────────────────────────────────────────
if uploaded_env is None and uploaded_growth is None:
    st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")
    st.info(
        "왼쪽 사이드바에서 **두 CSV 파일**을 모두 업로드해 주세요.\n"
        "- 환경 데이터: 시간별 온·습도, pH, EC, CO₂ 등\n"
        "- 생육 결과 데이터: 개체별 잎 수, 길이, 질량 등"
    )
    st.markdown("""
    ### 📋 열 이름 예시
    | 구분 | 열 이름 예시 |
    |------|--------------|
    | 환경 데이터 | `time, temperature, humidity, ph, ec, water_temp, co2` 또는 `categories, 온도, 습도, EC, PH` |
    | 생육 결과 데이터 | `개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)` |
    """)
    st.stop()

# ── 파일 읽기 ──────────────────────────────
try:
    env_raw = read_csv_safe(uploaded_env) if uploaded_env else pd.DataFrame()
except Exception as e:
    st.error(f"환경 데이터 읽기 중 오류: {e}")
    st.stop()

try:
    growth_raw = read_csv_safe(uploaded_growth) if uploaded_growth else pd.DataFrame()
except Exception as e:
    st.error(f"생육 데이터 읽기 중 오류: {e}")
    st.stop()

# ── 열 매핑 ────────────────────────────────
env_map    = map_env_cols(env_raw)    if not env_raw.empty    else {}
growth_map = map_growth_cols(growth_raw) if not growth_raw.empty else {}

env_mapped    = env_raw.rename(columns={v: k for k, v in env_map.items()}) if env_map else pd.DataFrame()
growth_mapped = growth_raw.rename(columns={v: k for k, v in growth_map.items()}) if growth_map else pd.DataFrame()

# ── 시간 파싱 ──────────────────────────────
if "time" in env_mapped.columns:
    env_mapped["time"] = pd.to_datetime(env_mapped["time"], errors="coerce")
    env_mapped = env_mapped.dropna(subset=["time"])
    env_mapped = env_mapped.sort_values("time").reset_index(drop=True)

# ── 생중량 계산 (없으면 지상부+지하부로) ───
if "fresh_weight" not in growth_mapped.columns and "shoot_mass" in growth_mapped.columns and "root_mass" in growth_mapped.columns:
    growth_mapped["fresh_weight"] = growth_mapped["shoot_mass"] + growth_mapped["root_mass"]
    st.info("생중량(g) 열이 없어 **지상부질량 + 지하부질량**으로 계산했습니다.")

# ── 결측치 기본 처리 ───────────────────────
for col in ["leaf_count","shoot_length","root_length","fresh_weight","shoot_mass","root_mass"]:
    if col in growth_mapped.columns:
        growth_mapped[col] = pd.to_numeric(growth_mapped[col], errors="coerce")

# ─────────────────────────────────────────────
# ⑥ 대시보드 렌더링
# ─────────────────────────────────────────────
st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")

# ── 파일 정보 카드 ─────────────────────────
if not env_raw.empty:
    st.subheader("📁 환경 데이터 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 측정 횟수", f"{len(env_mapped):,}회")
    c2.metric("측정 기간", f"{env_mapped['time'].min():%Y-%m-%d} ~ {env_mapped['time'].max():%Y-%m-%d}" if "time" in env_mapped.columns else "N/A")
    c3.metric("사용 열", len(env_map))
    c4.metric("결측치 비율", f"{(env_mapped.isnull().sum().sum() / env_mapped.size * 100):.1f}%")

    env_cols_show = st.multiselect(
        "표시할 환경 변수 선택",
        options=[k for k in ["temperature","humidity","ph","ec","water_temp","co2"] if k in env_mapped.columns],
        default=[k for k in ["temperature","humidity","ph","ec"] if k in env_mapped.columns],
        help="차트에 그릴 환경 변수를 고르세요.",
    )
else:
    env_cols_show = []

if not growth_raw.empty:
    st.subheader("📁 생육 결과 데이터 개요")
    c1, c2, c3 = st.columns(3)
    c1.metric("개체 수 (N)", f"{len(growth_mapped):,}개")
    c2.metric("사용 열", len(growth_map))
    c3.metric("결측치 비율", f"{(growth_mapped.isnull().sum().sum() / growth_mapped.size * 100):.1f}%")

    growth_cols_show = st.multiselect(
        "분석할 생육 지표 선택",
        options=[k for k in ["leaf_count","shoot_length","root_length","fresh_weight","shoot_mass","root_mass"] if k in growth_mapped.columns],
        default=[k for k in ["leaf_count","shoot_length","fresh_weight"] if k in growth_mapped.columns],
        help="개체별 막대차트·통계에 포함할 지표예요.",
    )
else:
    growth_cols_show = []

st.markdown("---")

# ─────────────────────────────────────────────
# ⑦ 1페이지: 생육 통계 요약
# ─────────────────────────────────────────────
if not growth_raw.empty and growth_cols_show:
    st.header("📊 개체별 생육 통계 요약")

    stats_df = growth_mapped[growth_cols_show].describe().T
    stats_df["변동계수(CV)"] = (stats_df["std"] / stats_df["mean"] * 100).round(1)
    stats_df = stats_df[["count","mean","std","min","25%","50%","75%","max","변동계수(CV)"]]
    stats_df.columns = ["개체수","평균","표준편차","최솟값","Q1","중앙값","Q3","최댓값","CV(%)"]

    col_expl = {
        "개체수": "count",
        "평균": "avg",
        "표준편차": "std",
        "CV(%)": "cv",
    }

    st.caption("💡 각 행은 하나의 생육 지표, 각 열은 통계 요약이에요.")
    st.dataframe(stats_df.round(2), use_container_width=True, hide_index=True)

    with st.expander("📖 통계 용어 해설"):
        for col_kr, col_key in col_expl.items():
            st.markdown(f"**{col_kr}** — {stat_explain(col_key)}")

    # 분포가 심하게 치우친 지표 확인
    skew_df = growth_mapped[growth_cols_show].skew().round(2)
    if (skew_df.abs() > 1).any():
        st.warning("일부 지표에서 데이터가 한쪽으로 치우쳐 있어요(왜도 > 1). 평균만 보면 대표성을 왜곡할 수 있으니 중앙값(Q2)도 함께 확인해 주세요.")

st.markdown("---")

# ─────────────────────────────────────────────
# ⑧ 2페이지: 개체별 생육 막대차트
# ─────────────────────────────────────────────
if not growth_raw.empty and growth_cols_show:
    st.header("📈 개체별 생육 지표 비교")

    id_col = growth_map.get("id", None)
    chart_df = growth_mapped.copy()

    if id_col and id_col in chart_df.columns:
        chart_df["label"] = chart_df[id_col].astype(str)
    else:
        chart_df["label"] = [f"개체{i+1}" for i in range(len(chart_df))]

    sel = st.selectbox(
        "차트 지표 선택",
        options=growth_cols_show,
        help="개체별 막대차트로 볼 지표를 고르세요.",
    )
    if sel:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        vals = chart_df[sel].dropna()
        labels = chart_df.loc[vals.index, "label"]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(vals)))
        ax.bar(range(len(vals)), vals.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"개체별 {sel} 분포", fontsize=13)
        ax.set_ylabel(sel)
        ax.axhline(vals.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"평균 = {vals.mean():.2f}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)

    st.caption(f"💡 막대는 개체별 {sel} 값, 빨간 점선은 평균이에요. 평균 위 막대는 평균보다 큰 개체예요.")

st.markdown("---")

# ─────────────────────────────────────────────
# ⑨ 3페이지: 생육 지표 간 상관관계
# ─────────────────────────────────────────────
if not growth_raw.empty and len(growth_cols_show) >= 2:
    st.header("🔗 생육 지표 간 상관관계")

    corr_cols = growth_cols_show
    corr_mat = growth_mapped[corr_cols].corr(method="pearson").round(3)

    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    im = ax2.imshow(corr_mat.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(corr_cols)))
    ax2.set_xticklabels(corr_cols, rotation=40, ha="right", fontsize=9)
    ax2.set_yticks(range(len(corr_cols)))
    ax2.set_yticklabels(corr_cols, fontsize=9)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax2.text(j, i, f"{corr_mat.iloc[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if abs(corr_mat.iloc[i, j]) > 0.6 else "black",
                     fontsize=10, fontweight="bold")
    ax2.set_title("Pearson 상관계수 행렬", fontsize=13)
    fig2.colorbar(im, ax=ax2, shrink=0.8, label="r (상관계수)")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.caption(
        "💡 셀의 숫자(r)는 두 지표 간 **Pearson 상관계수**예요. "
        "1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽 감소, "
        "0에 가까우면 선형 관계가 거의 없어요."
    )

    with st.expander("📖 상관계수 해석 가이드"):
        st.markdown("""
        | r 범위 | 관계 강도 |
        |--------|-----------|
        | 0.7 ~ 1.0 | 강한 양의 상관 |
        | 0.3 ~ 0.7 | 중간 양의 상관 |
        | 0.0 ~ 0.3 | 약한 상관 또는 거의 없음 |
        | -0.3 ~ 0.0 | 약한 음의 상관 |
        | -0.7 ~ -0.3 | 중간 음의 상관 |
        | -1.0 ~ -0.7 | 강한 음의 상관 |
        """)

    # 가장 강한 상관 쌍
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    flat = upper.stack().dropna()
    if len(flat) > 0:
        top_pair = flat.abs().idxmax()
        top_val = flat.loc[top_pair]
        c_a, c_b = top_pair
        st.info(
            f"가장 강한 상관: **{c_a} ↔ {c_b}** (r = {top_val:+.3f})"
            + (" → 함께 증가하는 경향" if top_val > 0 else " → 한쪽이 증가하면 다른 쪽이 감소하는 경향")
        )

st.markdown("---")

# ─────────────────────────────────────────────
# ⑩ 4페이지: 환경 vs 생육 상관 (시간 병합)
# ─────────────────────────────────────────────
if not env_raw.empty and not growth_raw.empty and env_cols_show and growth_cols_show:
    st.header("🌡️ 환경 조건 vs 생육 지표 상관 분석")

    st.info(
        "각 개체의 측정 시각이 환경 데이터와 같다고 가정하고, "
        "개체별 생육 지표와 해당 시각의 환경 값을 연결해 상관분석을 해요."
    )

    env_time_col = "time"
    if env_time_col in env_mapped.columns and "id" in growth_mapped.columns:
        # 환경: 시간별 → 하루/측정 시점 평균 등으로 요약 옵션
        env_summary_method = st.selectbox(
            "환경 데이터 요약 방법",
            options=["각 시간대 그대로 사용", "일별 평균"],
            help="환경 값이 시간별로 여러 개 있을 때 어떻게 요약할지 선택해요.",
        )

        if env_summary_method == "일별 평균":
            env_ts = env_mapped.copy()
            env_ts["date"] = env_ts["time"].dt.date
            env_daily = env_ts.groupby("date")[env_cols_show].mean().reset_index()
            env_daily["time"] = pd.to_datetime(env_daily["date"])
            env_use = env_daily
            st.caption("환경 데이터를 **일별 평균**으로 요약했어요.")
        else:
            env_use = env_mapped

        # 생육 데이터의 시간 열이 있다면 병합
        time_in_growth = find_col(growth_raw, ["시간", "time", "시각", "측정시간"])
        if time_in_growth and time_in_growth in growth_mapped.columns:
            growth_mapped["time"] = pd.to_datetime(growth_mapped[time_in_growth], errors="coerce")
            merged = pd.merge_asof(
                growth_mapped.sort_values("time").reset_index(drop=True),
                env_use.sort_values("time").reset_index(drop=True),
                on="time", direction="nearest", tolerance=pd.Timedelta("12h"),
            )
            st.caption(f"개체별 측정 시각과 가장 가까운 환경 값을 연결했어요 (최대 12시간 차이).")
        else:
            # 시간 정보가 없으면 전체 환경 평균 사용
            env_mean = env_mapped[env_cols_show].mean().to_frame().T
            merged = growth_mapped.copy()
            for col in env_cols_show:
                merged[col] = env_mean[col].values[0]
            st.caption("생육 데이터에 시간 정보가 없어 **전체 기간 환경 평균**을 모든 개체에 적용했어요.")

        merged_corr_cols = [c for c in growth_cols_show if c in merged.columns] + \
                           [c for c in env_cols_show if c in merged.columns]
        merged_corr_cols = list(dict.fromkeys(merged_corr_cols))

        if len(merged_corr_cols) >= 3:
            m_corr = merged[merged_corr_cols].corr(method="pearson").round(3)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            im3 = ax3.imshow(m_corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
            ax3.set_xticks(range(len(merged_corr_cols)))
            ax3.set_xticklabels(merged_corr_cols, rotation=45, ha="right", fontsize=8)
            ax3.set_yticks(range(len(merged_corr_cols)))
            ax3.set_yticklabels(merged_corr_cols, fontsize=8)
            for i in range(len(merged_corr_cols)):
                for j in range(len(merged_corr_cols)):
                    ax3.text(j, i, f"{m_corr.iloc[i,j]:.2f}",
                             ha="center", va="center",
                             color="white" if abs(m_corr.iloc[i,j]) > 0.55 else "black",
                             fontsize=9)
            ax3.set_title("환경·생육 통합 상관행렬", fontsize=13)
            fig3.colorbar(im3, ax=ax3, shrink=0.8, label="r")
            fig3.tight_layout()
            st.pyplot(fig3)

            # 생육-환경 상관만 추출
            growth_env_pairs = []
            for gc in growth_cols_show:
                for ec in env_cols_show:
                    if gc in m_corr.columns and ec in m_corr.columns:
                        growth_env_pairs.append((gc, ec, m_corr.loc[gc, ec]))
            if growth_env_pairs:
                gev_df = pd.DataFrame(growth_env_pairs, columns=["생육지표","환경변수","상관계수(r)"])
                gev_df = gev_df.sort_values("상관계수(r)", key=lambda s: s.abs(), ascending=False)
                st.dataframe(gev_df, use_container_width=True, hide_index=True)
                st.caption(
                    "💡 생육 지표와 환경 변수 간 상관계수예요. "
                    "절대값이 큰 쌍일수록 두 변수가 강한 관계로 보여요."
                )

st.markdown("---")

# ─────────────────────────────────────────────
# ⑪ 5페이지: 환경 시계열 차트
# ─────────────────────────────────────────────
if not env_raw.empty and env_cols_show:
    st.header("🌡️ 환경 조건 시계열")

    env_ts_df = env_mapped.copy()
    n_cols = min(3, len(env_cols_show))
    n_rows = int(np.ceil(len(env_cols_show) / n_cols))

    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), squeeze=False)
    for idx, col in enumerate(env_cols_show):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.plot(env_ts_df["time"], env_ts_df[col], linewidth=0.6, color=plt.cm.tab10(idx % 10))
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(linestyle=":", alpha=0.4)
        if idx % n_cols == 0:
            ax.set_ylabel("")
    for r in range(n_rows):
        for c in range(n_cols):
            if r * n_cols + c >= len(env_cols_show):
                axes[r][c].set_visible(False)
    fig4.suptitle("환경 데이터 시계열 (시간별)", fontsize=13, y=1.01)
    fig4.tight_layout()
    st.pyplot(fig4)

st.markdown("---")

# ─────────────────────────────────────────────
# ⑫ 6페이지: 데이터 미리보기
# ─────────────────────────────────────────────
with st.expander("🔬 원본 데이터 미리보기 (환경)"):
    if not env_raw.empty:
        st.caption(f"열: {list(env_raw.columns)}")
        st.dataframe(env_raw.head(20), use_container_width=True)
    else:
        st.caption("업로드된 환경 데이터가 없습니다.")

with st.expander("🔬 원본 데이터 미리보기 (생육)"):
    if not growth_raw.empty:
        st.caption(f"열: {list(growth_raw.columns)}  |  개체 수: {len(growth_raw)}개")
        st.dataframe(growth_raw.head(30), use_container_width=True)
    else:
        st.caption("업로드된 생육 데이터가 없습니다.")

st.markdown("---")
st.caption(
    "⚠️ 학교마다 열 이름·단위가 다를 수 있어요. "
    "오류가 생기면 사이드바 파일 업로드 전에 CSV 열 이름을 위 예시 형식에 맞춰 주세요.\n"
    "데이터 인코딩은 utf-8-sig / cp949를 자동으로 시도합니다."
)
