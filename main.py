#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
극지식물 개체별 생육 차이 분석 대시보드
- 인코딩(utf-8-sig/cp949) 자동 시도
- 학교마다 다른 열 이름 자동 매핑
- Streamlit Cloud 한글 폰트 자동 설치 (apt-get fonts-nanum)
- 고등학생 이해용 통계 설명 포함
"""
import warnings, io, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ──────────────────────────────────────────────
# ① 한글 폰트: Streamlit Cloud( Linux ) 대응
# ──────────────────────────────────────────────
def install_korean_font():
    # 이미 설치되어 있으면 그대로 사용
    if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        return "NanumGothic"
    # 없으면 설치 시도 (Streamlit Cloud apt-get)
    try:
        import subprocess
        subprocess.run(
            ["apt-get", "install", "-y", "fonts-nanum"],
            capture_output=True, check=False, timeout=60
        )
    except Exception:
        pass
    # 폰트 매니저 재로드
    fm._load_fontmanager(try_read_cache=False)
    for cand in ["NanumGothic", "NanumBarunGothic", "NanumSquare"]:
        hits = [f for f in fm.fontManager.ttflist if cand in f.name]
        if hits:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    # 최후의 수단 (win: Malgun Gothic fallback)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"

FONT_NAME = install_korean_font()

# ──────────────────────────────────────────────
# ② 열 이름 자동 매핑 (학교별 차이 대응)
#     find_col: 실제로 존재하는 첫 후보 열 이름 반환
# ──────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # 공백/단위 붙은 변형 대응: 열 이름 strip 비교
    stripped = [cc.strip() for cc in df.columns]
    for i, s in enumerate(stripped):
        for c in candidates:
            if s == c.strip():
                return df.columns[i]
    return None

ENV_COL_MAP = {
    "time":         ["time", "시각", "시간", "날짜", "일자", "datetime"],
    "temperature":  ["temperature", "temp", "온도", "기온", "공기온도"],
    "humidity":     ["humidity", "hum", "습도", "상대습도"],
    "ph":           ["ph", "PH", "수소이온농도", "산도"],
    "ec":           ["ec", "EC", "전기전도도"],
    "water_temp":   ["water_temp", "water temperature", "수온", "물온도"],
    "co2":          ["co2", "CO2", "이산화탄소", "co2_ppm"],
}

GROWTH_COL_MAP = {
    "id":           ["개체번호", " 개체번호", "ID", "id", "식물번호", "식물", "개체"],
    "leaf_count":   ["잎 수(장)", "잎 수", "잎수(장)", "잎수", "leaf_count", "leaf"],
    "shoot_length": ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이", "shoot_length", "지상부"],
    "root_length":  ["지하부길이(mm)", "지하부 길이(mm)", "지하부길이", "지하부 길이", "root_length", "지하부"],
    "fresh_weight": ["생중량(g)", "생중량", "fresh_weight", "생체량"],
    "shoot_mass":   ["지상부질량(g)", "지상부 질량(g)", "지상부질량", "shoot_mass", "지상부질량"],
    "root_mass":    ["지하부질량(g)", "지하부 질량(g)", "지하부질량", "root_mass", "지하부질량"],
}

def read_csv_safe(path):
    """인코딩 순서: utf-8-sig → cp949 → utf-8 → euc-kr → pandas 기본"""
    for enc in ["utf-8-sig", "cp949", "utf-8", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            # 빈 데이터인지 확인
            if df.shape[0] > 0:
                return df
        except Exception:
            continue
    df = pd.read_csv(path)
    st.warning("CSV 인코딩을 자동 감지했습니다. 글자가 깨지면 utf-8-sig 인코딩으로 다시 저장해 주세요.")
    return df

def map_env_cols(df_raw):
    return {k: v for k, v in ((k, find_col(df_raw, cand)) for k, cand in ENV_COL_MAP.items()) if v}

def map_growth_cols(df_raw):
    return {k: v for k, v in ((k, find_col(df_raw, cand)) for k, cand in GROWTH_COL_MAP.items()) if v}

# ──────────────────────────────────────────────
# ③ 통계 용어 설명 (고등학생 수준)
# ──────────────────────────────────────────────
EXPLAIN = {
    "개체수": "개체 수(N) — 분석에 포함된 식물 개체의 총 개수예요.",
    "평균":   "평균 — 모든 값을 더해 개수로 나눈 '대표값'이에요. 극단값(아주 큰/작은 값)에 영향을 받아요.",
    "표준편차": "표준편차 — 값들이 평균 주위에 얼마나 흩어져 있는지 나타내요. 클수록 개체 간 차이가 크다는 뜻이에요.",
    "최솟값": "최솟값 — 데이터에서 가장 작은 값이에요.",
    "최댓값": "최댓값 — 데이터에서 가장 큰 값이에요.",
    "Q1":     "Q1(제1사분위수) — 작은 값부터 나열했을 때 25% 지점의 값이에요. 하위 25%의 경계예요.",
    "중앙값": "중앙값(Q2) — 작은 값부터 나열했을 때 정가운데 값이에요. 극단값의 영향을 덜 받아요.",
    "Q3":     "Q3(제3사분위수) — 75% 지점의 값이에요. 상위 25%의 경계예요.",
    "CV(%)":  "변동계수(CV) — 표준편차를 평균으로 나눈 뒤 100을 곱한 값. 단위가 다른 지표도 상대 비교할 수 있어요.",
    "corr":   "상관계수(r, Pearson) — 두 변수가 함께 움직이는 정도(-1~1). 0이면 관계 없음, +1에 가까울수록 함께 증가, -1에 가까울수록 한쪽 증가 시 다른 쪽 감소예요.",
}

# ──────────────────────────────────────────────
# ④ 페이지 기본 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 대시보드",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🌱 극지식물 생육 대시보드")
st.sidebar.markdown("""
**사용법**
1. 왼쪽 파일 업로드에서 **환경 데이터**와 **생육 결과 데이터**를 각각 올려 주세요.
2. 학교마다 열 이름이 달라도 가능한 한 자동으로 맞춰 줍니다.
3. 통계 용어에 마우스를 올리면(또는 펼침 메뉴에서) 짧은 설명이 나와요.
""")

uploaded_env    = st.sidebar.file_uploader("🌿 환경 데이터 (CSV)", type=["csv"])
uploaded_growth = st.sidebar.file_uploader("📏 생육 결과 데이터 (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "열 이름 예시 (형식 A): `time, temperature, humidity, ph, ec, water_temp, co2`\n"
    "열 이름 예시 (형식 B): `categories, 온도, 습도, EC, PH`\n"
    "생육: `개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)`"
)

# ──────────────────────────────────────────────
# ⑤ 파일 읽기 & 열 매핑
# ──────────────────────────────────────────────
if uploaded_env is None and uploaded_growth is None:
    st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")
    st.info(
        "왼쪽 사이드바에서 **두 CSV 파일**을 모두 업로드해 주세요.\n"
        "- 환경 데이터: 시간별 온도·습도·pH·EC·CO₂ 등\n"
        "- 생육 결과 데이터: 개체별 잎 수·길이·질량 등"
    )
    st.stop()

# ── 환경 데이터 ──────────────────────────────
env_raw = pd.DataFrame()
if uploaded_env is not None:
    try:
        env_raw = read_csv_safe(uploaded_env)
    except Exception as e:
        st.error(f"환경 데이터 읽기 오류: {e}")
        st.stop()

# ── 생육 데이터 ──────────────────────────────
growth_raw = pd.DataFrame()
if uploaded_growth is not None:
    try:
        growth_raw = read_csv_safe(uploaded_growth)
    except Exception as e:
        st.error(f"생육 데이터 읽기 오류: {e}")
        st.stop()

# ── 열 이름 매핑 ──────────────────────────────
env_map    = map_env_cols(env_raw)    if not env_raw.empty    else {}
growth_map = map_growth_cols(growth_raw) if not growth_raw.empty else {}

env_mapped    = env_raw.rename(columns={v: k for k, v in env_map.items()})    if env_map    else pd.DataFrame()
growth_mapped = growth_raw.rename(columns={v: k for k, v in growth_map.items()}) if growth_map else pd.DataFrame()

# ── 시간 파싱 (있는 경우) ─────────────────────
if "time" in env_mapped.columns:
    env_mapped["time"] = pd.to_datetime(env_mapped["time"], errors="coerce")
    n_dropped = env_mapped["time"].isna().sum()
    if n_dropped > 0:
        env_mapped = env_mapped.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        st.warning(f"환경 데이터에서 시간 형식이 잘못된 {n_dropped}행을 제외했어요.")
    else:
        env_mapped = env_mapped.sort_values("time").reset_index(drop=True)

# ── 생중량(g) 없으면 계산 ─────────────────────
if "fresh_weight" not in growth_mapped.columns:
    if "shoot_mass" in growth_mapped.columns and "root_mass" in growth_mapped.columns:
        growth_mapped["fresh_weight"] = (
            pd.to_numeric(growth_mapped["shoot_mass"], errors="coerce")
            + pd.to_numeric(growth_mapped["root_mass"], errors="coerce")
        )
        st.info("⚠️ '생중량(g)' 열이 없어 **지상부질량 + 지하부질량**으로 계산했습니다.")
    elif "shoot_mass" in growth_mapped.columns:
        growth_mapped["fresh_weight"] = pd.to_numeric(growth_mapped["shoot_mass"], errors="coerce")
        st.info("⚠️ '생중량(g)'과 '지하부질량(g)'이 모두 없어 **지상부질량**만 사용했습니다.")

# ── 숫자형 변환 ───────────────────────────────
for col in ["leaf_count", "shoot_length", "root_length", "fresh_weight", "shoot_mass", "root_mass"]:
    if col in growth_mapped.columns:
        growth_mapped[col] = pd.to_numeric(growth_mapped[col], errors="coerce")

# ──────────────────────────────────────────────
# ⑥ 대시보드 시작
# ──────────────────────────────────────────────
st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")

# ──────────────────────────────────────────────
# ⑦ 데이터 개요 카드
# ──────────────────────────────────────────────
if not env_raw.empty:
    st.subheader("🌿 환경 데이터 개요")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 측정 횟수", f"{len(env_mapped):,}회")
    if "time" in env_mapped.columns:
        c2.metric("측정 기간",
                  f"{env_mapped['time'].min():%Y-%m-%d} ~ {env_mapped['time'].max():%Y-%m-%d}")
    else:
        c2.metric("측정 기간", "시간 정보 없음")
    total_cells = env_mapped.shape[0] * env_mapped.shape[1]
    miss = env_mapped.isnull().sum().sum()
    c3.metric("결측치 비율", f"{miss / total_cells * 100:.1f}% ({int(miss)}셀)")

    env_cols_available = [k for k in ["temperature","humidity","ph","ec","water_temp","co2"]
                          if k in env_mapped.columns]
    env_cols_show = st.multiselect(
        "📈 차트에 표시할 환경 변수",
        options=env_cols_available,
        default=[k for k in ["temperature","humidity"] if k in env_mapped.columns],
        help="시계열 차트와 환경-생육 상관에 사용할 변수예요.",
    )
else:
    env_cols_show = []

if not growth_raw.empty:
    st.subheader("📏 생육 결과 데이터 개요")
    c1, c2, c3 = st.columns(3)
    c1.metric("개체 수 (N)", f"{len(growth_mapped):,}개")
    c2.metric("사용 열", len(growth_map))
    total_cells_g = growth_mapped.shape[0] * growth_mapped.shape[1]
    miss_g = growth_mapped.isnull().sum().sum()
    c3.metric("결측치 비율", f"{miss_g / total_cells_g * 100:.1f}% ({int(miss_g)}셀)")

    growth_cols_available = [k for k in ["leaf_count","shoot_length","root_length",
                                          "fresh_weight","shoot_mass","root_mass"]
                             if k in growth_mapped.columns]
    growth_cols_show = st.multiselect(
        "📊 분석할 생육 지표",
        options=growth_cols_available,
        default=[k for k in ["leaf_count","shoot_length","fresh_weight"]
                 if k in growth_mapped.columns],
        help="막대차트·통계 요약표·상관분석에 포함할 지표예요.",
    )
else:
    growth_cols_show = []

st.markdown("---")

# ──────────────────────────────────────────────
# ⑧ [1페이지] 개체별 생육 통계 요약
# ──────────────────────────────────────────────
if not growth_raw.empty and growth_cols_show:
    st.header("📊 개체별 생육 통계 요약")

    stat_cols = growth_mapped[growth_cols_show].describe().T
    stat_cols["변동계수(CV)"] = (stat_cols["std"] / stat_cols["mean"] * 100).round(1)
    stat_cols = stat_cols[["count","mean","std","min","25%","50%","75%","max","변동계수(CV)"]]
    stat_cols.columns = ["개체수","평균","표준편차","최솟값","Q1","중앙값","Q3","최댓값","CV(%)"]
    stat_cols = stat_cols.round(2)

    # 왜도 확인 (치우침 경고)
    skew_vals = growth_mapped[growth_cols_show].skew()
    skewed = [cn for cn, sv in skew_vals.items() if abs(sv) > 1]
    if skewed:
        st.warning(
            f"⚠️ {', '.join(skewed)} 지표가 한쪽으로 치우쳐 있어요(왜도 > 1). "
            "평균만 보면 대표성이 떨어질 수 있으니 **중앙값(Q2)** 도 함께 확인해 주세요."
        )

    with st.expander("📖 통계 지표 설명 보기"):
        for col_kr in ["개체수","평균","표준편차","최솟값","최댓값","Q1","중앙값","Q3","CV(%)"]:
            st.markdown(f"**{col_kr}** — {EXPLAIN.get(col_kr, '')}")

    st.dataframe(stat_cols, use_container_width=True, hide_index=True)

    st.caption("💡 각 행 = 하나의 생육 지표, 각 열 = 통계 요약값이에요. CV(%)가 클수록 개체 간 차이가 커요.")

st.markdown("---")

# ──────────────────────────────────────────────
# ⑨ [2페이지] 개체별 생육 막대차트
# ──────────────────────────────────────────────
if not growth_raw.empty and growth_cols_show:
    st.header("📈 개체별 생육 지표 비교")

    id_col = growth_map.get("id")
    chart_df = growth_mapped.copy()

    if id_col and id_col in chart_df.columns:
        chart_df["label"] = chart_df[id_col].astype(str)
    else:
        chart_df["label"] = [f"개체{i+1}" for i in range(len(chart_df))]

    sel_metric = st.selectbox(
        "막대차트에 표시할 지표 선택",
        options=growth_cols_show,
        help="개체별 값을 막대로 비교할 지표를 고르세요.",
    )

    if sel_metric:
        vals = chart_df[sel_metric].dropna()
        labels = chart_df.loc[vals.index, "label"]
        color_map = plt.cm.viridis(np.linspace(0.15, 0.85, len(vals)))

        fig, ax = plt.subplots(figsize=(max(10, len(vals) * 0.18), 4.5))
        ax.bar(range(len(vals)), vals.values, color=color_map,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"개체별 [{sel_metric}] 분포 (N={len(vals)})",
                     fontsize=13, pad=10)
        ax.set_ylabel(sel_metric)
        ax.axhline(vals.mean(), color="red", linestyle="--", linewidth=1.3,
                   label=f"평균 = {vals.mean():.2f}")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"💡 막대 = 개체별 {sel_metric} 값, 빨간 점선 = 평균. 평균 위 막대는 평균보다 큰 개체예요.")

st.markdown("---")

# ──────────────────────────────────────────────
# ⑩ [3페이지] 생육 지표 간 상관행렬
# ──────────────────────────────────────────────
if not growth_raw.empty and len(growth_cols_show) >= 2:
    st.header("🔗 생육 지표 간 Pearson 상관계수")

    corr_mat = growth_mapped[growth_cols_show].corr(method="pearson").round(3)

    fig2, ax2 = plt.subplots(figsize=(max(5.5, len(growth_cols_show) * 0.7),
                                       max(5.5, len(growth_cols_show) * 0.7)))
    im = ax2.imshow(corr_mat.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(growth_cols_show)))
    ax2.set_xticklabels(growth_cols_show, rotation=40, ha="right", fontsize=9)
    ax2.set_yticks(range(len(growth_cols_show)))
    ax2.set_yticklabels(growth_cols_show, fontsize=9)
    for i in range(len(growth_cols_show)):
        for j in range(len(growth_cols_show)):
            ax2.text(j, i, f"{corr_mat.iloc[i,j]:.2f}",
                     ha="center", va="center",
                     color="white" if abs(corr_mat.iloc[i,j]) > 0.6 else "black",
                     fontsize=10, fontweight="bold")
    ax2.set_title("생육 지표 간 Pearson 상관계수 행렬", fontsize=13, pad=10)
    fig2.colorbar(im, ax=ax2, shrink=0.8, label="r (상관계수)")
    fig2.tight_layout()
    st.pyplot(fig2)

    # 가장 강한 상관이 있는 쌍
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    flat = upper.stack().dropna()
    if len(flat) > 0:
        top_pair = flat.abs().idxmax()
        top_val = flat.loc[top_pair]
        c_a, c_b = top_pair
        direction = "함께 증가" if top_val > 0 else "한쪽 증가 시 다른 쪽 감소"
        st.info(
            f"🔍 가장 강한 상관: **{c_a} ↔ {c_b}** (r = {top_val:+.3f})\n"
            f"→ {direction} 하는 경향이 있어요."
        )

    st.caption("💡 셀 안의 숫자(r)는 두 지표 간 **Pearson 상관계수**예요. "
               "1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽 감소, "
               "0에 가까우면 선형 관계가 거의 없어요.")

    with st.expander("📖 상관계수 해석 가이드"):
        st.markdown("""
        | r 범위 | 관계 강도 | 의미 |
        |--------|-----------|------|
        | 0.7 ~ 1.0 | 강한 양의 상관 | 두 지표가 함께 뚜렷하게 증가 |
        | 0.3 ~ 0.7 | 중간 양의 상관 | 어느 정도 함께 움직임 |
        | 0.0 ~ 0.3 | 약한 상관 | 관계가 거의 없음 |
        | -0.3 ~ 0.0 | 약한 음의 상관 | 거의 관계 없음 |
        | -0.7 ~ -0.3 | 중간 음의 상관 | 한쪽 증가 시 다른 쪽 감소 경향 |
        | -1.0 ~ -0.7 | 강한 음의 상관 | 뚜렷하게 반대 방향 |
        """)

    st.markdown("---")
    st.markdown("💡 상관 ≠ 인과예요. 두 지표가 함께 움직인다고 해서 하나가 다른 하나의 '원인'이라고 단정할 수 없어요.")

# ──────────────────────────────────────────────
# ⑪ [4페이지] 환경 vs 생육 상관 (시간 병합 또는 평균)
# ──────────────────────────────────────────────
if not env_raw.empty and not growth_raw.empty and env_cols_show and growth_cols_show:
    st.header("🌡️ 환경 조건 ↔ 생육 지표 상관 분석")

    # 생육 데이터에 시간 열이 있는지 확인
    time_in_growth = find_col(growth_raw, ["시간", "time", "시각", "측정시간", "측정 시각"])
    has_time = time_in_growth is not None and time_in_growth in growth_mapped.columns

    if has_time:
        growth_mapped["time"] = pd.to_datetime(growth_mapped[time_in_growth], errors="coerce")
        st.caption(f"생육 데이터의 [{time_in_growth}] 열을 기준으로 환경 데이터와 연결해요.")
        merge_method = st.selectbox(
            "연결 방법",
            options=[
                "nearest (가장 가까운 시각 사용)",
                "일별 평균 → 연결",
            ],
            help="환경 값은 시간별로 여러 개 있을 수 있어요.",
        )
        if merge_method == "일별 평균 → 연결":
            env_use = env_mapped.copy()
            env_use["date"] = env_use["time"].dt.date
            env_daily = env_use.groupby("date")[env_cols_show].mean().reset_index()
            env_daily["time"] = pd.to_datetime(env_daily["date"])
            env_use = env_daily
            st.caption("환경 데이터를 **일별 평균**으로 요약한 뒤 연결했어요.")
        else:
            env_use = env_mapped

        merged = pd.merge_asof(
            growth_mapped.dropna(subset=["time"]).sort_values("time").reset_index(drop=True),
            env_use.sort_values("time").reset_index(drop=True),
            on="time", direction="nearest", tolerance=pd.Timedelta("12h"),
        )
        st.caption("개체 측정 시각과 가장 가까운(±12시간) 환경 값을 연결했어요.")
    else:
        # 시간 정보가 없으면 환경 전체 평균을 모든 개체에 적용
        env_mean = env_mapped[env_cols_show].mean(numeric_only=True)
        merged = growth_mapped.copy()
        for col in env_cols_show:
            merged[col] = env_mean[col]
        st.info(
            "⚠️ 생육 데이터에 **시간 열이 없어** 전체 기간의 환경 평균값을 "
            "모든 개체에 동일하게 적용했어요. "
            "정확한 시간별 연결 분석을 원하면 생육 데이터에 시간 열을 추가해 주세요."
        )

    merged_corr_cols = list(dict.fromkeys(
        [c for c in growth_cols_show if c in merged.columns] +
        [c for c in env_cols_show if c in merged.columns]
    ))
    if len(merged_corr_cols) >= 3:
        m_corr = merged[merged_corr_cols].corr(method="pearson").round(3)

        fig3, ax3 = plt.subplots(figsize=(max(6, len(merged_corr_cols) * 0.7),
                                          max(6, len(merged_corr_cols) * 0.7)))
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
        ax3.set_title("환경·생육 통합 상관행렬 (Pearson)", fontsize=13, pad=10)
        fig3.colorbar(im3, ax=ax3, shrink=0.8, label="r")
        fig3.tight_layout()
        st.pyplot(fig3)

        # 생육 ↔ 환경 쌍만 추출
        pairs = []
        for gc in growth_cols_show:
            for ec in env_cols_show:
                if gc in m_corr.columns and ec in m_corr.columns:
                    pairs.append((gc, ec, m_corr.loc[gc, ec]))
        if pairs:
            gev = pd.DataFrame(pairs, columns=["생육지표", "환경변수", "상관계수(r)"])
            gev["상관계수(r)"] = gev["상관계수(r)"].round(3)
            gev = gev.sort_values("상관계수(r)", key=lambda s: s.abs(), ascending=False)
            st.dataframe(gev, use_container_width=True, hide_index=True)
            st.caption("💡 생육 지표와 환경 변수 간 상관계수예요. "
                       "절대값이 큰 쌍일수록 두 변수가 강한 관계로 보여요. "
                       "단, 시간 정보가 없어 평균 값을 쓴 경우이므로 참고용이에요.")

st.markdown("---")

# ──────────────────────────────────────────────
# ⑫ [5페이지] 환경 시계열 차트
# ──────────────────────────────────────────────
if not env_raw.empty and env_cols_show and "time" in env_mapped.columns:
    st.header("🌡️ 환경 조건 시계열 (시간별)")

    n_cols = 3
    n_rows = int(np.ceil(len(env_cols_show) / n_cols))
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.8 * n_rows), squeeze=False)

    for idx, col in enumerate(env_cols_show):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = env_mapped[["time", col]].dropna()
        ax.plot(sub["time"], sub[col], linewidth=0.55,
                color=plt.cm.tab10(idx % 10))
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_ylabel("") if idx % n_cols != 0 else None

    for idx in range(len(env_cols_show), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig4.suptitle("환경 데이터 시간별 변화 (2026-06-16 ~ 2026-10-21)", fontsize=13, y=1.01)
    fig4.tight_layout()
    st.pyplot(fig4)
elif not env_raw.empty and env_cols_show:
    st.header("🌡️ 환경 조건 요약")
    st.caption("⚠️ 시간 열이 없거나 파싱에 실패해 시계열 차트를 그릴 수 없어요.")

st.markdown("---")

# ──────────────────────────────────────────────
# ⑬ [6페이지] 원본 데이터 미리보기
# ──────────────────────────────────────────────
with st.expander("🔬 환경 데이터 원본 미리보기"):
    if not env_raw.empty:
        st.caption(f"열({len(env_raw.columns)}개): {list(env_raw.columns)}\n"
                   f"행 수: {len(env_raw)}행")
        st.dataframe(env_raw.head(20), use_container_width=True)
    else:
        st.caption("업로드된 환경 데이터가 없습니다.")

with st.expander("🔬 생육 결과 데이터 원본 미리보기"):
    if not growth_raw.empty:
        st.caption(f"열({len(growth_raw.columns)}개): {list(growth_raw.columns)}\n"
                   f"개체 수: {len(growth_raw)}개")
        st.dataframe(growth_raw.head(30), use_container_width=True)
    else:
        st.caption("업로드된 생육 데이터가 없습니다.")

st.markdown("---")
st.caption(
    "⚠️ 학교마다 열 이름이 다를 수 있어요. 오류가 나면 CSV 열 이름을 "
    "위 형식(예: `time, temperature, humidity, …` 또는 `개체번호, 잎 수(장), …`)에 맞춰 저장해 주세요.\n"
    "인코움은 utf-8-sig / cp949 순으로 자동 시도합니다.\n"
    "생육 데이터에 시간 열이 있으면 환경-생육 시간 병합 분석이 정확해져요."
)
