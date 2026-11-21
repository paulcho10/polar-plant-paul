#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
극지식물 개체별 생육 차이 분석 대시보드
- 버튼(3탭) 기반 화면 전환
- 인코딩(utf-8-sig/cp949) 자동 시도
- 학교별 열 이름 자동 매핑
- Streamlit Cloud 한글 폰트 자동 설치
- 통계 용어 간략 설명 포함
"""
import warnings, io, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")           # Streamlit Cloud 백엔드 대응
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ══════════════════════════════════════════════
# ① 한글 폰트 설치 (Streamlit Cloud / 로컬)
# ══════════════════════════════════════════════
def _install_korean_font():
    # Linux(Streamlit Cloud)에 Nanum 폰트 설치
    if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        return "NanumGothic"
    try:
        import subprocess
        subprocess.run(
            ["apt-get", "install", "-y", "fonts-nanum"],
            capture_output=True, check=False, timeout=90,
        )
    except Exception:
        pass
    fm._load_fontmanager(try_read_cache=False)
    for cand in ["NanumGothic", "NanumBarunGothic", "NanumSquare", "NanumMyeongjo"]:
        hits = [f for f in fm.fontManager.ttflist if cand in f.name]
        if hits:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"

FONT = _install_korean_font()

# ══════════════════════════════════════════════
# ② 열 이름 매핑 (학교별 차이 대응)
# ══════════════════════════════════════════════
def find_col(df, candidates):
    """candidates 중 df에 있는 첫 열 이름 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    stripped = [cc.strip() for cc in df.columns]
    for i, s in enumerate(stripped):
        for c in candidates:
            if s == c.strip():
                return df.columns[i]
    return None

ENV_COL_MAP = {
    "time":        ["time", "시각", "시간", "날짜", "일자", "datetime"],
    "temperature": ["temperature", "temp", "온도", "기온", "공기온도"],
    "humidity":    ["humidity", "hum", "습도", "상대습도"],
    "ph":          ["ph", "PH", "수소이온농도", "산도"],
    "ec":          ["ec", "EC", "전기전도도"],
    "water_temp":  ["water_temp", "water temperature", "수온", "물온도"],
    "co2":         ["co2", "CO2", "이산화탄소", "co2_ppm"],
}

GROWTH_COL_MAP = {
    "id":           ["개체번호", " 개체번호", "ID", "id", "식물번호", "식물", "개체"],
    "leaf_count":   ["잎 수(장)", "잎 수", "잎수(장)", "잎수", "leaf_count", "leaf"],
    "shoot_length": ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이", "shoot_length"],
    "root_length":  ["지하부길이(mm)", "지하부 길이(mm)", "지하부길이", "지하부 길이", "root_length"],
    "fresh_weight": ["생중량(g)", "생중량", "fresh_weight", "생체량"],
    "shoot_mass":   ["지상부질량(g)", "지상부 질량(g)", "지상부질량", "shoot_mass"],
    "root_mass":    ["지하부질량(g)", "지하부 질량(g)", "지하부질량", "root_mass"],
}

def read_csv_safe(path):
    """인코딩 순서: utf-8-sig → cp949 → utf-8 → euc-kr → pandas 기본"""
    for enc in ["utf-8-sig", "cp949", "utf-8", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[0] > 0:
                return df
        except Exception:
            continue
    df = pd.read_csv(path)
    st.warning("CSV 인코딩을 자동 감지했어요. 글자가 깨지면 utf-8-sig 인코딩으로 다시 저장해 주세요.")
    return df

def map_cols(df_raw, col_map):
    return {k: v for k, v in ((k, find_col(df_raw, cand)) for k, cand in col_map.items()) if v}

# ══════════════════════════════════════════════
# ③ 통계 용어 설명
# ══════════════════════════════════════════════
EXPLAIN = {
    "개체수":   "개체 수(N) — 분석에 포함된 식물 개체의 총 개수예요.",
    "평균":     "평균 — 모든 값을 더해 개수로 나눈 대표값이에요. 극단값(아주 크거나 작은 값)에 영향을 받아요.",
    "표준편차": "표준편차 — 값들이 평균 주위에 얼마나 흩어져 있는지 나타내요. 클수록 개체 간 차이가 커요.",
    "최솟값":   "최솟값 — 데이터에서 가장 작은 값이에요.",
    "최댓값":   "최댓값 — 데이터에서 가장 큰 값이에요.",
    "Q1":       "Q1(제1사분위수) — 작은 값부터 나열했을 때 25% 지점의 값. 하위 25%의 경계예요.",
    "중앙값":   "중앙값(Q2) — 정가운데 값. 극단값의 영향을 덜 받아요.",
    "Q3":       "Q3(제3사분위수) — 75% 지점의 값. 상위 25%의 경계예요.",
    "CV(%)":    "변동계수(CV) — 표준편차를 평균으로 나누고 100을 곱한 값. 단위가 다른 지표도 상대 비교할 수 있어요.",
    "상관계수": "상관계수(r, Pearson) — 두 변수가 함께 움직이는 정도(-1~1). +1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽 감소, 0이면 관계 없음이에요.",
}

# ══════════════════════════════════════════════
# ④ 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="극지식물 생육 대시보드",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")

# ══════════════════════════════════════════════
# ⑤ 사이드바: 파일 업로드
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("📂 파일 업로드")
    uploaded_env    = st.file_uploader("🌿 환경 데이터 (CSV)", type=["csv"])
    uploaded_growth = st.file_uploader("📏 생육 결과 데이터 (CSV)", type=["csv"])
    st.markdown("---")
    st.caption(
        "열 이름 예시 (영문 기반): `time, temperature, humidity, ph, ec, water_temp, co2`\n"
        "열 이름 예시 (한글 기반): `개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)`\n"
        "인코딩: utf-8-sig / cp949 자동 시도"
    )

if uploaded_env is None and uploaded_growth is None:
    st.info("왼쪽 사이드바에서 **환경 데이터**와 **생육 결과 데이터** CSV 파일을 모두 업로드해 주세요.")
    st.stop()

# ══════════════════════════════════════════════
# ⑥ 파일 읽기 & 열 매핑
# ══════════════════════════════════════════════
env_raw = pd.DataFrame()
if uploaded_env is not None:
    try:
        env_raw = read_csv_safe(uploaded_env)
    except Exception as e:
        st.error(f"환경 데이터 읽기 오류: {e}")
        st.stop()

growth_raw = pd.DataFrame()
if uploaded_growth is not None:
    try:
        growth_raw = read_csv_safe(uploaded_growth)
    except Exception as e:
        st.error(f"생육 데이터 읽기 오류: {e}")
        st.stop()

env_map    = map_cols(env_raw, ENV_COL_MAP)    if not env_raw.empty    else {}
growth_map = map_cols(growth_raw, GROWTH_COL_MAP) if not growth_raw.empty else {}

env_mapped    = env_raw.rename(columns={v: k for k, v in env_map.items()})    if env_map    else pd.DataFrame()
growth_mapped = growth_raw.rename(columns={v: k for k, v in growth_map.items()}) if growth_map else pd.DataFrame()

# 시간 파싱
if "time" in env_mapped.columns:
    env_mapped["time"] = pd.to_datetime(env_mapped["time"], errors="coerce")
    if env_mapped["time"].isna().any():
        n_drop = env_mapped["time"].isna().sum()
        env_mapped = env_mapped.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        st.warning(f"시간 형식이 잘못된 {n_drop}행을 제외했어요.")
    else:
        env_mapped = env_mapped.sort_values("time").reset_index(drop=True)

# 생중량(g) 없으면 계산
if "fresh_weight" not in growth_mapped.columns:
    if "shoot_mass" in growth_mapped.columns and "root_mass" in growth_mapped.columns:
        growth_mapped["fresh_weight"] = (
            pd.to_numeric(growth_mapped["shoot_mass"], errors="coerce")
            + pd.to_numeric(growth_mapped["root_mass"], errors="coerce")
        )
        st.sidebar.info("⚠️ '생중량(g)' 열이 없어 **지상부질량 + 지하부질량**으로 계산했어요.")
    elif "shoot_mass" in growth_mapped.columns:
        growth_mapped["fresh_weight"] = pd.to_numeric(growth_mapped["shoot_mass"], errors="coerce")
        st.sidebar.info("⚠️ '생중량(g)'과 '지하부질량(g)'이 모두 없어 **지상부질량**만 사용했어요.")

# 숫자형 변환
for col in ["leaf_count","shoot_length","root_length","fresh_weight","shoot_mass","root_mass"]:
    if col in growth_mapped.columns:
        growth_mapped[col] = pd.to_numeric(growth_mapped[col], errors="coerce")

# ── 사용 가능 열 ──────────────────────────────
ENV_COLS     = [k for k in ["temperature","humidity","ph","ec","water_temp","co2"] if k in env_mapped.columns]
GROWTH_COLS  = [k for k in ["leaf_count","shoot_length","root_length","fresh_weight","shoot_mass","root_mass"]
                if k in growth_mapped.columns]

# ══════════════════════════════════════════════
# ⑦ 3탭 버튼 (수평 배치)
# ══════════════════════════════════════════════
tab_choice = st.radio(
    "📌 분석할 내용을 선택하세요",
    options=["① 개체 생육 통계 · 막대차트 · 상관행렬",
             "② 환경 조건 vs 생육 상관 · 환경 시계열",
             "③ 원본 데이터 미리보기"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    help="탭을 선택하면 해당 분석 화면으로 이동해요.",
)

st.markdown("---")

# ──────────────────────────────────────────────
#  탭 1 — 개체 생육 통계 · 막대차트 · 상관행렬
# ──────────────────────────────────────────────
if tab_choice.startswith("①"):

    if not growth_raw.empty:
        st.header("📊 개체별 생육 통계 요약")

        stat_df = growth_mapped[GROWTH_COLS].describe().T
        stat_df["변동계수(CV)"] = (stat_df["std"] / stat_df["mean"] * 100).round(1)
        stat_df = stat_df[["count","mean","std","min","25%","50%","75%","max","변동계수(CV)"]]
        stat_df.columns = ["개체수","평균","표준편차","최솟값","Q1","중앙값","Q3","최댓값","CV(%)"]
        stat_df = stat_df.round(2)

        # 왜도 경고
        skew = growth_mapped[GROWTH_COLS].skew()
        skewed_cols = [c for c, v in skew.items() if abs(v) > 1]
        if skewed_cols:
            st.warning(
                f"⚠️ {', '.join(skewed_cols)} 지표가 한쪽으로 치우쳐 있어요(왜도 > 1). "
                "평균만 보면 대표성이 떨어질 수 있으니 **중앙값(Q2)**도 함께 확인해 주세요."
            )

        st.dataframe(stat_df, use_container_width=True, hide_index=True)

        with st.expander("📖 통계 용어 설명"):
            for k, v in EXPLAIN.items():
                if k in ["개체수","평균","표준편차","최솟값","최댓값","Q1","중앙값","Q3","CV(%)"]:
                    st.markdown(f"**{k}** — {v}")

        st.markdown("---")

        # ── 개체별 막대차트 ──────────────────────
        st.header("📈 개체별 생육 지표 비교")

        id_col = growth_map.get("id")
        chart_df = growth_mapped.copy()
        chart_df["label"] = (chart_df[id_col].astype(str) if id_col and id_col in chart_df.columns
                              else [f"개체{i+1}" for i in range(len(chart_df))])

        sel_metric = st.selectbox(
            "막대차트에 표시할 지표 선택",
            options=GROWTH_COLS,
        )
        if sel_metric:
            vals = chart_df[sel_metric].dropna()
            labels = chart_df.loc[vals.index, "label"]
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(vals)))

            fig, ax = plt.subplots(figsize=(max(9, len(vals) * 0.16), 4.2))
            ax.bar(range(len(vals)), vals.values, color=colors,
                   edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"개체별 [{sel_metric}] 분포  (N = {len(vals)})",
                         fontsize=13, pad=8)
            ax.set_ylabel(sel_metric)
            ax.axhline(vals.mean(), color="red", linestyle="--", linewidth=1.3,
                       label=f"평균 = {vals.mean():.2f}")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption(
                f"💡 막대 = 개체별 {sel_metric} 값, 빨간 점선 = 평균. "
                "평균 위 막대는 평균보다 큰 개체예요."
            )

        st.markdown("---")

        # ── 생육 지표 간 상관행렬 ────────────────
        if len(GROWTH_COLS) >= 2:
            st.header("🔗 생육 지표 간 Pearson 상관계수")

            corr_mat = growth_mapped[GROWTH_COLS].corr(method="pearson").round(3)
            n = len(GROWTH_COLS)
            fig2, ax2 = plt.subplots(figsize=(max(5.5, n * 0.7), max(5.5, n * 0.7)))
            im = ax2.imshow(corr_mat.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
            ax2.set_xticks(range(n))
            ax2.set_xticklabels(GROWTH_COLS, rotation=40, ha="right", fontsize=9)
            ax2.set_yticks(range(n))
            ax2.set_yticklabels(GROWTH_COLS, fontsize=9)
            for i in range(n):
                for j in range(n):
                    ax2.text(j, i, f"{corr_mat.iloc[i,j]:.2f}",
                             ha="center", va="center",
                             color="white" if abs(corr_mat.iloc[i,j]) > 0.6 else "black",
                             fontsize=10, fontweight="bold")
            ax2.set_title("생육 지표 간 Pearson 상관계수 행렬", fontsize=13, pad=10)
            fig2.colorbar(im, ax=ax2, shrink=0.8, label="r (상관계수)")
            fig2.tight_layout()
            st.pyplot(fig2)

            # 가장 강한 상관 쌍
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            flat = upper.stack().dropna()
            if len(flat) > 0:
                top_pair = flat.abs().idxmax()
                top_val  = flat.loc[top_pair]
                c_a, c_b = top_pair
                direction = "함께 증가" if top_val > 0 else "한쪽 증가 시 다른 쪽 감소"
                st.info(
                    f"🔍 가장 강한 상관: **{c_a} ↔ {c_b}**  (r = {top_val:+.3f})\n"
                    f"→ {direction} 하는 경향이 있어요."
                )

            st.caption(
                "💡 셀 안의 숫자(r)는 두 지표 간 **Pearson 상관계수**예요. "
                "1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽 감소, "
                "0에 가까우면 선형 관계가 거의 없어요."
            )

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

            st.markdown("💡 **상관 ≠ 인과**예요. 두 지표가 함께 움직인다고 해서 하나가 다른 하나의 원인이라고 단정할 수 없어요.")

    else:
        st.info("생육 결과 데이터를 업로드해 주세요.")

# ──────────────────────────────────────────────
#  탭 2 — 환경 조건 vs 생육 상관 · 환경 시계열
# ──────────────────────────────────────────────
elif tab_choice.startswith("②"):

    if not env_raw.empty and not growth_raw.empty:
        st.header("🌡️ 환경 조건 vs 생육 지표 상관 분석")

        # 생육 데이터에 시간 열이 있는지 확인
        time_in_growth = find_col(growth_raw, ["시간", "time", "시각", "측정시간", "측정 시각"])
        has_time = time_in_growth is not None and time_in_growth in growth_mapped.columns

        if not ENV_COLS or not GROWTH_COLS:
            st.warning("사용 가능한 환경/생육 열이 없어 분석을 진행할 수 없어요.")
        else:
            if has_time:
                growth_mapped["time"] = pd.to_datetime(
                    growth_mapped[time_in_growth], errors="coerce"
                )
                st.caption(f"생육 데이터의 [{time_in_growth}] 열 기준으로 환경 데이터와 연결해요.")

                merge_method = st.selectbox(
                    "환경 데이터 요약 방법",
                    options=["각 측정 시각 그대로 사용 (nearest)",
                             "일별 평균으로 요약 후 연결"],
                    help="환경 값에 시간별 여러 건 있을 때 요약 방식이에요.",
                )
                if merge_method == "일별 평균으로 요약 후 연결":
                    env_use = env_mapped.copy()
                    env_use["date"] = env_use["time"].dt.date
                    env_daily = env_use.groupby("date")[ENV_COLS].mean().reset_index()
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
                env_mean = env_mapped[ENV_COLS].mean(numeric_only=True)
                merged = growth_mapped.copy()
                for col in ENV_COLS:
                    merged[col] = env_mean[col]
                st.info(
                    "⚠️ 생육 데이터에 **시간 열이 없어** 전체 기간의 환경 평균값을 "
                    "모든 개체에 동일하게 적용했어요. "
                    "정확한 시간별 분석을 원하면 생육 데이터에 시간 열을 추가해 주세요."
                )

            merged_corr_cols = list(dict.fromkeys(
                [c for c in GROWTH_COLS if c in merged.columns]
                + [c for c in ENV_COLS if c in merged.columns]
            ))
            if len(merged_corr_cols) >= 3:
                m_corr = merged[merged_corr_cols].corr(method="pearson").round(3)
                n = len(merged_corr_cols)
                fig3, ax3 = plt.subplots(figsize=(max(6, n * 0.7), max(6, n * 0.7)))
                im3 = ax3.imshow(m_corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
                ax3.set_xticks(range(n))
                ax3.set_xticklabels(merged_corr_cols, rotation=45, ha="right", fontsize=8)
                ax3.set_yticks(range(n))
                ax3.set_yticklabels(merged_corr_cols, fontsize=8)
                for i in range(n):
                    for j in range(n):
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
                for gc in GROWTH_COLS:
                    for ec in ENV_COLS:
                        if gc in m_corr.columns and ec in m_corr.columns:
                            pairs.append((gc, ec, m_corr.loc[gc, ec]))
                if pairs:
                    gev = pd.DataFrame(pairs, columns=["생육지표", "환경변수", "상관계수(r)"])
                    gev["상관계수(r)"] = gev["상관계수(r)"].round(3)
                    gev = gev.sort_values("상관계수(r)", key=lambda s: s.abs(), ascending=False)
                    st.dataframe(gev, use_container_width=True, hide_index=True)
                    st.caption(
                        "💡 생육 지표와 환경 변수 간 상관계수예요. "
                        "절대값이 큰 쌍일수록 두 변수가 강한 관계로 보여요. "
                        "단, 시간 정보가 없어 평균 값을 쓴 경우이므로 참고용이에요."
                    )

        st.markdown("---")

        # ── 환경 시계열 차트 ──────────────────────
        if "time" in env_mapped.columns and ENV_COLS:
            st.header("🌡️ 환경 조건 시계열 (시간별)")

            n_cols = 3
            n_rows = int(np.ceil(len(ENV_COLS) / n_cols))
            fig4, axes = plt.subplots(n_rows, n_cols,
                                       figsize=(14, 2.8 * n_rows), squeeze=False)

            for idx, col in enumerate(ENV_COLS):
                r, c = divmod(idx, n_cols)
                ax = axes[r][c]
                sub = env_mapped[["time", col]].dropna()
                ax.plot(sub["time"], sub[col], linewidth=0.55,
                        color=plt.cm.tab10(idx % 10))
                ax.set_title(col, fontsize=10)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle=":", alpha=0.4)

            for idx in range(len(ENV_COLS), n_rows * n_cols):
                r, c = divmod(idx, n_cols)
                axes[r][c].set_visible(False)

            fig4.suptitle("환경 데이터 시간별 변화",
                          fontsize=13, y=1.01)
            fig4.tight_layout()
            st.pyplot(fig4)

    else:
        if not env_raw.empty:
            st.info("생육 결과 데이터를 함께 업로드해 주세요.")
        elif not growth_raw.empty:
            st.info("환경 데이터를 함께 업로드해 주세요.")
        else:
            st.info("환경 데이터와 생육 결과 데이터를 모두 업로드해 주세요.")

# ──────────────────────────────────────────────
#  탭 3 — 원본 데이터 미리보기
# ──────────────────────────────────────────────
elif tab_choice.startswith("③"):

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🔬 환경 데이터 원본 미리보기"):
            if not env_raw.empty:
                st.caption(f"열({len(env_raw.columns)}개): {list(env_raw.columns)}\n행 수: {len(env_raw)}행")
                st.dataframe(env_raw.head(20), use_container_width=True)
            else:
                st.caption("업로드된 환경 데이터가 없습니다.")
    with c2:
        with st.expander("🔬 생육 결과 데이터 원본 미리보기"):
            if not growth_raw.empty:
                st.caption(f"열({len(growth_raw.columns)}개): {list(growth_raw.columns)}\n개체 수: {len(growth_raw)}개")
                st.dataframe(growth_raw.head(30), use_container_width=True)
            else:
                st.caption("업로드된 생육 데이터가 없습니다.")

    st.markdown("---")
    st.caption(
        "⚠️ 학교마다 열 이름이 다를 수 있어요. 오류가 나면 CSV 열 이름을 "
        "위 예시 형식에 맞춰 저장해 주세요.\n"
        "인코딩은 utf-8-sig / cp949 순으로 자동 시도합니다.\n"
        "생육 데이터에 시간 열이 있으면 환경-생육 시간 병합 분석이 정확해져요."
    )
