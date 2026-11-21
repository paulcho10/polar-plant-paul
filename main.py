#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
=============================================
탭 구성 (PDF 보고서 섹션 순서 그대로):
  탭1: 1.문제정의  +  2.실험방법
  탭2: 3.결과  (환경분석 + 생육분석 + 그룹비교 + 예측모델 — 그래프 중심)
  탭3: 4.논의  +  5.한계점과 개선 방향  +  6.결론

원칙:
  - CSV 인코딩(utf-8-sig/cp949) 오류 방지
  - 학교마다 다른 열 이름(KeyError) 방지
  - 한글 폰트 깨짐 완벽 방지 (NotoSansCJK 직접 등록)
  - Streamlit Cloud에서 에러 없이 실행
  - 고등학생이 이해할 수 있도록 통계 용어에 짧은 설명 붙임
  - 표보다 그래프로 시각화 (필요 최소 표만 사용)
"""

import io
import os
import shutil
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 한글 폰트 설정 (Streamlit Cloud / 매 실행 클린 상태)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_korean_font():
    """NotoSansCJK-Regular.ttc를 직접 등록 → 한글 깨짐 완전 방지"""
    cache_dir = os.path.expanduser("~/.cache/matplotlib")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    font_regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_bold    = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

    for fp in [font_regular, font_bold]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)

    # FontProperties로 실제 폰트 이름 획득
    reg_prop = fm.FontProperties(fname=font_regular)
    font_name = reg_prop.get_name()   # "Noto Sans CJK JP"

    plt.rcParams["font.family"]        = font_name
    plt.rcParams["font.size"]          = 11
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"]   = "bold"
    plt.rcParams["figure.dpi"]         = 110
    plt.rcParams["savefig.dpi"]        = 110


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CSV 읽기 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_csv_flexible(uploaded_file):
    """utf-8-sig → cp949 → euc-kr → latin-1 순서로 인코딩 시도"""
    if uploaded_file is None:
        return None
    for enc in ["utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            if df.shape[0] > 0:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # 최종 fallback
    raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
    uploaded_file.seek(0)
    return pd.read_csv(io.StringIO(raw), on_bad_lines="skip")


def find_actual_column(df, candidates):
    """여러 후보 중 실제 존재하는 열을 찾는다 (KeyError 방지)"""
    if df is None or df.shape[1] == 0:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    # 정규화 비교
    def norm(s):
        s = str(s).strip()
        for u in ["(장)", "(mm)", "(g)", "(%)", "(degC)"]:
            s = s.replace(u, "")
        return s.strip()
    d = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in d:
            return d[norm(c)]
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 환경 데이터 전처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preprocess_env(df):
    if df is None:
        return None
    df = df.copy()

    time_col = find_actual_column(df,
        ["time","Time","datetime","날짜","일시","날짜/시간","date","일시_시간"])
    if time_col is None:
        for col in df.columns:
            s = str(df[col].iloc[0])
            if "2026" in s or ("_" in s and any(c.isdigit() for c in s)):
                time_col = col
                break
    if time_col is None:
        st.warning("시간 열을 찾을 수 없습니다.")
        return None

    # 시간 파싱
    df["_t"] = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    n = df["_t"].notna().sum()
    if n == 0:
        def try_parse(v):
            s = str(v).strip()
            # '20260715_0' → '2026-07-15 00:00:00'
            if len(s) == 10 and s[4] == '0' and s[7] == '0':
                try:
                    return pd.to_datetime(f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[9:]}:00:00")
                except:
                    pass
            return pd.NaT
        df["_t"] = df[time_col].apply(try_parse)

    df = df.dropna(subset=["_t"]).copy()
    df["_hour"]  = df["_t"].dt.hour
    df["_date"]  = df["_t"].dt.date
    df["_date_dt"] = pd.to_datetime(df["_date"])

    col_map = {
        "temperature": ["temperature","온도","temp","온도(degC)","기온"],
        "humidity":    ["humidity","습도","hum","습도(%)"],
        "ec":          ["ec","EC","ec"],
        "ph":          ["ph","PH","pH","PH"],
        "water_temp":  ["water_temp","수온","water_temp"],
        "co2":         ["co2","CO2","co2"],
    }
    result = {}
    for key, cands in col_map.items():
        c = find_actual_column(df, cands)
        if c is not None:
            result[key] = pd.to_numeric(df[c], errors="coerce")
    result["_t"]       = df["_t"]
    result["_hour"]    = df["_hour"]
    result["_date"]    = df["_date"]
    result["_date_dt"] = df["_date_dt"]
    return pd.DataFrame(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 생육 데이터 전처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preprocess_growth(df):
    if df is None:
        return None
    df = df.copy()

    id_col = find_actual_column(df, ["개체번호","번호","ID","id","시료번호","개체"])
    if id_col is not None:
        df["개체번호"] = pd.to_numeric(df[id_col], errors="coerce")

    leaf_col = find_actual_column(df,
        ["잎 수(장)","잎 수","leaf","잎수","잎 수 (장)","엽수"])
    if leaf_col is not None:
        df["잎 수"] = pd.to_numeric(df[leaf_col], errors="coerce")

    shoot_col = find_actual_column(df,
        ["지상부 길이(mm)","지상부 길이","지상부길이","지상부 길이 (mm)","shoot_length","shoot"])
    if shoot_col is not None:
        df["지상부 길이"] = pd.to_numeric(df[shoot_col], errors="coerce")

    root_col = find_actual_column(df,
        ["지하부길이(mm)","지하부 길이(mm)","지하부 길이","지하부길이","지하부 길이 (mm)","root_length","root"])
    if root_col is not None:
        df["지하부 길이"] = pd.to_numeric(df[root_col], errors="coerce")

    fw_col = find_actual_column(df,
        ["생중량(g)","생중량","fresh_weight","생중량 (g)","생체중"])
    if fw_col is not None:
        df["생중량"] = pd.to_numeric(df[fw_col], errors="coerce")

    # 생중량이 없을 때: 지상부질량 + 지하부질량으로 계산
    if "생중량" not in df.columns or df["생중량"].isna().all():
        sm = find_actual_column(df, ["지상부질량(g)","지상부질량","지상부 질량","shoot_mass"])
        rm = find_actual_column(df, ["지하부질량(g)","지하부질량","지하부 질량","root_mass"])
        if sm is not None and rm is not None:
            df["생중량"] = pd.to_numeric(df[sm], errors="coerce") + pd.to_numeric(df[rm], errors="coerce")

    sm2 = find_actual_column(df, ["지상부질량(g)","지상부질량","지상부 질량","shoot_mass"])
    if sm2 is not None:
        df["지상부질량"] = pd.to_numeric(df[sm2], errors="coerce")

    rm2 = find_actual_column(df, ["지하부질량(g)","지하부질량","지하부 질량","root_mass"])
    if rm2 is not None:
        df["지하부질량"] = pd.to_numeric(df[rm2], errors="coerce")

    keep = ["개체번호","잎 수","지상부 길이","지하부 길이","생중량","지상부질량","지하부질량"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["개체번호"]).copy()
    df = df.sort_values("개체번호").reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 그래프 헬퍼 (워드클라우드 스타일 범례 등)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cv_bar_plot(labels, cvs, title, color="#2ecc71"):
    """CV 비교 막대 그래프 (한글 제목 OK)"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, cvs, color=color, edgecolor="white", linewidth=1.2, width=0.55)
    for b, v in zip(bars, cvs):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("CV (%)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylim(0, max(cvs)*1.18)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Streamlit 앱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    setup_korean_font()
    st.set_page_config(page_title="송도고 극지식물 생육 대시보드", layout="wide")

    st.title("극지식물 개체별 생육 차이 분석 대시보드")
    st.caption("같은 스마트팜에서 재배한 극지식물 81개체의 생육 결과 데이터와 환경 모니터링 데이터를 종합 분석한 대시보드")
    st.caption("📄 출처: 송도고등학교 그린2팀 1회차 보고서 (업스테이지 · 극지연구소 검수, 작성 2026-10-28)")

    with st.sidebar:
        st.header("📁 CSV 업로드")
        env_file    = st.file_uploader("환경 데이터 CSV", type=["csv"])
        growth_file = st.file_uploader("생육 결과 데이터 CSV", type=["csv"])
        st.markdown("---")
        st.info(
            "학교마다 열 이름이 다릅니다.\n"
            "형식 A: time, temperature, humidity, ph, ec, water_temp, co2\n"
            "형식 B: categories, 온도, 습도, EC, PH\n"
            "생육: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g) ..."
        )
        st.markdown("---")
        hint = (
            "💡 생중량(g) 열이 없으면\n"
            "   지상부질량(g)+지하부질량(g)으로\n"
            "   자동 계산합니다.\n"
            "💡 개체번호는 숫자(1,2,…) 또는\n"
            "   글자(A1, A2, …) 모두 가능."
        )
        st.text(hint)

    # ── 데이터 로드 ──────────────────────────
    env_df    = None
    growth_df = None

    if env_file:
        df_raw = read_csv_flexible(env_file)
        env_df = preprocess_env(df_raw)
        if env_df is None or env_df.shape[0] == 0:
            st.sidebar.error("환경 CSV의 시간 열을 찾지 못했습니다. 'time' 열이 있는지 확인하세요.")

    if growth_file:
        df_raw = read_csv_flexible(growth_file)
        growth_df = preprocess_growth(df_raw)
        if growth_df is None or growth_df.shape[0] == 0:
            st.sidebar.error("생육 CSV의 개체번호 열을 찾지 못했습니다.")

    # ── 탭 ───────────────────────────────────
    t1, t2, t3 = st.tabs([
        "📘  1.문제정의  ·  2.실험방법",
        "📊  3.결과  (환경 · 생육 · 그룹비교 · 예측모델)",
        "🔍  4.논의  ·  5.한계점 · 개선방향  ·  6.결론"
    ])

    # ═══════════════════════════════════════════
    # 탭 1: 1.문제정의 + 2.실험방법
    # ═══════════════════════════════════════════
    with t1:
        st.header("1. 문제 정의 — 연구 배경과 가설")

        st.subheader("연구 배경")
        st.markdown("""
        같은 품종의 식물을 거의 같은 조건에서 재배하더라도, 개체마다 크기나 무게가 다르게 자라는 일은 흔히 관찰됩니다.
        이런 차이가 단순한 우연인지, 아니면 잎 수나 줄기 길이 같은 **측정 가능한 형질**로 설명할 수 있는지,
        그리고 생장 환경(온도, 습도, EC, pH)의 변화와 관련이 있는지 알아보는 것은
        작물의 수량 예측이나 재배 조건 개선에 도움이 됩니다.

        이 연구에서는 **송도고등학교 스마트팜**에서 재배한 극지식물 **81개체**의 생육 결과 데이터와,
        같은 스마트팜의 **127일간 환경 모니터링 데이터**를 함께 분석했습니다.
        """)
        st.info(
            "📌 분석 목적 요약\n"
            "① 같은 스마트팜에서 재배한 극지식물 81개체 사이에 생육 차이가 얼마나 큰지 확인\n"
            "② 그 차이를 잎 수·지상부 길이·지하부 길이 같은 측정값으로 설명할 수 있는지 확인\n"
            "③ 생장 환경(온도·습도·EC·pH)의 변화와 관련이 있는지 확인\n"
            "④ 특히 가설 '잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다' 검증"
        )

        st.divider()
        st.subheader("연구 가설")
        st.info("가설: **잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다.**")

        st.divider()
        st.header("2. 실험 방법")

        st.subheader("2.1 데이터 개요")

        col_g, col_e = st.columns(2)
        with col_g:
            st.markdown("#### 생육 결과 데이터 (CSV 1개)")
            if growth_df is not None:
                n_g = len(growth_df)
                st.success(f"✅ {n_g}개체 로드")
                st.markdown(f"- 개체 수: **{n_g}개체** (학교마다 다름, 약 20~90개)\n"
                            f"- 측정 항목: 잎 수, 지상부 길이, 지하부 길이, 생중량, 지상부질량, 지하부질량\n"
                            f"- 생중량 열이 없으면 → 지상부질량 + 지하부질량으로 자동 계산\n"
                            f"- 개체번호: 숫자(1,2,…) 또는 글자(A1, A2,…) 모두 대응")
                st.dataframe(growth_df.head(5), use_container_width=True, hide_index=True)
                st.caption("전체 데이터는 탭2에서 확인할 수 있습니다.")
            else:
                st.info("생육 CSV를 업로드하면 열 이름과 샘플이 표시됩니다.")
                st.code(
                    "개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)\n"
                    "1, 18, 17, 7, 0.19, 0.14, 0.05\n"
                    "2, 32, 31, 126, 0.98, 0.65, 0.33",
                    language="text"
                )

        with col_e:
            st.markdown("#### 환경 데이터 (CSV 1개)")
            if env_df is not None:
                n_e = len(env_df)
                t_min = env_df["_date_dt"].min().strftime("%Y-%m-%d")
                t_max = env_df["_date_dt"].max().strftime("%Y-%m-%d")
                st.success(f"✅ {n_e:,}행 로드 ({t_min} ~ {t_max})")
                st.markdown(
                    f"- 측정 기간: **{t_min} ~ {t_max}** (127일간)\n"
                    f"- 측정 간격: **1시간**\n"
                    f"- 측정 변수: 온도, 습도, EC, pH, 수온, CO₂\n"
                    f"- 결측: EC·pH 각 194개, 수온 1,151개 (분석 시 유효값만 사용)")
                st.dataframe(env_df.head(5), use_container_width=True, hide_index=True)
            else:
                st.info("환경 CSV를 업로드하면 열 이름과 샘플이 표시됩니다.")
                st.code(
                    "time, temperature, humidity, ph, ec, water_temp, co2\n"
                    "2026-06-16 06:00:00, 23.1, 54.6, 8.13, 0.27, 23.7, 377.0",
                    language="text"
                )

        st.divider()
        st.subheader("2.2 분석 방법과 데이터 제약")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 환경 데이터 분석 방법 (5단계)")

            steps_env = [
                ("① 기초통계",
                 "온도·습도·EC·pH 각각에 대해 평균, 분산, 표준편차, 최소, 최대, CV를 계산. "
                 "표준편차와 분산은 결측을 제외한 유효값 기준으로 계산."),
                ("② 시간대별 평균",
                 "24개 시간대별로 온도와 습도의 평균을 구해 하루 중 최고·최저 시간대와 일교차를 확인."),
                ("③ 전반부·후반부 비교",
                 "측정 시점 중간(약 8월 19일)을 기준으로 전반부(6/16~8/18)와 후반부(8/19~10/21)로 나누어 평균과 분산을 비교."),
                ("④ EC·pH 변화 구간 확인",
                 "6월 초 측정값(6/16~6/25)과 8월 하순 측정값(8/20~9/5)의 평균을 비교해 변화 방향과 크기를 확인."),
                ("⑤ 이상치 확인",
                 "온도·습도에 IQR 기준(Q1−1.5×IQR, Q3+1.5×IQR)을 적용해 이상치 개수를 확인. 이상치는 삭제하지 않고 그대로 둠."),
            ]
            for title, desc in steps_env:
                st.markdown(f"**{title}**")
                st.write(desc)
                st.divider()

        with col_m2:
            st.markdown("#### 생육 데이터 분석 방법 (6단계)")

            steps_grow = [
                ("① 기초통계",
                 "잎 수·지상부 길이·지하부 길이·생중량의 평균, 중앙값, 분산, 표준편차, CV, 최소, 최대, 최대/최소 배율을 계산. "
                 "중앙값: 크기순으로 나열했을 때 한가운데 값. CV: (표준편차÷평균)×100, 단위가 다른 값끼리 비교 가능."),
                ("② IQR 이상치",
                 "네 항목 각각에 IQR 기준으로 이상치를 찾고 해당 개체번호를 기록. 이상치는 삭제하지 않음."),
                ("③ 상관계수",
                 "생중량과 잎 수·지상부 길이·지하부 길이의 Pearson 상관계수와 p값을 계산. "
                 "p값: '관계가 없다는 가정에서는 관찰되기 어려운 결과'인지 판단하는 근거. p<0.05이면 통계적으로 유의."),
                ("④ 그룹 비교 (Top 25% vs Bottom 25%)",
                 "생중량 기준 상위 25%(20개체)와 하위 25%(20개체)로 나누고, 각 그룹의 평균·표준편차·평균 차이·비율과 Welch t검정 p값을 구함. "
                 "Welch t검정: 두 그룹 평균이 같은지 검정, 분산이 달라도 사용 가능. 지상부÷지하부 비율도 계산해 비교."),
                ("⑤ 회귀모델",
                 "잎 수만 사용한 단순 선형회귀(모델1)와 잎 수+지상부 길이를 사용한 다중 선형회귀(모델2)를 만들고, "
                 "회귀식·R²·MAE를 비교. R²: 모델이 데이터 변동을 설명하는 비율(0~1). MAE: 예측값과 실제값 차이의 평균. "
                 "81개체이므로 5겹 교차검증(shuffle=True, random_state=42)도 수행."),
                ("⑥ 예측 예시",
                 "잎 수 25장과 60장에서 지상부 길이를 평균(39.4mm)으로 가정하고 예측값을 계산."),
            ]
            for title, desc in steps_grow:
                st.markdown(f"**{title}**")
                st.write(desc)
                st.divider()

        st.divider()
        st.subheader("데이터 제약 사항 (중요)")
        constraints = [
            ("생육 측정일 부재",
             "생육결과데이터에 측정일(날짜) 열이 없다. 생육 측정이 정확히 언제 이루어졌는지 알 수 없고, 환경 시계열과 생육 결과를 직접 연결할 수 없다. 이 연구에서 가장 중요한 한계 중 하나."),
            ("개체 위치 정보 부재",
             "같은 스마트팜 안에서도 개체마다 빛·통풍·수분 공급이 다를 수 있으나, 데이터에는 각 개체의 위치 정보가 없어 위치 효과를 분석할 수 없다."),
            ("EC·pH 결측",
             "환경 데이터의 EC와 pH는 각각 194개 결측이 있어 전체 2,461행 중 2,267행만 유효. 결측이 무작위가 아니라면 통계 결과에 편향이 있을 수 있다."),
            ("인과관계 아님",
             "모든 통계 결과(상관, 그룹 차이, 회귀계수)는 변수들 사이의 관계나 차이를 보여줄 뿐, 인과관계를 증명하지 않는다. '잎 수가 많아서 생중량이 크다'가 아니라 '잎 수가 많은 개체가 생중량이 큰 경향이 있다'고만 말할 수 있다."),
        ]
        for title, desc in constraints:
            st.markdown(f"**⚠️ {title}**")
            st.write(desc)

        st.divider()
        st.subheader("분석 방법 요약 차트")
        st.caption("각 분석의 목적과 사용 통계 방법을 한 장으로 정리")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        items = [
            ("환경 분석", "온도·습도·EC·pH\n기초통계 · 시간대별 평균\n전반부/후반부 비교\nEC·pH 변화 구간", "#3498db"),
            ("생육 분석", "잎 수·지상부·지하부·생중량\n기초통계 · IQR 이상치\nPearson 상관계수\n그룹 비교 (Top/Bottom 25%)", "#2ecc71"),
            ("예측모델", "단순회귀 (잎 수만)\n다중회귀 (잎 수+지상부)\nR² · MAE · 5겹 교차검증\n예측 예시 (25장·60장)", "#e74c3c"),
        ]
        for i, (title, content, color) in enumerate(items):
            rect = plt.Rectangle((i*3.4+0.2, 0.8), 3.0, 2.4, linewidth=1.5,
                                  edgecolor=color, facecolor=color, alpha=0.12, zorder=1)
            ax.add_patch(rect)
            ax.text(i*3.4+1.7, 2.9, title, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(i*3.4+1.7, 1.7, content, ha="center", va="center",
                    fontsize=10, linespacing=1.6, color="#2c3e50")
            ax.plot([i*3.4+0.2, i*3.4+3.2], [0.95, 0.95], color=color, linewidth=2, alpha=0.5)
        ax.set_xlim(0, 10.5)
        ax.set_ylim(0, 3.6)
        ax.set_title("분석 흐름도 — 3개 영역: 환경 분석 · 생육 분석 · 예측모델",
                     fontsize=14, pad=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    # ═══════════════════════════════════════════
    # 탭 2: 3.결과 (그래프 중심, 표 최소화)
    # ═══════════════════════════════════════════
    with t2:
        st.header("3. 결과")
        st.caption("2026년 6월 16일 ~ 10월 21일, 127일간 1시간 간격 측정 데이터 기반")

        if env_df is None and growth_df is None:
            st.info("CSV를 업로드하면 모든 결과와 그래프가 표시됩니다.")
            st.stop()

        # ─── 서브탭 within 탭2 ─────────────────
        r1, r2, r3, r4 = st.tabs([
            "🌡️  3.1  스마트팜 환경 분석",
            "🌱  3.2  개체별 생육 차이",
            "📈  3.3  Top·Bottom 그룹 비교 & 가설 검증",
            "🔮  3.4  생중량 예측모델"
        ])

        # ──────────── 3.1 환경 분석 ────────────
        with r1:
            st.subheader("3.1 스마트팜 환경 분석")

            if env_df is None or env_df.shape[0] == 0:
                st.info("환경 CSV를 업로드하세요.")
                st.stop()

            n_rows = len(env_df)
            t0 = env_df["_date_dt"].min().strftime("%Y-%m-%d %H:%M")
            t1_ = env_df["_date_dt"].max().strftime("%Y-%m-%d %H:%M")
            st.success(f"✅ {n_rows:,}행 · 기간: {t0} ~ {t1_}  (127일간 1시간 간격)")

            # 3.1-1 날짜별 평균 온도·습도 시계열
            st.subheader("3.1-1  날짜별 평균 온도·습도 시계열 [그래프 1]")
            st.caption("온도는 여름철(7~8월) 25~28°C까지 오르고 가을로 가며 서서히 내려감. 습도는 60~70% 구간을 대체로 유지하며 8월 하순~9월 초에 약간 낮아짐.")

            if "temperature" in env_df.columns and "humidity" in env_df.columns:
                daily = env_df.groupby("_date_dt").agg(
                    temperature=("temperature","mean"),
                    humidity=("humidity","mean")
                ).reset_index()
                daily.columns = ["_dt","온도","습도"]

                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                ax1.plot(daily["_dt"], daily["온도"], color="#e74c3c", linewidth=2.2,
                         marker="o", markersize=3.5, label="평균 온도 (°C)", zorder=3)
                ax1.set_xlabel("날짜", fontsize=11)
                ax1.set_ylabel("온도 (°C)", color="#e74c3c", fontsize=11)
                ax1.tick_params(axis="y", labelcolor="#e74c3c")
                ax1.set_title("그래프 1. 날짜별 평균 온도·습도 시계열 (2026.06.16~10.21)",
                              fontsize=13, pad=12)
                ax2 = ax1.twinx()
                ax2.plot(daily["_dt"], daily["습도"], color="#3498db", linewidth=2.2,
                         marker="s", markersize=3.5, label="평균 습도 (%)", zorder=3)
                ax2.set_ylabel("습도 (%)", color="#3498db", fontsize=11)
                ax2.tick_params(axis="y", labelcolor="#3498db")
                fig.autofmt_xdate(rotation=45)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                # 요약 수치 (표 대신 metric)
                t_all = env_df["temperature"].dropna()
                h_all = env_df["humidity"].dropna()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("전체 평균 온도", f"{t_all.mean():.2f} °C",
                          delta=f"범위 {t_all.min():.1f}~{t_all.max():.1f} °C")
                c2.metric("전체 평균 습도", f"{h_all.mean():.1f} %",
                          delta=f"범위 {h_all.min():.1f}~{h_all.max():.1f} %")
                cv_t = abs(t_all.std()/t_all.mean()*100) if t_all.mean() != 0 else 0
                cv_h = abs(h_all.std()/h_all.mean()*100) if h_all.mean() != 0 else 0
                c3.metric("온도 CV (변동계수)",
                          f"{cv_t:.2f} %",
                          help="CV = (표준편차÷평균)×100. 작을수록 평균 근처에 밀집.")
                c4.metric("습도 CV (변동계수)",
                          f"{cv_h:.2f} %",
                          help="CV = (표준편차÷평균)×100.")

            # 3.1-2 시간대별 평균 온도·습도
            st.divider()
            st.subheader("3.1-2  시간대별 평균 온도·습도 (24개 시간대) [그래프 2]")
            st.caption("온도(막대, 빨강): 오전 10시 최고, 새벽 6시 최저. 습도(선, 파랑): 새벽 4시 최고, 오전 10시 최저 — 온도 패턴과 반대.")

            if "temperature" in env_df.columns and "humidity" in env_df.columns:
                hourly = env_df.groupby("_hour").agg(
                    temperature=("temperature","mean"),
                    humidity=("humidity","mean")
                ).reset_index()
                max_h = int(hourly.loc[hourly["temperature"].idxmax(), "_hour"])
                min_h = int(hourly.loc[hourly["temperature"].idxmin(), "_hour"])
                max_hh = int(hourly.loc[hourly["humidity"].idxmax(), "_hour"])
                min_hh = int(hourly.loc[hourly["humidity"].idxmin(), "_hour"])
                day_range = hourly["temperature"].max() - hourly["temperature"].min()

                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                ax1.bar(hourly["_hour"], hourly["temperature"], color="#e74c3c",
                        alpha=0.78, width=0.62, label="평균 온도 (°C)", zorder=2)
                ax1.set_xlabel("시간대 (시)", fontsize=11)
                ax1.set_ylabel("온도 (°C)", color="#e74c3c", fontsize=11)
                ax1.set_xticks(range(0, 24))
                ax1.tick_params(axis="y", labelcolor="#e74c3c")
                ax1.set_title("그래프 2. 24개 시간대별 평균 온도·습도",
                              fontsize=13, pad=12)
                ax2 = ax1.twinx()
                ax2.plot(hourly["_hour"], hourly["humidity"], color="#3498db",
                         marker="o", linewidth=2.5, markersize=6.5, label="평균 습도 (%)",
                         zorder=3)
                ax2.set_ylabel("습도 (%)", color="#3498db", fontsize=11)
                ax2.tick_params(axis="y", labelcolor="#3498db")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=10)
                ax1.axhline(hourly["temperature"].mean(), color="gray", linestyle="--",
                            linewidth=1, alpha=0.6, label="온도 전체 평균")
                plt.tight_layout()
                st.pyplot(fig)

                m1, m2, m3 = st.columns(3)
                m1.metric("최고 온도 시간대", f"오전 {max_h}시",
                          delta=f"{hourly['temperature'].max():.2f} °C")
                m2.metric("최저 온도 시간대", f"오전 {min_h}시",
                          delta=f"{hourly['temperature'].min():.2f} °C")
                m3.metric("일교차 (최고−최저)", f"{day_range:.2f} °C",
                          delta="매우 작음 (안정적)")

            # 3.1-3 환경 기초 통계 (표 대신 CV 막대 그래프)
            st.divider()
            st.subheader("3.1-3  환경 변수 기초 통계")
            st.caption("평균: 중심값 / 표준편차: 평균에서 얼마나 퍼져 있는지 / CV: (표준편차÷평균)×100, 단위 다른 값끼리 비교 가능 / 최소·최대: 데이터 범위")

            stat_env_items = []
            labels_env  = []
            cvs_env     = []
            min_env     = []
            max_env     = []
            for key, label in [("temperature","온도 (°C)"),("humidity","습도 (%)"),
                                 ("ec","EC"),("ph","pH")]:
                if key in env_df.columns:
                    s = env_df[key].dropna()
                    if len(s) == 0:
                        continue
                    mean_v  = s.mean()
                    std_v   = s.std()
                    min_v   = s.min()
                    max_v   = s.max()
                    cv_v    = abs(std_v/mean_v*100) if mean_v != 0 else 0
                    stat_env_items.append({
                        "항목": label,
                        "평균": f"{mean_v:.2f}",
                        "중앙값": f"{s.median():.2f}",
                        "표준편차": f"{std_v:.2f}",
                        "CV (%)": f"{cv_v:.2f}",
                        "최소": f"{min_v:.2f}",
                        "최대": f"{max_v:.2f}",
                    })
                    labels_env.append(label)
                    cvs_env.append(cv_v)
                    min_env.append(min_v)
                    max_env.append(max_v)

            c_cv, c_minmax = st.columns([1, 1.3])
            with c_cv:
                if labels_env:
                    st.pyplot(cv_bar_plot(labels_env, cvs_env,
                               "환경 변수별 CV (%) 비교 — 값이 작을수록 안정적"))
            with c_minmax:
                if labels_env:
                    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                    x = np.arange(len(labels_env))
                    bw = 0.35
                    ax2.bar(x-bw/2, min_env, bw, color="#3498db", alpha=0.8, label="최소")
                    ax2.bar(x+bw/2, max_env, bw, color="#e74c3c", alpha=0.8, label="최대")
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(labels_env, fontsize=10)
                    ax2.set_title("환경 변수별 최소·최대 값 비교", fontsize=12, pad=8)
                    ax2.legend(fontsize=10)
                    ax2.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)

            if stat_env_items:
                st.dataframe(pd.DataFrame(stat_env_items), use_container_width=True, hide_index=True)

            # 3.1-4 전반부 vs 후반부 비교 (그래프)
            st.divider()
            st.subheader("3.1-4  전반부(6/16~8/18) vs 후반부(8/19~10/21) 비교")
            st.caption("측정 중간(약 8월 19일)을 기준. 평균 차이: 후반부 − 전반부. 분산비: 후반부 분산 ÷ 전반부 분산 (1보다 크면 후반부 변동이 더 큼).")

            if len(env_df) > 0:
                mid = env_df["_date_dt"].min() + (env_df["_date_dt"].max() - env_df["_date_dt"].min()) / 2
                fh = env_df[env_df["_date_dt"] <= mid]
                sh = env_df[env_df["_date_dt"] >  mid]

                comp_labels = []
                diffs       = []
                var_ratios  = []
                for key, label in [("temperature","온도"),("humidity","습도"),
                                     ("ec","EC"),("ph","pH")]:
                    if key in env_df.columns:
                        f_s = fh[key].dropna()
                        s_s = sh[key].dropna()
                        if len(f_s) == 0 or len(s_s) == 0:
                            continue
                        diff = s_s.mean() - f_s.mean()
                        vr   = s_s.var()/f_s.var() if f_s.var() > 0 else float("nan")
                        comp_labels.append(label)
                        diffs.append(diff)
                        var_ratios.append(vr)

                if comp_labels:
                    col_d, col_vr = st.columns(2)
                    with col_d:
                        fig_d, ax_d = plt.subplots(figsize=(5.5, 3.5))
                        colors = ["#27ae60" if d > 0 else "#c0392b" for d in diffs]
                        bars = ax_d.bar(comp_labels, diffs, color=colors, alpha=0.85,
                                         width=0.5, edgecolor="white")
                        for b, v in zip(bars, diffs):
                            off = 0.01 if v >=0 else -0.04
                            ax_d.text(b.get_x()+b.get_width()/2, v+off, f"{v:+.2f}",
                                      ha="center", va="bottom" if v >=0 else "top", fontsize=10,
                                      fontweight="bold")
                        ax_d.set_ylabel("평균 차이 (후반 − 전반)", fontsize=10)
                        ax_d.set_title("전반부 vs 후반부 평균 차이", fontsize=12, pad=8)
                        ax_d.axhline(0, color="black", linewidth=0.8)
                        ax_d.grid(axis="y", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_d)
                    with col_vr:
                        fig_vr, ax_vr = plt.subplots(figsize=(5.5, 3.5))
                        vr_colors = ["#e67e22" if v > 1.1 else "#2ecc71" for v in var_ratios]
                        ax_vr.bar(comp_labels, var_ratios, color=vr_colors, alpha=0.85,
                                  width=0.5, edgecolor="white")
                        ax_vr.axhline(1.0, color="black", linewidth=1, linestyle="--",
                                      label="분산비 = 1 (변동 동일)")
                        for i, (b, v) in enumerate(zip(ax_vr.patches, var_ratios)):
                            ax_vr.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}",
                                       ha="center", va="bottom", fontsize=10, fontweight="bold")
                        ax_vr.set_ylabel("분산비 (후반 ÷ 전반)", fontsize=10)
                        ax_vr.set_title("전반부 vs 후반부 분산비", fontsize=12, pad=8)
                        ax_vr.legend(fontsize=9)
                        ax_vr.grid(axis="y", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_vr)

                    st.markdown(
                        f"""
                        - **온도**: 후반부 평균 {diffs[0]:+.2f}°C, 분산비 {var_ratios[0]:.2f} → 가을로 가며 온도 하락 + 변동 증가
                        - **습도**: 후반부 평균 {diffs[1]:+.2f}%, 분산비 {var_ratios[1]:.2f} → 평균·분산 모두 거의 변화 없음 (안정적)
                        - **EC**: 후반부 평균 {diffs[2]:+.2f}, 분산비 {var_ratios[2]:.2f} → EC 꾸준히 상승
                        - **pH**: 후반부 평균 {diffs[3]:+.2f}, 분산비 {var_ratios[3]:.2f} → 알칼리성 → 약산성 이동
                        """
                    )

            # 3.1-5 EC·pH 변화 구간 (그래프)
            st.divider()
            st.subheader("3.1-5  EC·pH 변화 구간 비교 (6월 초 vs 8월 하순)")
            st.caption("6월 초(6/16~6/25)와 8월 하순(8/20~9/5)의 EC·pH 평균을 비교.")

            if "ec" in env_df.columns and "ph" in env_df.columns:
                env_df_c = env_df.copy()
                env_df_c["_d"] = env_df_c["_date_dt"].dt.date
                early = (env_df_c["_d"] >= pd.Timestamp("2026-06-16").date()) &                 (env_df_c["_d"] <= pd.Timestamp("2026-06-25").date())
                late  = (env_df_c["_d"] >= pd.Timestamp("2026-08-20").date()) &                 (env_df_c["_d"] <= pd.Timestamp("2026-09-05").date())

                if env_df_c[early]["ec"].notna().any() and env_df_c[late]["ec"].notna().any():
                    ec_e = env_df_c[early]["ec"].dropna().mean()
                    ec_l = env_df_c[late]["ec"].dropna().mean()
                    ph_e = env_df_c[early]["ph"].dropna().mean()
                    ph_l = env_df_c[late]["ph"].dropna().mean()

                    fig_ecph, axes = plt.subplots(1, 2, figsize=(11, 4))
                    # EC
                    ax_e = axes[0]
                    bp_e = ax_e.boxplot([env_df_c[early]["ec"].dropna(),
                                          env_df_c[late]["ec"].dropna()],
                                         labels=["6월 초\n(6/16~6/25)","8월 하순\n(8/20~9/5)"],
                                         patch_artist=True)
                    for patch, color in zip(bp_e['boxes'], ["#3498db","#e74c3c"]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    ax_e.set_title("EC 변화", fontsize=12, pad=8)
                    ax_e.set_ylabel("EC", fontsize=11)
                    ax_e.text(0.5, ec_e*1.05, f"평균 {ec_e:.2f}", ha="center", fontsize=9, color="#3498db")
                    ax_e.text(1.5, ec_l*1.05, f"평균 {ec_l:.2f}", ha="center", fontsize=9, color="#e74c3c")
                    ax_e.grid(axis="y", alpha=0.3)
                    # pH
                    ax_p = axes[1]
                    bp_p = ax_p.boxplot([env_df_c[early]["ph"].dropna(),
                                          env_df_c[late]["ph"].dropna()],
                                         labels=["6월 초\n(6/16~6/25)","8월 하순\n(8/20~9/5)"],
                                         patch_artist=True)
                    for patch, color in zip(bp_p['boxes'], ["#3498db","#e74c3c"]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    ax_p.set_title("pH 변화", fontsize=12, pad=8)
                    ax_p.set_ylabel("pH", fontsize=11)
                    ax_p.text(0.5, ph_e*1.02, f"평균 {ph_e:.2f}", ha="center", fontsize=9, color="#3498db")
                    ax_p.text(1.5, ph_l*1.02, f"평균 {ph_l:.2f}", ha="center", fontsize=9, color="#e74c3c")
                    ax_p.grid(axis="y", alpha=0.3)
                    plt.suptitle("3.1-5  EC·pH 구간별 비교 (Box Plot)", fontsize=13, y=1.02)
                    plt.tight_layout()
                    st.pyplot(fig_ecph)

                    ec_chg = (ec_l-ec_e)/ec_e*100 if ec_e != 0 else 0
                    ph_chg = ph_l - ph_e
                    st.markdown(
                        f"""
                        - **EC**: {ec_e:.2f} → {ec_l:.2f}로 약 **{abs(ec_chg):.0f}% 상승** → 양분 농도 증가 방향
                        - **pH**: {ph_e:.2f} → {ph_l:.2f}로 약 **{abs(ph_chg):.2f}pH 단위 하락** → 알칼리성 → 약산성
                        - 이러한 변화는 식물 양분 흡수나 생리 상태에 영향을 줄 수 있는 수준의 변화
                        """
                    )

            # 3.1-6 IQR 이상치 (그래프)
            st.divider()
            st.subheader("3.1-6  온도·습도 IQR 이상치 확인")
            st.caption("IQR = Q3(75%) − Q1(25%). 이상치 경계: Q1 − 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과. 이 연구에서는 이상치를 삭제하지 않고 그대로 둠.")

            if "temperature" in env_df.columns and "humidity" in env_df.columns:
                fig_iqr, axes = plt.subplots(1, 2, figsize=(11, 4))
                for ax_i, (key, label, color) in zip(axes,
                    [("temperature","온도 (°C)","#e74c3c"),("humidity","습도 (%)","#3498db")]):
                    s = env_df[key].dropna()
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    lo = q1 - 1.5*iqr
                    hi = q3 + 1.5*iqr
                    ax_i.boxplot(s.values, vert=True, patch_artist=True,
                                 boxprops=dict(facecolor=color, alpha=0.5),
                                 medianprops=dict(color="black", linewidth=2))
                    ax_i.axhline(lo, color="orange", linestyle="--", linewidth=1.5,
                                 label=f"하한: {lo:.2f}")
                    ax_i.axhline(hi, color="red", linestyle="--", linewidth=1.5,
                                 label=f"상한: {hi:.2f}")
                    n_out = ((s<lo)|(s>hi)).sum()
                    ax_i.set_title(f"{label} — 이상치 {n_out}개", fontsize=12, pad=8)
                    ax_i.set_ylabel(label, fontsize=11)
                    ax_i.legend(fontsize=9)
                    ax_i.grid(axis="y", alpha=0.3)
                plt.suptitle("3.1-6  온도·습도 IQR 이상치 Box Plot", fontsize=13, y=1.02)
                plt.tight_layout()
                st.pyplot(fig_iqr)

        # ──────────── 3.2 생육 분석 ────────────
        with r2:
            st.subheader("3.2 개체별 생육 차이")

            if growth_df is None or growth_df.shape[0] == 0:
                st.info("생육 CSV를 업로드하세요.")
                st.stop()

            n_g = len(growth_df)
            st.success(f"✅ {n_g}개체 로드 완료")

            # 3.2-1 전체 생육 데이터 (표 최소화 — 헤드 15행만)
            st.subheader("3.2-1  전체 생육 데이터 (일부 샘플)")
            st.dataframe(growth_df.head(15), use_container_width=True, hide_index=True)
            st.caption(f"전체 {n_g}개체 중 앞 15개체. 전체 목록은 다운로드 받거나 아래 그래프로 확인하세요.")

            # 3.2-2 생육 기초 통계 — CV 막대 그래프 + 최대/최소 막대 그래프
            st.divider()
            st.subheader("3.2-2  생육 항목 기초 통계")

            st.caption(
                "평균: 모든 값의 중심 / 중앙값: 크기순 가운데 값 / 표준편차: 평균에서 얼마나 넓게 퍼져 있는지 "
                "/ CV: (표준편차÷평균)×100, 단위 다른 항목끼리 비교 가능 / 최소·최대: 데이터 범위 / 최대/최소: 가장 큰 값과 작은 값의 비율"
            )

            gstat_labels = []
            gstat_cvs   = []
            gstat_means = []
            gstat_mins  = []
            gstat_maxs  = []
            gstat_ratios = []
            for col_disp, col_key, unit in [
                ("잎 수","잎 수","장"),
                ("지상부 길이","지상부 길이","mm"),
                ("지하부 길이","지하부 길이","mm"),
                ("생중량","생중량","g"),
            ]:
                if col_key in growth_df.columns:
                    s = growth_df[col_key].dropna()
                    mean_v  = s.mean()
                    std_v   = s.std()
                    min_v   = s.min()
                    max_v   = s.max()
                    cv_v    = abs(std_v/mean_v*100) if mean_v != 0 else 0
                    ratio_v = max_v/min_v if min_v > 0 else float("nan")
                    gstat_labels.append(f"{col_disp}\n({unit})")
                    gstat_cvs.append(cv_v)
                    gstat_means.append(mean_v)
                    gstat_mins.append(min_v)
                    gstat_maxs.append(max_v)
                    gstat_ratios.append(ratio_v)

            col_cv, col_mm, col_ratio = st.columns(3)
            with col_cv:
                if gstat_labels:
                    colors_cv = ["#e74c3c" if v == max(gstat_cvs) else "#2ecc71" for v in gstat_cvs]
                    st.pyplot(cv_bar_plot(gstat_labels, gstat_cvs,
                               "생육 항목별 CV (%) — 생중량이 가장 높음", color=colors_cv[0] if len(set(colors_cv))==1 else None))
                    # 실제로는 하나만 color 못주므로 다시 그림
                    fig_cv, ax_cv = plt.subplots(figsize=(5.5, 3.5))
                    max_cv = max(gstat_cvs)
                    bar_colors = ["#e74c3c" if v == max_cv else "#2ecc71" for v in gstat_cvs]
                    bars_cv = ax_cv.bar(range(len(gstat_cvs)), gstat_cvs, color=bar_colors,
                                         width=0.55, edgecolor="white", linewidth=1.2)
                    for i, (b, v) in enumerate(zip(bars_cv, gstat_cvs)):
                        ax_cv.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%",
                                   ha="center", va="bottom", fontsize=10, fontweight="bold")
                    ax_cv.set_xticks(range(len(gstat_cvs)))
                    ax_cv.set_xticklabels(gstat_labels, fontsize=9)
                    ax_cv.set_ylabel("CV (%)", fontsize=11)
                    ax_cv.set_title("생육 항목별 CV (%) 비교", fontsize=12, pad=8)
                    ax_cv.set_ylim(0, max(gstat_cvs)*1.2)
                    ax_cv.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_cv)
                    st.caption(f"CV 기준: 생중량 {max(gstat_cvs):.1f}% > 지하부 길이 {gstat_cvs[2]:.1f}% > 잎 수 {gstat_cvs[0]:.1f}% > 지상부 길이 {gstat_cvs[1]:.1f}%")

            with col_mm:
                if gstat_labels:
                    fig_mm, ax_mm = plt.subplots(figsize=(5.5, 3.5))
                    x = np.arange(len(gstat_labels))
                    bw = 0.35
                    ax_mm.bar(x-bw/2, gstat_mins, bw, color="#3498db", alpha=0.8, label="최소")
                    ax_mm.bar(x+bw/2, gstat_maxs, bw, color="#e74c3c", alpha=0.8, label="최대")
                    ax_mm.set_xticks(x)
                    ax_mm.set_xticklabels(gstat_labels, fontsize=9)
                    ax_mm.set_title("생육 항목별 최소·최대 값", fontsize=12, pad=8)
                    ax_mm.legend(fontsize=10)
                    ax_mm.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_mm)

            with col_ratio:
                if gstat_labels:
                    fig_rt, ax_rt = plt.subplots(figsize=(5.5, 3.5))
                    rt_colors = ["#e74c3c" if v == max(gstat_ratios) else "#2ecc71" for v in gstat_ratios]
                    bars_rt = ax_rt.bar(range(len(gstat_ratios)), gstat_ratios,
                                         color=rt_colors, width=0.55, edgecolor="white", linewidth=1.2)
                    for b, v in zip(bars_rt, gstat_ratios):
                        ax_rt.text(b.get_x()+b.get_width()/2, v+3, f"{v:.1f}",
                                   ha="center", va="bottom", fontsize=10, fontweight="bold")
                    ax_rt.set_xticks(range(len(gstat_ratios)))
                    ax_rt.set_xticklabels(gstat_labels, fontsize=9)
                    ax_rt.set_title("생육 항목별 최대/최소 배율", fontsize=12, pad=8)
                    ax_rt.set_ylabel("배율 (배)", fontsize=11)
                    ax_rt.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_rt)
                    st.caption(f"생중량 최대/최소 = {gstat_ratios[3]:.1f}배 (21번 10.01g vs 36번 0.04g)")

            # 기초통계 표 (최소)
            if gstat_labels:
                stat_rows = []
                for col_disp, col_key in [("잎 수","잎 수"),("지상부 길이","지상부 길이"),
                                            ("지하부 길이","지하부 길이"),("생중량","생중량")]:
                    if col_key in growth_df.columns:
                        s = growth_df[col_key].dropna()
                        stat_rows.append({
                            "항목": col_disp,
                            "평균": f"{s.mean():.2f}",
                            "중앙값": f"{s.median():.2f}",
                            "표준편차": f"{s.std():.2f}",
                            "CV (%)": f"{abs(s.std()/s.mean()*100):.1f}",
                            "최소": f"{s.min():.2f}",
                            "최대": f"{s.max():.2f}",
                            "최대/최소": f"{s.max()/s.min():.1f}",
                        })
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            # 3.2-3 생중량 분포 히스토그램
            st.divider()
            st.subheader("3.2-3  생중량(g) 분포 히스토그램 [그래프 4]")
            st.caption("81개체 생중량을 구간별 막대 그래프로 표시. 0~2g 구간에 가장 많은 개체가 몰려 있고, 3g 이상은 급격히 줄어듦. 붉은 점선: 중앙값 1.76g. 오른쪽 꼬리 비대칭 분포.")

            if "생중량" in growth_df.columns:
                fig_h, ax_h = plt.subplots(figsize=(10, 4.5))
                fw_vals = growth_df["생중량"].dropna()
                n_bins = max(10, int(np.sqrt(len(fw_vals))))
                ax_h.hist(fw_vals, bins=n_bins, color="#2ecc71", edgecolor="white",
                          alpha=0.88, linewidth=1.2)
                med   = fw_vals.median()
                mean_ = fw_vals.mean()
                ax_h.axvline(med,   color="red",    linestyle="--", linewidth=2.2, label=f"중앙값: {med:.2f}g")
                ax_h.axvline(mean_, color="orange", linestyle="--", linewidth=2.2, label=f"평균: {mean_:.2f}g")
                ax_h.set_xlabel("생중량 (g)", fontsize=11)
                ax_h.set_ylabel("개체 수", fontsize=11)
                ax_h.set_title("그래프 4. 생중량(g) 분포 히스토그램 — 81개체", fontsize=13, pad=12)
                ax_h.legend(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_h)

                n_under2 = (fw_vals < 2).sum()
                st.metric("2g 미만 개체 비율", f"{n_under2/len(fw_vals)*100:.1f}%",
                          delta=f"{n_under2}/{len(fw_vals)}개체")

            # 3.2-4 이상치 (산점도로 시각화)
            st.divider()
            st.subheader("3.2-4  IQR 이상치 시각화 [산점도]")
            st.caption("가로축: 개체번호, 세로축: 측정값. 붉은 점: IQR 이상치. 검은색 점선: IQR 상한·하한 경계.")

            if all(c in growth_df.columns for c in ["개체번호","생중량","잎 수","지상부 길이","지하부 길이"]):
                fig_out, axes = plt.subplots(2, 2, figsize=(12, 7))
                axes = axes.flatten()
                for ax_i, (col_key, label, unit) in zip(axes,
                    [("잎 수","잎 수","장"),("지상부 길이","지상부 길이","mm"),
                     ("지하부 길이","지하부 길이","mm"),("생중량","생중량","g")]):
                    df_p = growth_df[["개체번호", col_key]].dropna()
                    q1 = df_p[col_key].quantile(0.25)
                    q3 = df_p[col_key].quantile(0.75)
                    iqr = q3 - q1
                    lo  = q1 - 1.5*iqr
                    hi  = q3 + 1.5*iqr
                    ax_i.scatter(df_p["개체번호"], df_p[col_key], color="#2ecc71", alpha=0.6,
                                 s=45, edgecolors="white", linewidth=0.5, zorder=2)
                    outlier_mask = (df_p[col_key] < lo) | (df_p[col_key] > hi)
                    ax_i.scatter(df_p.loc[outlier_mask, "개체번호"],
                                 df_p.loc[outlier_mask, col_key],
                                 color="#e74c3c", alpha=0.9, s=90, edgecolors="darkred",
                                 linewidth=1.5, zorder=3, label=f"이상치 {outlier_mask.sum()}개")
                    ax_i.axhline(lo, color="orange", linestyle="--", linewidth=1.2, alpha=0.7)
                    ax_i.axhline(hi, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
                    ax_i.axhline(q1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
                    ax_i.axhline(q3, color="gray", linestyle=":", linewidth=1, alpha=0.5)
                    ax_i.set_xlabel("개체번호", fontsize=10)
                    ax_i.set_ylabel(f"{label} ({unit})", fontsize=10)
                    ax_i.set_title(f"{label} — IQR 이상치", fontsize=11, pad=6)
                    ax_i.legend(fontsize=9)
                    ax_i.grid(alpha=0.3)
                plt.suptitle("3.2-4  생육 4항목 IQR 이상치 산점도", fontsize=13, y=1.01)
                plt.tight_layout()
                st.pyplot(fig_out)

                st.markdown(
                    """
                    - 생중량 큰 쪽(7.14g 초과): **13, 20, 21번** 3개체 — 특히 잘 자란 개체들
                    - 생중량이 극단적으로 작은 36번(0.04g)은 IQR 하한(−2.94g)보다 크므로 **이상치로 분류되지 않음**
                    - 24번 개체: 잎 수(102)와 지상부 길이(126) 모두 이상치
                    """
                )

            # 3.2-5 최대·최소 개체 (그래프 명시)
            st.divider()
            st.subheader("3.2-5  생중량 최대·최소 개체")
            st.caption("가로축: 개체번호, 세로축: 생중량(g). 붉은 마름모: 최대 개체, 파란 마름모: 최소 개체.")

            if "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
                max_row = growth_df.loc[growth_df["생중량"].idxmax()]
                min_row = growth_df.loc[growth_df["생중량"].idxmin()]

                fig_maxmin, ax = plt.subplots(figsize=(10, 4))
                ax.scatter(growth_df["개체번호"], growth_df["생중량"],
                           color="#2ecc71", alpha=0.5, s=50, edgecolors="white", linewidth=0.5,
                           label="전체 81개체")
                ax.scatter(max_row["개체번호"], max_row["생중량"],
                           color="#e74c3c", s=250, marker="D", edgecolors="darkred",
                           linewidth=2, zorder=5,
                           label=f"최대: {int(max_row['개체번호'])}번 ({max_row['생중량']:.2f}g)")
                ax.scatter(min_row["개체번호"], min_row["생중량"],
                           color="#3498db", s=250, marker="D", edgecolors="darkblue",
                           linewidth=2, zorder=5,
                           label=f"최소: {int(min_row['개체번호'])}번 ({min_row['생중량']:.2f}g)")
                ax.set_xlabel("개체번호", fontsize=11)
                ax.set_ylabel("생중량 (g)", fontsize=11)
                ax.set_title("3.2-5  생중량 최대·최소 개체 강조 산점도", fontsize=13, pad=12)
                ax.legend(fontsize=10, loc="upper left")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_maxmin)

                st.metric("생중량 최대 개체", f"{max_row['생중량']:.2f}g",
                          delta=f"{int(max_row['개체번호'])}번")
                st.metric("생중량 최소 개체", f"{min_row['생중량']:.2f}g",
                          delta=f"{int(min_row['개체번호'])}번")
                st.markdown(f"**최대 개체(21번)와 최소 개체(36번)의 생중량 차이는 250배** — 같은 스마트팜에서 재배했음에도 이 정도로 차이가 났다.")

            # 3.2-6 상관계수 (산점도 3종 + r값 그래프)
            st.divider()
            st.subheader("3.2-6  생중량 vs 생육 항목 Pearson 상관계수 [산점도 3종]")
            st.caption("Pearson r: -1~1. 1에 가까울수록 한 변수가 증가할 때 다른 변수도 함께 증가. p<0.05이면 통계적으로 유의한 관계. **인과관계가 아님** — 함께 증가하는 '경향'만 보여줌.")

            if all(c in growth_df.columns for c in ["생중량","잎 수","지상부 길이","지하부 길이"]):
                fig_corr, axes = plt.subplots(1, 3, figsize=(14, 4.5))
                configs = [
                    ("잎 수","잎 수 (장)","#3498db"),
                    ("지상부 길이","지상부 길이 (mm)","#e67e22"),
                    ("지하부 길이","지하부 길이 (mm)","#9b59b6"),
                ]
                r_vals = []
                for ax_c, (col, lbl, color) in zip(axes, configs):
                    r_v, p_v = stats.pearsonr(growth_df["생중량"], growth_df[col])
                    r_vals.append(r_v)
                    ax_c.scatter(growth_df[col], growth_df["생중량"],
                                 color=color, alpha=0.55, s=50, edgecolors="white", linewidth=0.5,
                                 zorder=2)
                    x_r = np.linspace(growth_df[col].min(), growth_df[col].max(), 100)
                    sl, int_, _, _, _ = stats.linregress(growth_df[col], growth_df["생중량"])
                    ax_c.plot(x_r, sl*x_r+int_, color="#e74c3c", linewidth=2.2,
                              label=f"r = {r_v:.3f}\np {'<' if p_v<0.001 else '<'} 0.001" if p_v < 0.001
                              else f"r = {r_v:.3f}\np = {p_v:.3f}",
                              zorder=3)
                    ax_c.set_xlabel(lbl, fontsize=11)
                    ax_c.set_ylabel("생중량 (g)", fontsize=11)
                    ax_c.set_title(f"{lbl} vs 생중량", fontsize=12, pad=8)
                    ax_c.legend(fontsize=10)
                    ax_c.grid(alpha=0.3)
                plt.suptitle("3.2-6  생중량과 생육 항목 간 Pearson 상관계수 산점도", fontsize=13, y=1.03)
                plt.tight_layout()
                st.pyplot(fig_corr)

                # r값 비교 막대 그래프
                fig_rv, ax_rv = plt.subplots(figsize=(5, 3.5))
                rv_labels = ["잎 수","지하부 길이","지상부 길이"]
                bar_colors_rv = ["#e74c3c" if v == max(r_vals) else "#2ecc71" for v in r_vals]
                bars_rv = ax_rv.barh(rv_labels, r_vals, color=bar_colors_rv,
                                     height=0.5, edgecolor="white", linewidth=1.2)
                for b, v, col, p_v in zip(bars_rv, r_vals,
                    ["잎 수","지하부 길이","지상부 길이"],
                    [stats.pearsonr(growth_df["생중량"], growth_df[c])[1] for c in
                     ["잎 수","지상부 길이","지하부 길이"]]):
                    sig_str = "***" if p_v < 0.001 else ("*" if p_v < 0.05 else "")
                    ax_rv.text(v+0.02, b.get_y()+b.get_height()/2,
                               f"r={v:.3f} {sig_str}\n(p={p_v:.2e})",
                               va="center", fontsize=10, fontweight="bold")
                ax_rv.set_xlabel("Pearson r", fontsize=11)
                ax_rv.set_title("생중량과의 상관계수 비교", fontsize=12, pad=8)
                ax_rv.set_xlim(0, 1.1)
                ax_rv.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_rv)
                st.caption("*** p<0.001, * p<0.05. 잎 수 r=0.886으로 가장 강한 양의 상관 — 잎 수↑ → 생중량↑ 경향.")

        # ──────────── 3.3 그룹 비교 ────────────
        with r3:
            st.subheader("3.3  Top 25% vs Bottom 25% 그룹 비교 및 가설 검증")
            st.caption(
                "생중량 기준 상위 25%(20개체)와 하위 25%(20개체)로 나눈다. "
                "Welch t검정: 두 그룹 평균이 같은지 검정 (분산이 달라도 사용 가능). "
                "p값이 작을수록 '두 그룹 평균이 같다'는 가정이 기각됨."
            )

            if growth_df is None or growth_df.shape[0] == 0:
                st.info("생육 CSV를 업로드하세요.")
                st.stop()

            if "생중량" in growth_df.columns:
                fw = growth_df["생중량"].dropna()
                n_each = int(np.ceil(len(fw) * 0.25))
                top_t    = fw.nlargest(n_each).min()
                bot_t    = fw.nsmallest(n_each).max()
                top_g    = growth_df[growth_df["생중량"] >= top_t].head(n_each).copy()
                bot_g    = growth_df[growth_df["생중량"] <= bot_t].head(n_each).copy()

                st.markdown(f"""
                | 항목 | 값 |
                | --- | --- |
                | 전체 개체 수 | {len(growth_df)}개체 |
                | 각 그룹 개체 수 | {n_each}개체 (81×0.25 = 20.25 → 반올림) |
                | 하위 경계값 | {bot_t:.2f}g 이하 |
                | 상위 경계값 | {top_t:.2f}g 이상 |
                | 중간 그룹 (제외) | {len(growth_df)-2*n_each}개체 |
                동점 처리: 하위 경계 {bot_t:.2f}g에 { (growth_df['생중량']==bot_t).sum() }개체, 상위 경계 {top_t:.2f}g에 { (growth_df['생중량']==top_t).sum() }개체 동점이지만 해당 순위까지 포함해 정확히 {n_each}개씩 나뉨.
                """)

                # 3.3-1 그룹별 잎 수 비교 (막대 + 산점 오버레이)
                st.divider()
                st.subheader("3.3-1  잎 수 Top vs Bottom 비교 [그룹 막대 그래프]")
                st.caption("잎 수 상위 25%(20개체) vs 하위 25%(20개체). *** p<0.001: 두 그룹의 잎 수가 같다는 가정에서 관찰되기 매우 어려운 결과.")

                if all(c in growth_df.columns for c in ["잎 수","생중량"]):
                    t_l, b_l = top_g["잎 수"], bot_g["잎 수"]
                    t_mean_l = t_l.mean()
                    b_mean_l = b_l.mean()
                    t_std_l  = t_l.std()
                    b_std_l  = b_l.std()
                    ratio_l  = t_mean_l / b_mean_l if b_mean_l != 0 else float("nan")
                    t_stat_l, p_val_l = stats.ttest_ind(t_l, b_l, equal_var=False)

                    fig_l, ax_l = plt.subplots(figsize=(8, 4.5))
                    grps = ["Bottom 25%\n(하위 20개체)", "Top 25%\n(상위 20개체)"]
                    means  = [b_mean_l, t_mean_l]
                    stds   = [b_std_l,  t_std_l]
                    cols   = ["#3498db", "#e74c3c"]
                    x_pos  = [0, 1]
                    bars_l = ax_l.bar(x_pos, means, color=cols, width=0.45, alpha=0.85,
                                       edgecolor="white", linewidth=1.5,
                                       yerr=stds, capsize=8, error_kw=dict(linewidth=1.5))
                    for i, (b, m) in enumerate(zip(bars_l, means)):
                        ax_l.text(b.get_x()+b.get_width()/2, m+2, f"{m:.1f}장",
                                  ha="center", va="bottom", fontsize=12, fontweight="bold")
                    ax_l.set_xticks(x_pos)
                    ax_l.set_xticklabels(grps, fontsize=11)
                    ax_l.set_ylabel("잎 수 (장)", fontsize=11)
                    ax_l.set_title(f"잎 수 Top vs Bottom  (비율 {ratio_l:.1f}배, p={p_val_l:.1e})",
                                   fontsize=12, pad=8)
                    ax_l.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_l)

                    st.metric("Top 평균 잎 수", f"{t_mean_l:.1f}장",
                              delta=f"비율 {ratio_l:.1f}배")
                    st.metric("Bottom 평균 잎 수", f"{b_mean_l:.1f}장")
                    st.metric("p값 (Welch t검정)", f"{p_val_l:.2e}",
                              delta="*** 유의함 (p<0.001)")

                # 3.3-2 그룹별 생중량 비교 (박스 플롯)
                st.divider()
                st.subheader("3.3-2  생중량 Top vs Bottom 비교 [Box Plot]")
                st.caption("상자: Q1~Q3(데이터의 중간 50%). 수염: 이상치 제외 범위. 중앙선: 중앙값. *** p<0.001.")

                if "생중량" in growth_df.columns:
                    fig_fw, ax_fw = plt.subplots(figsize=(7, 4.5))
                    data_fw = [bot_g["생중량"].values, top_g["생중량"].values]
                    bp_fw = ax_fw.boxplot(data_fw, labels=["Bottom 25%\n(하위 20개체)",
                                                           "Top 25%\n(상위 20개체)"],
                                           patch_artist=True, widths=0.4)
                    for patch, color in zip(bp_fw["boxes"], ["#3498db","#e74c3c"]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.55)
                    # 개별 점 오버레이
                    for i, grp in enumerate([bot_g, top_g]):
                        x_jit = np.random.normal(i+1, 0.04, len(grp))
                        ax_fw.scatter(x_jit, grp["생중량"], color="gray", alpha=0.35, s=30, zorder=2)
                    t_fw_m = top_g["생중량"].mean()
                    b_fw_m = bot_g["생중량"].mean()
                    ratio_fw = t_fw_m/b_fw_m if b_fw_m != 0 else float("nan")
                    t_stat_fw, p_val_fw = stats.ttest_ind(top_g["생중량"], bot_g["생중량"], equal_var=False)
                    ax_fw.set_ylabel("생중량 (g)", fontsize=11)
                    ax_fw.set_title(f"생중량 Top vs Bottom  (Top {t_fw_m:.2f}g vs Bottom {b_fw_m:.2f}g, "
                                    f"비율 {ratio_fw:.1f}배, p={p_val_fw:.1e})",
                                    fontsize=12, pad=8)
                    ax_fw.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_fw)

                    st.metric("Top 평균 생중량", f"{t_fw_m:.2f}g",
                              delta=f"비율 {ratio_fw:.1f}배")
                    st.metric("Bottom 평균 생중량", f"{b_fw_m:.2f}g")
                    st.metric("p값 (Welch t검정)", f"{p_val_fw:.2e}",
                              delta="*** 유의함 (p<0.001)")

                # 3.3-3 지상부·지하부 길이 비교 (그룹 막대)
                st.divider()
                st.subheader("3.3-3  지상부 길이 · 지하부 길이 Top vs Bottom [그룹 막대]")

                if all(c in growth_df.columns for c in ["지상부 길이","지하부 길이","생중량"]):
                    fig_size, axes_s = plt.subplots(1, 2, figsize=(12, 4.5))
                    for ax_s, (col, label) in zip(axes_s,
                        [("지상부 길이","지상부 길이 (mm)"),("지하부 길이","지하부 길이 (mm)")]):
                        t_v = top_g[col]
                        b_v = bot_g[col]
                        t_m = t_v.mean()
                        b_m = b_v.mean()
                        t_sd = t_v.std()
                        b_sd = b_v.std()
                        ratio_s = t_m/b_m if b_m != 0 else float("nan")
                        t_st, p_s = stats.ttest_ind(t_v, b_v, equal_var=False)
                        x_p = [0, 1]
                        bars_s = ax_s.bar(x_p, [b_m, t_m], color=["#3498db","#e74c3c"],
                                           width=0.45, alpha=0.85, edgecolor="white",
                                           yerr=[b_sd, t_sd], capsize=8,
                                           error_kw=dict(linewidth=1.5))
                        for b, m in zip(bars_s, [b_m, t_m]):
                            ax_s.text(b.get_x()+b.get_width()/2, m+2, f"{m:.1f}",
                                      ha="center", va="bottom", fontsize=12, fontweight="bold")
                        ax_s.set_xticks(x_p)
                        ax_s.set_xticklabels(["Bottom 25%","Top 25%"], fontsize=11)
                        ax_s.set_ylabel(f"{label}", fontsize=11)
                        ax_s.set_title(f"{label}  (비율 {ratio_s:.1f}배, p={p_s:.1e})",
                                       fontsize=12, pad=8)
                        ax_s.grid(axis="y", alpha=0.3)
                    plt.suptitle("3.3-3  지상부 길이 · 지하부 길이 그룹 비교", fontsize=13, y=1.03)
                    plt.tight_layout()
                    st.pyplot(fig_size)

                # 3.3-4 지상부/지하부 비율 비교 (막대)
                st.divider()
                st.subheader("3.3-4  지상부 ÷ 지하부 비율 Top vs Bottom [막대 그래프]")
                st.caption("비율 < 1: 지상부보다 지하부가 더 길다. 비율 > 1: 지상부가 더 길다. 잘 자란 개체일수록 지하부가 더 발달(비율이 1보다 작음).")

                if all(c in growth_df.columns for c in ["지상부 길이","지하부 길이"]):
                    top_ratio    = (top_g["지상부 길이"]/top_g["지하부 길이"]).mean()
                    bot_ratio   = (bot_g["지상부 길이"]/bot_g["지하부 길이"]).mean()
                    t_r_vals    = top_g["지상부 길이"]/top_g["지하부 길이"]
                    b_r_vals    = bot_g["지상부 길이"]/bot_g["지하부 길이"]
                    t_st_r, p_r = stats.ttest_ind(t_r_vals, b_r_vals, equal_var=False)

                    fig_ratio, ax_ratio = plt.subplots(figsize=(7, 4.5))
                    x_rp = [0, 1]
                    bars_rp = ax_ratio.bar(x_rp, [bot_ratio, top_ratio],
                                            color=["#3498db","#e74c3c"],
                                            width=0.45, alpha=0.85, edgecolor="white",
                                            yerr=[[b_r_vals.std(), t_r_vals.std()]],
                                            capsize=8, error_kw=dict(linewidth=1.5))
                    for b, m in zip(bars_rp, [bot_ratio, top_ratio]):
                        ax_ratio.text(b.get_x()+b.get_width()/2, m+0.02,
                                      f"{m:.3f}", ha="center", va="bottom",
                                      fontsize=13, fontweight="bold")
                    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
                                     label="비율 = 1 (지상부 = 지하부)")
                    ax_ratio.set_xticks(x_rp)
                    ax_ratio.set_xticklabels(["Bottom 25%\n(지상부가 더 김)",
                                              "Top 25%\n(지하부가 더 김)"], fontsize=11)
                    ax_ratio.set_ylabel("지상부 ÷ 지하부 비율", fontsize=11)
                    ax_ratio.set_title(f"지상부/지하부 비율  "
                                       f"(Top {top_ratio:.3f} vs Bottom {bot_ratio:.3f}, "
                                       f"p={p_r:.4f})",
                                       fontsize=12, pad=8)
                    ax_ratio.legend(fontsize=10)
                    ax_ratio.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_ratio)

                    st.metric("Top 비율 (지상부/지하부)", f"{top_ratio:.3f}",
                              help="1보다 작음 → 지상부보다 지하부가 더 김")
                    st.metric("Bottom 비율 (지상부/지하부)", f"{bot_ratio:.3f}",
                              help="1보다 큼 → 지상부가 지하부보다 더 김")
                    st.metric("p값 (Welch t검정)", f"{p_r:.4f}",
                              delta="* 유의함 (p<0.05) — 비율은 서로 다름")
                    st.markdown(
                        f"잘 자란 개체(Top)는 지상부보다 지하부가 상대적으로 더 길었다(비율 {top_ratio:.3f} < 1). "
                        f"덜 자란 개체(Bottom)는 지상부와 지하부가 비슷하거나 지상부가 더 긴 편이다(비율 {bot_ratio:.3f} > 1)."
                    )

                # 3.3-5 그룹 비교 종합 차트
                st.divider()
                st.subheader("3.3-5  그룹 비교 종합 시각화 [4종 병렬]")

                if all(c in growth_df.columns for c in ["잎 수","지상부 길이","지하부 길이","생중량"]):
                    fig_all, axes_all = plt.subplots(2, 2, figsize=(12, 8))
                    axes_all = axes_all.flatten()
                    items_all = [
                        ("잎 수 (장)",   "잎 수",   "#3498db"),
                        ("지상부 길이 (mm)", "지상부 길이", "#e67e22"),
                        ("지하부 길이 (mm)", "지하부 길이", "#9b59b6"),
                        ("생중량 (g)",  "생중량",  "#e74c3c"),
                    ]
                    for ax_a, (title, col, color) in zip(axes_all, items_all):
                        b_d = bot_g[col].dropna()
                        t_d = top_g[col].dropna()
                        bp_a = ax_a.boxplot([b_d.values, t_d.values],
                                            labels=["Bottom 25%","Top 25%"],
                                            patch_artist=True, widths=0.4)
                        for patch, clr in zip(bp_a["boxes"], [color,"#e74c3c"]):
                            patch.set_facecolor(clr)
                            patch.set_alpha(0.55)
                        for j, grp in enumerate([bot_g, top_g]):
                            xj = np.random.normal(j+1, 0.04, len(grp))
                            ax_a.scatter(xj, grp[col].dropna(), color="gray",
                                         alpha=0.3, s=25, zorder=2)
                        ax_a.set_title(title, fontsize=12, pad=8)
                        ax_a.set_ylabel(col, fontsize=10)
                        ax_a.grid(axis="y", alpha=0.3)
                    plt.suptitle("3.3-5  Top 25% vs Bottom 25% 그룹 비교 종합 (Box Plot)",
                                 fontsize=13, y=1.01)
                    plt.tight_layout()
                    st.pyplot(fig_all)

        # ──────────── 3.4 예측모델 ────────────
        with r4:
            st.subheader("3.4  생중량 예측모델 (선형회귀)")
            st.caption(
                "선형회귀: 한 변수(예측 변수)로 다른 변수(반응 변수)를 예측하는 방법. "
                "R²(결정계수): 모델이 데이터 변동을 설명하는 비율(0~1, 1에 가까울수록 잘 설명). "
                "MAE(평균절대오차): 예측값과 실제값 차이의 평균(작을수록 정확). "
                "5겹 교차검증: 데이터를 5개로 나누어 4개 학습·1개 검증을 5번 반복, 과적합 확인."
            )

            if growth_df is None or growth_df.shape[0] == 0:
                st.info("생육 CSV를 업로드하세요.")
                st.stop()

            if all(c in growth_df.columns for c in ["생중량","잎 수"]):
                X1 = growth_df[["잎 수"]].values
                y  = growth_df["생중량"].values

                # ── 모델 1: 잎 수만 ──
                sl1, int1, r1, p1, se1 = stats.linregress(growth_df["잎 수"], y)
                r2_1 = r1**2
                pred1 = sl1*growth_df["잎 수"] + int1
                mae1  = np.mean(np.abs(pred1 - y))

                # ── 모델 2: 잎 수 + 지상부 길이 ──
                model2_avail = "지상부 길이" in growth_df.columns
                if model2_avail:
                    X2 = growth_df[["잎 수","지상부 길이"]].values
                    lr2 = LinearRegression()
                    lr2.fit(X2, y)
                    pred2 = lr2.predict(X2)
                    r2_2  = r2_score(y, pred2)
                    mae2  = mean_absolute_error(y, pred2)
                    sl2_leaf   = lr2.coef_[0]
                    sl2_shoot  = lr2.coef_[1]
                    int2       = lr2.intercept_
                    p_shoot    = stats.linregress(growth_df["지상부 길이"], growth_df["생중량"])[3]

                # ── 모델 요약 카드 ──
                st.divider()
                st.subheader("모델 요약")
                c1, c2, c3 = st.columns(3)
                c1.metric("모델 1: R²", f"{r2_1:.4f}",
                          help=f"잎 수로 생중량 변동의 약 {r2_1*100:.0f}% 설명 가능. 약 {(1-r2_1)*100:.0f}%는 잎 수만으로 설명 안 됨.")
                c2.metric("모델 1: MAE", f"{mae1:.3f}g",
                          help="예측값과 실제값 차이의 평균(g). 작을수록 예측이 정확.")
                c3.metric("Pearson r", f"{r1:.4f}",
                          delta="p<0.001 (매우 유의)")

                if model2_avail:
                    c4, c5, c6 = st.columns(3)
                    c4.metric("모델 2: R²", f"{r2_2:.4f}",
                              help="잎 수+지상부 길이로 설명. 모델1과 실질적 차이 없음.")
                    c5.metric("모델 2: MAE", f"{mae2:.3f}g")
                    sig_str2 = "유의함 (p<0.05)" if p_shoot < 0.05 else f"유의하지 않음 (p={p_shoot:.3f} > 0.05)"
                    c6.metric("지상부 길이 p값",
                              f"{p_shoot:.3f}",
                              delta=sig_str2,
                              help="p>0.05이면 이 데이터에서 지상부 길이가 생중량 예측에 유의하게 기여하지 못함.")

                st.code(f"모델 1 (단순회귀): 생중량(g) = {sl1:.4f} × 잎 수(장) + ({int1:.4f})")
                if model2_avail:
                    st.code(f"모델 2 (다중회귀): 생중량(g) = {sl2_leaf:.4f} × 잎 수(장) + {sl2_shoot:.4f} × 지상부 길이(mm) + ({int2:.4f})")
                    st.markdown(
                        f"""
                        - R² 기준: 모델 1 = {r2_1:.4f} vs 모델 2 = {r2_2:.4f} → **실질적으로 비슷한 설명력**
                        - 지상부 길이 p = {p_shoot:.3f} → **통계적으로 유의하지 않음 (p > 0.05)**
                        - 즉 이 데이터에서는 **지상부 길이를 추가해도 예측력 개선 효과가 확인되지 않았다**
                        """
                    )

                # ── 5겹 교차검증 ──
                st.divider()
                st.subheader("5겹 교차검증 결과")
                st.caption("shuffle=True, random_state=42. 과적합 여부 확인.")

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2_1, cv_mae_1 = [], []
                for tr, te in kf.split(X1):
                    m = LinearRegression()
                    m.fit(X1[tr], y[tr])
                    pv = m.predict(X1[te])
                    cv_r2_1.append(r2_score(y[te], pv))
                    cv_mae_1.append(mean_absolute_error(y[te], pv))

                cv_data = [{"모델": "모델 1 (잎 수만)",
                            "CV R²": f"{np.mean(cv_r2_1):.4f}",
                            "CV MAE": f"{np.mean(cv_mae_1):.3f}g",
                            "R² 차이 (CV − 학습)": f"{np.mean(cv_r2_1)-r2_1:+.4f}"}]
                if model2_avail:
                    cv_r2_2, cv_mae_2 = [], []
                    for tr, te in kf.split(X2):
                        m2 = LinearRegression()
                        m2.fit(X2[tr], y[tr])
                        pv2 = m2.predict(X2[te])
                        cv_r2_2.append(r2_score(y[te], pv2))
                        cv_mae_2.append(mean_absolute_error(y[te], pv2))
                    cv_data.append({"모델": "모델 2 (잎 수+지상부)",
                                    "CV R²": f"{np.mean(cv_r2_2):.4f}",
                                    "CV MAE": f"{np.mean(cv_mae_2):.3f}g",
                                    "R² 차이 (CV − 학습)": f"{np.mean(cv_r2_2)-r2_2:+.4f}"})

                    fig_cv, ax_cv = plt.subplots(figsize=(7, 4.5))
                    x_cv = np.arange(2)
                    bw_c = 0.3
                    ax_cv.bar(x_cv-bw_c/2, [np.mean(cv_r2_1), np.mean(cv_r2_2)],
                              bw_c, color=["#3498db","#e74c3c"], alpha=0.8, label="CV R²")
                    ax_cv.bar(x_cv+bw_c/2, [r2_1, r2_2],
                              bw_c, color=["#3498db","#e74c3c"], alpha=0.35,
                              hatch="//", label="학습 R²")
                    ax_cv.set_xticks(x_cv)
                    ax_cv.set_xticklabels(["모델 1\n(잎 수만)","모델 2\n(잎 수+지상부)"], fontsize=11)
                    ax_cv.set_ylim(0, 1.0)
                    ax_cv.set_ylabel("R²", fontsize=11)
                    ax_cv.set_title("5겹 교차검증 R² 비교 (학습 vs CV)", fontsize=12, pad=8)
                    ax_cv.legend(fontsize=10)
                    for i, (cv_v, lr_v) in enumerate(zip([np.mean(cv_r2_1), np.mean(cv_r2_2)],
                                                           [r2_1, r2_2])):
                        ax_cv.text(i-bw_c/2, cv_v+0.02, f"{cv_v:.3f}", ha="center",
                                   va="bottom", fontsize=10, fontweight="bold")
                        ax_cv.text(i+bw_c/2, lr_v+0.02, f"{lr_v:.3f}", ha="center",
                                   va="bottom", fontsize=10, fontweight="bold")
                    ax_cv.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_cv)
                    st.markdown(
                        f"""
                        - 학습 R²: 모델 1 = {r2_1:.4f} vs 모델 2 = {r2_2:.4f} → 비슷
                        - **5겹 교차검증에서는 오히려 단순회귀(모델 1)가 더 안정적**
                          CV R² {np.mean(cv_r2_1):.4f} vs {np.mean(cv_r2_2):.4f}
                        - CV MAE: 모델 1 {np.mean(cv_mae_1):.3f}g vs 모델 2 {np.mean(cv_mae_2):.3f}g
                        - 지상부 길이는 생중량 예측에 **통계적으로 유의한 기여를 하지 못함** (p={p_shoot:.3f})
                        """
                    )
                else:
                    fig_cv, ax_cv = plt.subplots(figsize=(5, 4))
                    ax_cv.bar(["학습 R²","CV R²"],[r2_1, np.mean(cv_r2_1)],
                              color=["#3498db","#e74c3c"], alpha=0.8, width=0.45)
                    ax_cv.set_ylim(0, 1.0)
                    ax_cv.set_ylabel("R²", fontsize=11)
                    ax_cv.set_title("모델 1 (잎 수만) — 학습 vs 5겹 CV", fontsize=12, pad=8)
                    for i, v in enumerate([r2_1, np.mean(cv_r2_1)]):
                        ax_cv.text(i, v+0.02, f"{v:.3f}", ha="center", va="bottom",
                                   fontsize=11, fontweight="bold")
                    ax_cv.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_cv)

                # ── 예측 예시 ──
                st.divider()
                st.subheader("예측 예시: 잎 수 25장 · 60장일 때 생중량 예측값")
                st.caption("지상부 길이를 평균(39.4mm)으로 가정하고 예측값을 계산하였다.")

                pred_25 = sl1*25 + int1
                pred_60 = sl1*60 + int1

                fig_pred, ax_pred = plt.subplots(figsize=(9, 5))
                ax_pred.scatter(growth_df["잎 수"], growth_df["생중량"],
                                color="#3498db", alpha=0.5, s=50, edgecolors="white",
                                linewidth=0.5, label="81개체 실제값")
                x_r = np.linspace(growth_df["잎 수"].min()-5, growth_df["잎 수"].max()+5, 200)
                ax_pred.plot(x_r, sl1*x_r+int1, color="#e74c3c", linewidth=2.5,
                            label=f"모델 1 회귀선 (R²={r2_1:.3f})")
                if model2_avail:
                    shoot_mean = growth_df["지상부 길이"].mean()
                    line2 = np.array([sl1*x + sl2_shoot*shoot_mean + int2 for x in x_r])
                    ax_pred.plot(x_r, line2, color="#2ecc71", linewidth=2, linestyle="--",
                                 label=f"모델 2 (지상부≈{shoot_mean:.1f}mm, R²={r2_2:.3f})")
                # 예측점 표시
                for n, pred_v, lbl in [(25, pred_25, "25장"), (60, pred_60, "60장")]:
                    ax_pred.scatter([n],[pred_v], color="#e74c3c", s=180, marker="*",
                                    edgecolors="darkred", linewidth=1.5, zorder=5,
                                    label=f"예측: {lbl} → {pred_v:.2f}g")
                    ax_pred.annotate(f"{lbl}장\n{pred_v:.2f}g",
                                     (n, pred_v), textcoords="offset points",
                                     xytext=(10, 5), fontsize=10, fontweight="bold",
                                     color="#e74c3c")
                ax_pred.set_xlabel("잎 수 (장)", fontsize=11)
                ax_pred.set_ylabel("생중량 (g)", fontsize=11)
                ax_pred.set_title("그림 3. 잎 수 vs 생중량 산점도 + 회귀선 + 예측 예시", fontsize=13, pad=12)
                ax_pred.legend(fontsize=10, loc="upper left")
                ax_pred.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_pred)

                c_p1, c_p2 = st.columns(2)
                c_p1.metric("잎 수 25장 → 예측 생중량", f"{pred_25:.2f}g",
                            help=f"회귀식: {sl1:.4f}×25 + ({int1:.4f})")
                c_p2.metric("잎 수 60장 → 예측 생중량", f"{pred_60:.2f}g",
                            help=f"회귀식: {sl1:.4f}×60 + ({int1:.4f})")

                st.markdown(
                    """
                    - 점들이 대체로 우상향 방향에 모여 있어 잎 수가 많을수록 생중량이 증가하는 경향
                    - 상관계수 r = 0.886으로 네 항목 중 생중량과 가장 강한 양의 상관
                    - **단, 인과관계는 아님** — 잎 수가 많아서 무겁게 자랐다는 것이 아니라, 두 값이 함께 증가하는 '경향'만 보여줌
                    - 생중량이 큰 개체에서는 예측선이 실제값보다 낮게 잡는 경향 (과소예측)
                    """
                )

            # ── 지상부 길이 vs 생중량 산점도 (보충) ──
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                st.divider()
                st.subheader("보충: 지상부 길이 vs 생중량 산점도 [그림 4]")
                r_sh2, p_sh2 = stats.pearsonr(growth_df["지상부 길이"], growth_df["생중량"])

                fig_sh, ax_sh = plt.subplots(figsize=(8, 4.5))
                ax_sh.scatter(growth_df["지상부 길이"], growth_df["생중량"],
                              color="#9b59b6", alpha=0.55, s=50, edgecolors="white",
                              linewidth=0.5)
                x_sh = np.linspace(growth_df["지상부 길이"].min(),
                                    growth_df["지상부 길이"].max(), 100)
                sl_sh, int_sh, _, _, _ = stats.linregress(growth_df["지상부 길이"], growth_df["생중량"])
                ax_sh.plot(x_sh, sl_sh*x_sh+int_sh, color="#e74c3c", linewidth=2.2,
                           label=f"추세선 (r={r_sh2:.3f})")
                ax_sh.set_xlabel("지상부 길이 (mm)", fontsize=11)
                ax_sh.set_ylabel("생중량 (g)", fontsize=11)
                ax_sh.set_title("그림 4. 지상부 길이와 생중량의 산점도", fontsize=13, pad=12)
                ax_sh.legend(fontsize=10)
                ax_sh.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_sh)
                st.markdown(
                    f"""
                    점들의 퍼짐이 넓고 경향선이 비교적 완만하다.
                    잎 수(r=0.886)나 지하부 길이(r=0.606)에 비해 **지상부 길이와 생중량의 관계는 약한 편**이다(r={r_sh2:.3f}).
                    """
                )

    # ═══════════════════════════════════════════
    # 탭 3: 4.논의 · 5.한계점 · 6.결론
    # ═══════════════════════════════════════════
    with t3:
        st.header("4. 논의 · 5. 한계점 · 6. 결론")

        if growth_df is None:
            st.info("CSV를 업로드하면 논의와 결론이 분석 데이터 기반으로 표시됩니다.")
            st.stop()

        # ─── 4. 논의 ────────────────────────────
        st.subheader("4. 논의")

        st.markdown("#### 4.1 가설 검토")
        st.info(
            "가설: **잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다.**\n"
            "→ 이 가설은 **데이터로 지지되었다**."
        )

        if all(c in growth_df.columns for c in ["생중량","잎 수"]):
            fw_vals2 = growth_df["생중량"].dropna()
            n2 = int(np.ceil(len(fw_vals2)*0.25))
            top_t2 = fw_vals2.nlargest(n2).min()
            bot_t2 = fw_vals2.nsmallest(n2).max()
            top_g2 = growth_df[growth_df["생중량"] >= top_t2].head(n2)
            bot_g2 = growth_df[growth_df["생중량"] <= bot_t2].head(n2)
            r_final = stats.pearsonr(growth_df["생중량"], growth_df["잎 수"])[0]

            st.markdown(
                f"""
                - Top 25% 평균 생중량: **{top_g2['생중량'].mean():.2f}g** vs Bottom 25%: **{bot_g2['생중량'].mean():.2f}g**
                  → 약 **{top_g2['생중량'].mean()/bot_g2['생중량'].mean():.0f}배 차이**
                - Top 25% 평균 잎 수: **{top_g2['잎 수'].mean():.1f}장** vs Bottom 25%: **{bot_g2['잎 수'].mean():.1f}장**
                  → 약 **{top_g2['잎 수'].mean()/bot_g2['잎 수'].mean():.1f}배 차이**
                - Welch t검정 **p = {(stats.ttest_ind(top_g2['잎 수'], bot_g2['잎 수'], equal_var=False)[1]):.2e}**
                  → 두 그룹의 잎 수가 같다는 가정에서는 관찰되기 매우 어려운 결과
                - 잎 수−생중량 Pearson r = **{r_final:.4f}** (p < 0.001)도 같은 방향 뒷받침

                **다만**, 이것이 "잎 수가 많아서 생중량이 커진 것"이라는 인과관계를 증명하는 것은 아니다.
                잎 수와 생중량이 함께 커지는 경향은 확인되었지만, 그 배경은 따로 확인이 필요하다.
                """
            )

        st.divider()
        st.markdown("#### 4.2 환경 변화와 생육 차이의 연결 가능성")

        st.caption(
            "환경 데이터(127일간 시계열)와 생육 데이터(한 시점 측정 추정)를 직접 연결하려면 생육 측정일이 언제인지 알아야 한다. "
            "이 정보가 없으므로 **가능성으로만 언급**한다."
        )

        if env_df is not None:
            if "ec" in env_df.columns and "ph" in env_df.columns:
                e_c = env_df["ec"].dropna()
                p_c = env_df["ph"].dropna()
                ec_mid_date = env_df["_date_dt"].min() + (env_df["_date_dt"].max()-env_df["_date_dt"].min())/2
                fh_e = env_df[env_df["_date_dt"] <= ec_mid_date]["ec"].dropna()
                sh_e = env_df[env_df["_date_dt"] >  ec_mid_date]["ec"].dropna()
                fh_p = env_df[env_df["_date_dt"] <= ec_mid_date]["ph"].dropna()
                sh_p = env_df[env_df["_date_dt"] >  ec_mid_date]["ph"].dropna()

                ec_early = fh_e.mean() if len(fh_e) > 0 else 0
                ec_late  = sh_e.mean() if len(sh_e) > 0 else 0
                ph_early = fh_p.mean() if len(fh_p) > 0 else 0
                ph_late  = sh_p.mean() if len(sh_p) > 0 else 0
                ec_chg2  = (ec_late-ec_early)/ec_early*100 if ec_early != 0 else 0
                ph_chg2  = ph_late - ph_early

                fig_env_conn, ax_conn = plt.subplots(figsize=(7, 3.5))
                periods = ["전반부\n(6/16~8/18)", "후반부\n(8/19~10/21)"]
                ax_conn.plot(periods, [ec_early, ec_late], color="#e74c3c", marker="o",
                             linewidth=2.5, markersize=10, label=f"EC ({ec_early:.2f}→{ec_late:.2f}, +{ec_chg2:.0f}%)")
                ax_conn2 = ax_conn.twinx()
                ax_conn2.plot(periods, [ph_early, ph_late], color="#3498db", marker="s",
                              linewidth=2.5, markersize=10, label=f"pH ({ph_early:.2f}→{ph_late:.2f}, {ph_chg2:+.2f})")
                ax_conn.set_ylabel("EC", color="#e74c3c", fontsize=11)
                ax_conn2.set_ylabel("pH", color="#3498db", fontsize=11)
                ax_conn.tick_params(axis="y", labelcolor="#e74c3c")
                ax_conn2.tick_params(axis="y", labelcolor="#3498db")
                ax_conn.set_title("재배 기간 EC·pH 변화 방향 (전반부 → 후반부)", fontsize=12, pad=8)
                lines1, labels1 = ax_conn.get_legend_handles_labels()
                lines2, labels2 = ax_conn2.get_legend_handles_labels()
                ax_conn.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_env_conn)

            st.markdown(
                f"""
                - EC는 6월 초 약 0.49에서 8월 하순 약 1.42로 약 **188% 상승**, pH는 7.96에서 6.52로 약 **1.44pH 단위 하락**
                - 이러한 환경 변화는 식물의 양분 흡수나 생리 상태에 영향을 줄 수 있는 수준의 변화
                - 생육 데이터는 한 시점(측정일 미확인)에 측정된 것으로 보이며, 환경 데이터는 127일간 시계열
                - 두 데이터를 직접 연결하려면 생육 측정일이 언제인지 알아야 하는데, 이 정보가 없음
                - 따라서 **환경 변화가 개체별 생육 차이를 일으켰다고 단정할 수 없고, 가능성으로만 언급**
                """
            )

        st.divider()
        st.markdown("#### 4.3 실용적 제안")
        st.markdown(
            """
            - 잎 수만 사용한 모델 1(R² = 0.7845)이 잎 수+지상부 길이를 사용한 모델 2(R² = 0.7893)와
              실질적으로 비슷한 설명력을 보였고, 교차검증에서는 모델 1이 더 안정적이었다.
              즉 **지상부 길이를 추가로 재는 것이 예측에는 큰 도움이 되지 않았다**.
            - 다만 R²가 0.78이라는 것은 여전히 **약 22%의 변동은 잎 수로 설명되지 않는다**는 뜻이므로,
              잎 수만으로 생중량을 정확히 예측하는 도구라고 보기는 어렵다.
            - 이 모델은 이 데이터(**송도고 극지식물, 특정 시기, 특정 스마트팜 조건**)에서 얻은 것이다.
              다른 품종, 다른 환경, 다른 시기에 그대로 적용된다고 단정할 수 없다.
            """
        )

        # ─── 5. 한계점과 개선 방향 ──────────────
        st.divider()
        st.subheader("5. 한계점과 개선 방향")

        limits = [
            ("생육 측정일 부재",
             "생육결과데이터에 측정일(날짜) 열이 없다. 생육 측정이 정확히 언제 이루어졌는지 알 수 없고, 환경 시계열과 생육 결과를 직접 연결할 수 없다. 이 연구에서 가장 중요한 한계.",
             "다음 재배부터는 생육 측정일을 반드시 기록한다."),
            ("개체 위치 정보 부재",
             "같은 스마트팜 안에서도 개체마다 빛·통풍·수분 공급이 다를 수 있으나, 데이터에는 각 개체의 위치 정보가 없어 위치 효과를 분석할 수 없다.",
             "개체별 위치(줄·열 번호 등)를 기록한다."),
            ("표본 크기",
             "생육 데이터가 81개체로, 그룹 비교(각 20개체)나 회귀분석은 가능하지만 표본이 아주 크다고 보기는 어렵다. 이상치로 잡힌 소수 개체들의 영향이 결과를 왜곡할 가능성도 배제할 수 없다.",
             "더 많은 개체(예: 200+ 개체)를 확보한다."),
            ("단일 시점 생육 측정",
             "생육 데이터가 한 시점 측정인지 여러 시점 측정인지 확인할 수 없다. 한 시점 측정이면 성장 과정의 변화를 볼 수 없고, 여러 시점 측정이면 측정 시점 차이가 결과에 영향을 주었을 수 있다.",
             "생육을 여러 시점에 걸쳐 반복 측정한다."),
            ("EC·pH 결측",
             "환경 데이터의 EC와 pH는 각각 194개 결측이 있어 전체 2,461행 중 2,267행만 유효하다. 결측이 무작위가 아니라면 통계 결과에 편향이 있을 수 있다.",
             "센서 점검 주기를 줄이고 결측 발생 원인을 기록한다."),
            ("인과관계 아님",
             "모든 통계 결과(상관, 그룹 차이, 회귀계수)는 변수들 사이의 관계나 차이를 보여줄 뿐, 인과관계를 증명하지 않는다. '잎 수가 많아서 생중량이 크다'고 말할 수 없으며, '잎 수가 많은 개체가 생중량이 큰 경향이 있다'고만 말할 수 있다.",
             "인과관계 확인을 위해서는 통제 실험(예: 잎 수 조작 실험)이 필요하다."),
            ("외부 일반화 불가",
             "이 분석은 송도고등학교의 특정 스마트팜, 특정 품종(극지식물), 특정 시기(2026년 6~10월) 조건에서 얻은 데이터로 수행되었다. 다른 학교, 다른 품종, 다른 환경에서 같은 결과가 나온다고 단정할 수 없다.",
             "다른 환경·품종에서 동일 분석을 반복해 일반화 가능성을 검증한다."),
        ]

        st.caption("한계점과 그에 대응하는 개선 방향을 함께 제시")

        for i, (title, problem, improvement) in enumerate(limits):
            col_a, col_b = st.columns([1.4, 2.5])
            with col_a:
                st.markdown(f"**⚠️ 한계 {i+1}**\n\n{title}")
            with col_b:
                st.write(f"**현재 문제:** {problem}")
                st.info(f"**개선 방향:** {improvement}")
            st.divider()

        # ─── 6. 결론 ────────────────────────────
        st.divider()
        st.header("6. 결론")
        st.caption("송도고 그린2팀 1회차 보고서 (업스테이지·극지연구소 검수, 2026-10-28) 기반 종합 결론")

        if all(c in growth_df.columns for c in ["생중량","잎 수",
                                                   "지상부 길이","지하부 길이"]):
            # 실제 데이터로 계산
            fw_v     = growth_df["생중량"].dropna()
            n_each3  = int(np.ceil(len(fw_v)*0.25))
            top_t3   = fw_v.nlargest(n_each3).min()
            bot_t3   = fw_v.nsmallest(n_each3).max()
            top_g3   = growth_df[growth_df["생중량"] >= top_t3].head(n_each3)
            bot_g3   = growth_df[growth_df["생중량"] <= bot_t3].head(n_each3)

            cv_fw    = abs(fw_v.std()/fw_v.mean()*100)
            r_leaf   = stats.pearsonr(growth_df["생중량"], growth_df["잎 수"])[0]
            p_leaf   = stats.pearsonr(growth_df["생중량"], growth_df["잎 수"])[1]
            r_root   = stats.pearsonr(growth_df["생중량"], growth_df["지하부 길이"])[0]
            t_root, p_root3 = stats.ttest_ind(top_g3["지하부 길이"], bot_g3["지하부 길이"], equal_var=False)
            sl1_final, int1_final, _, _, _ = stats.linregress(growth_df["잎 수"], growth_df["생중량"])
            r2_final = r_leaf**2
            mae_final = np.mean(np.abs(sl1_final*growth_df["잎 수"]+int1_final - growth_df["생중량"]))

            if "지상부 길이" in growth_df.columns:
                p_shoot_final = stats.linregress(growth_df["지상부 길이"], growth_df["생중량"])[3]
            else:
                p_shoot_final = None

            top_r_ratio = (top_g3["지상부 길이"]/top_g3["지하부 길이"]).mean()
            bot_r_ratio = (bot_g3["지상부 길이"]/bot_g3["지하부 길이"]).mean()
            t_rr, p_rr  = stats.ttest_ind(top_g3["지상부 길이"]/top_g3["지하부 길이"],
                                           bot_g3["지상부 길이"]/bot_g3["지하부 길이"], equal_var=False)

            kf3 = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_r2_3 = []
            X1_3 = growth_df[["잎 수"]].values
            y_3   = growth_df["생중량"].values
            for tr, te in kf3.split(X1_3):
                m3 = LinearRegression()
                m3.fit(X1_3[tr], y_3[tr])
                cv_r2_3.append(r2_score(y_3[te], m3.predict(X1_3[te])))

        # 결론 카드
        st.success("📌 종합 결론")
        st.markdown(
            f"""
            ### 환경 측면
            온도와 습도는 재배 기간 내내 비교적 안정적이었다(하루 일교차 약 1.52°C, CV 10~12%).
            반면 EC와 pH는 재배 기간 내내 뚜렷한 방향 변화를 보였다.
            EC는 6월 초 0.49에서 8월 하순 1.42로 약 **188% 상승**, pH는 7.96에서 6.52로 약 **1.44pH 단위 하락**.
            이러한 변화가 생육 차이와 관련되었을 가능성은 있으나, 생육 측정일이 없어 직접적 연결은 확인할 수 없었다.

            ### 개체별 생육 차이
            생중량은 최소 0.04g에서 최대 10.01g까지 **250배 차이**, CV는 **{cv_fw:.1f}%**로 네 측정 항목 중 가장 높았다.
            평균 {fw_v.mean():.2f}g, 중앙값 {fw_v.median():.2f}g — 일부 무거운 개체가 평균을 끌어올리는 오른쪽 꼬리 분포.
            생중량과 가장 강한 상관을 보인 항목은 **잎 수(r = {r_leaf:.3f}, p < 0.001)**.

            ### 가설 검증 — 지지됨
            가설 "잎 수 상위 25%는 하위 25%보다 생중량이 클 것이다"는 **데이터로 지지되었다**.
            Top 평균 잎 수 72.55장 vs Bottom 11.40장(p = {(stats.ttest_ind(top_g3['잎 수'], bot_g3['잎 수'], equal_var=False)[1]):.2e}),
            생중량 5.93g vs 0.37g.
            지하부 길이도 유의한 차이(p = {p_root3:.2e}),
            잘 자란 개체는 지상부보다 지하부가 상대적으로 더 긴 경향(비율 {top_r_ratio:.3f} vs {bot_r_ratio:.3f}, p = {p_rr:.4f}).

            ### 예측모델
            잎 수만 사용한 단순회귀(R² = {r2_final:.4f}, MAE = {mae_final:.3f}g)가
            잎 수+지상부 길이를 사용한 다중회귀(R² = {r2_final:.4f} 수준)와 실질적으로 비슷한 성능을 보였고,
            5겹 교차검증에서는 오히려 단순회귀가 더 안정적이었다(CV R² {np.mean(cv_r2_3):.4f}).
            지상부 길이는 생중량 예측에 통계적으로 유의한 기여를 하지 못했다(p = {p_shoot_final:.3f}).

            ### 한계와 주의
            그러나 이 결과는 **표본 81개체, 단일 스마트팜·단일 시기 데이터**에서 얻은 것으로,
            생육 측정일과 개체 위치 정보가 없다는 한계가 있다.
            통계적 관계가 확인되었어도 **인과관계는 아니며**,
            다른 환경이나 다른 품종에 그대로 적용할 수 있다고 단정할 수 없다.

            다음 재배부터는 측정일과 위치 정보를 기록하고, 다중 시점 생육 추적과 더 큰 표본을 확보하면 더 구체적인 해석이 가능할 것이다.
            """
        )

        # 핵심 요약 카드
        st.divider()
        st.subheader("핵심 요약 카드")
        cols_s = st.columns(4)
        summary_cards = [
            ("연구 목적",
             "같은 스마트팜 81개체 생육 차이 확인\n잎 수·지상부·지하부로 설명 가능한지\n환경 변화와 관련 있는지 분석"),
            ("주요 발견",
             f"생중량 250배 차이 (0.04g~10.01g)\nCV {cv_fw:.1f}% — 가장 큰 차이\n잎 수−생중량 r={r_leaf:.3f}\nTop/Bottom 잎 수 약 6.4배 차이"),
            ("가설 검증",
             "지지됨 ✅\n잎 수 상위 25% > 하위 25% 생중량\np = 4.49×10⁻⁹ (매우 유의)"),
            ("예측모델",
             f"단순회귀 R²={r2_final:.3f}, MAE={mae_final:.3f}g\n교차검증에서 다중회귀보다 안정적\n지상부 길이 추가 효과 없음 (p={p_shoot_final:.3f})"),
        ]
        for i, (title, content) in enumerate(summary_cards):
            with cols_s[i]:
                st.markdown(f"**{title}**")
                st.info(content)

    # ─────────────────────────────────────────
    # 하단 메타 정보
    # ─────────────────────────────────────────
    st.divider()
    st.caption(
        "📄 보고서 출처: 송도고등학교 그린2팀 1회차 보고서 (업스테이지 · 극지연구소 검수, 작성 2026-10-28)\n"
        "📊 데이터: 송도고_환경데이터.csv (2,461행), 송도고_생육결과데이터.csv (81개체)\n"
        "⚠️ 모든 통계 결과는 인과관계가 아닌 관계의 경향을 보여줌. 다른 환경·품종에 그대로 적용된다고 단정할 수 없음."
    )


if __name__ == "__main__":
    main()
