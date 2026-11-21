#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
=============================================
CSV 업로드 시 인코딩(utf-8-sig/cp949) 오류 방지
학교마다 다른 열 이름(KeyError) 방지
한글 폰트 깨짐 완벽 방지
Streamlit Cloud에서 에러 없이 실행

고등학생이 이해할 수 있도록 통계 용어에 짧은 설명을 붙였습니다.
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
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. 한글 폰트 설정 (Streamlit Cloud 대응)
# ─────────────────────────────────────────────
def setup_korean_font():
    """NotoSansCJK 폰트를 직접 등록하여 한글 깨짐을 방지"""
    # 폰트 캐시 제거 (매 실행 시 클린 상태 보장)
    cache_dir = os.path.expanduser("~/.cache/matplotlib")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # 실제 존재하는 폰트 파일 경로
    font_regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_bold    = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

    # 폰트 파일 등록
    for fp in [font_regular, font_bold]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)

    # FontProperties로 실제 폰트 이름 획득
    reg_prop = fm.FontProperties(fname=font_regular)
    font_name = reg_prop.get_name()   # "Noto Sans CJK JP"

    # matplotlib 전역 설정
    plt.rcParams["font.family"]       = font_name
    plt.rcParams["font.size"]         = 11
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"]  = "bold"
    plt.rcParams["figure.dpi"]        = 100


# ─────────────────────────────────────────────
# 1. CSV 읽기 (인코딩·열 이름 대응)
# ─────────────────────────────────────────────
def read_csv_flexible(uploaded_file):
    """utf-8-sig → cp949 → euc-kr → latin-1 순서로 시도"""
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
    """여러 후보 열 이름 중 실제 존재하는 열을 찾는다 (KeyError 방지)"""
    if df is None or df.shape[1] == 0:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    # 정규화 비교 (공백 제거, 단위 분리)
    norm = {str(c).strip().replace("(장)","").replace("(mm)","").replace("(g)","").replace("(%)","").replace("(degC)","").strip(): c
            for c in df.columns}
    for c in candidates:
        if str(c).strip().replace("(장)","").replace("(mm)","").replace("(g)","").replace("(%)","").replace("(degC)","").strip() in norm:
            return norm[str(c).strip().replace("(장)","").replace("(mm)","").replace("(g)","").replace("(%)","").replace("(degC)","").strip()]
    return None


# ─────────────────────────────────────────────
# 2. 환경 데이터 전처리
# ─────────────────────────────────────────────
def preprocess_env(df):
    if df is None:
        return None
    df = df.copy()

    # 시간 열 찾기
    time_col = find_actual_column(df,
        ["time","Time","datetime","날짜","일시","날짜/시간","date","일시_시간"])
    if time_col is None:
        # 패턴으로 찾기
        for col in df.columns:
            s = str(df[col].iloc[0])
            if "2026" in s or ("_" in s and any(c.isdigit() for c in s)):
                time_col = col
                break
    if time_col is None:
        st.warning("시간 열을 찾을 수 없습니다.")
        return None

    # 시간 파싱 (두 형식 모두 대응)
    df["_parsed_time"] = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    n_parsed = df["_parsed_time"].notna().sum()
    if n_parsed == 0:
        # '20260715_0' 같은 형식 처리
        def try_parse(v):
            s = str(v).strip()
            # '20260715_0' → '2026-07-15 00:00:00'
            if len(s) == 10 and s[4] == '0' and s[7] == '0':
                try:
                    return pd.to_datetime(s[:4]+"-"+s[4:6]+"-"+s[6:8]+" "+s[9:]+":00:00", errors="coerce")
                except:
                    pass
            return pd.NaT
        df["_parsed_time"] = df[time_col].apply(try_parse)

    df = df.dropna(subset=["_parsed_time"]).copy()
    df["_hour"]  = df["_parsed_time"].dt.hour
    df["_date"]  = df["_parsed_time"].dt.date
    df["_date_dt"] = pd.to_datetime(df["_date"])

    result = {"_parsed_time": df["_parsed_time"], "_hour": df["_hour"], "_date": df["_date"], "_date_dt": df["_date_dt"]}

    col_map = {
        "temperature": ["temperature","온도","temp","온도(degC)","기온"],
        "humidity":    ["humidity","습도","hum","습도(%)"],
        "ec":          ["ec","EC","ec"],
        "ph":          ["ph","PH","pH","PH"],
        "water_temp":  ["water_temp","수온","water_temp"],
        "co2":         ["co2","CO2","co2"],
    }
    for key, cands in col_map.items():
        col = find_actual_column(df, cands)
        if col is not None:
            result[key] = pd.to_numeric(df[col], errors="coerce")

    return pd.DataFrame(result)


# ─────────────────────────────────────────────
# 3. 생육 데이터 전처리
# ─────────────────────────────────────────────
def preprocess_growth(df):
    if df is None:
        return None
    df = df.copy()

    # 개체번호
    id_col = find_actual_column(df, ["개체번호","번호","ID","id","시료번호","개체"])
    if id_col is not None:
        df["개체번호"] = pd.to_numeric(df[id_col], errors="coerce")

    # 잎 수
    leaf_col = find_actual_column(df, ["잎 수(장)","잎 수","leaf","잎수","잎 수 (장)","잎 수 (장)","엽수"])
    if leaf_col is not None:
        df["잎 수"] = pd.to_numeric(df[leaf_col], errors="coerce")

    # 지상부 길이
    shoot_col = find_actual_column(df, ["지상부 길이(mm)","지상부 길이","지상부길이","지상부 길이 (mm)","지상부 길이 (mm)","shoot_length","shoot"])
    if shoot_col is not None:
        df["지상부 길이"] = pd.to_numeric(df[shoot_col], errors="coerce")

    # 지하부 길이
    root_col = find_actual_column(df, ["지하부길이(mm)","지하부 길이(mm)","지하부 길이","지하부길이","지하부 길이 (mm)","root_length","root"])
    if root_col is not None:
        df["지하부 길이"] = pd.to_numeric(df[root_col], errors="coerce")

    # 생중량
    fw_col = find_actual_column(df, ["생중량(g)","생중량","fresh_weight","생중량 (g)","생중량 (g)","fw","생체중"])
    if fw_col is not None:
        df["생중량"] = pd.to_numeric(df[fw_col], errors="coerce")

    # 생중량이 없을 때: 지상부질량 + 지하부질량으로 계산
    if "생중량" not in df.columns or df["생중량"].isna().all():
        sm_col = find_actual_column(df, ["지상부질량(g)","지상부질량","지상부 질량","shoot_mass","지상부질량 (g)","지상부질량"])
        rm_col = find_actual_column(df, ["지하부질량(g)","지하부질량","지하부 질량","root_mass","지하부질량 (g)","지하부질량"])
        if sm_col is not None and rm_col is not None:
            df["생중량"] = pd.to_numeric(df[sm_col], errors="coerce") + pd.to_numeric(df[rm_col], errors="coerce")

    # 지상부질량
    sm_col2 = find_actual_column(df, ["지상부질량(g)","지상부질량","지상부 질량","shoot_mass","지상부질량 (g)","지상부질량"])
    if sm_col2 is not None:
        df["지상부질량"] = pd.to_numeric(df[sm_col2], errors="coerce")

    # 지하부질량
    rm_col2 = find_actual_column(df, ["지하부질량(g)","지하부질량","지하부 질량","root_mass","지하부질량 (g)","지하부질량"])
    if rm_col2 is not None:
        df["지하부질량"] = pd.to_numeric(df[rm_col2], errors="coerce")

    keep_cols = ["개체번호","잎 수","지상부 길이","지하부 길이","생중량","지상부질량","지하부질량"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["개체번호"]).copy()
    df = df.sort_values("개체번호").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 4. Streamlit 대시보드
# ─────────────────────────────────────────────
def main():
    setup_korean_font()
    st.set_page_config(page_title="송도고 극지식물 생육 대시보드", layout="wide")

    st.title("극지식물 개체별 생육 차이 분석 대시보드")
    st.caption("같은 스마트팜에서 재배한 극지식물 81개체의 생육 결과 데이터와 환경 모니터링 데이터를 종합 분석합니다.")
    st.caption("출처: 송도고등학교 그린2팀 1회차 보고서 (업스테이지·극지연구소 검수, 2026-10-28)")

    st.sidebar.header("📁 데이터 업로드")
    env_file    = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
    growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.info(
        "학교마다 열 이름이 다릅니다.\n"
        "형식 A: time, temperature, humidity, ph, ec, water_temp, co2\n"
        "형식 B: categories, 온도, 습도, EC, PH\n"
        "생육: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g) ..."
    )

    # 데이터 로드
    env_df    = None
    growth_df = None

    if env_file:
        df_raw = read_csv_flexible(env_file)
        env_df = preprocess_env(df_raw)
        if env_df is None or env_df.shape[0] == 0:
            st.sidebar.error("환경 CSV의 시간 열을 찾을 수 없습니다. 열 이름이 'time'인지 확인하세요.")

    if growth_file:
        df_raw = read_csv_flexible(growth_file)
        growth_df = preprocess_growth(df_raw)
        if growth_df is None or growth_df.shape[0] == 0:
            st.sidebar.error("생육 CSV의 개체번호 열을 찾을 수 없습니다.")

    # ─── 탭 정의 ──────────────────────────────
    tab_env, tab_growth, tab_analysis = st.tabs([
        "🌡️  환경 데이터 분석",
        "🌱  개체별 생육 결과",
        "📊  분석 결과 · 가설 검증 · 결론"
    ])

    # ══════════════════════════════════════════
    # 탭 1: 환경 데이터 분석
    # ══════════════════════════════════════════
    with tab_env:
        st.header("🌡️  스마트팜 환경 데이터 분석")
        st.caption("2026년 6월 16일 ~ 10월 21일, 127일간 1시간 간격으로 측정. 온도·습도·EC·pH 분석.")

        if env_df is None or env_df.shape[0] == 0:
            st.info("환경 데이터 CSV를 업로드하세요.")
            st.markdown("""
            **예시 형식 A (송도고 실제):**
```
time, temperature, humidity, ph, ec, water_temp, co2
2026-06-16 06:00:00, 23.1, 54.6, 8.13, 0.27, 23.7, 377.0
```
            **예시 형식 B (다른 학교):**
```
categories, 온도, 습도, EC, PH
20260715_0, 24.5, 60.1, 0.5, 7.2
```
            """)
        else:
            n_rows = len(env_df)
            st.success(f"✅ {n_rows:,}행 로드됨 (측정 기간: {env_df['_date_dt'].min().strftime('%Y-%m-%d')} ~ {env_df['_date_dt'].max().strftime('%Y-%m-%d')})")

            # ─── 1-1. 날짜별 평균 온도·습도 시계열 ───
            st.subheader("1. 날짜별 평균 온도·습도 시계열")

            if "temperature" in env_df.columns and "humidity" in env_df.columns:
                daily = env_df.groupby("_date_dt").agg(
                    temperature=("temperature","mean"),
                    humidity=("humidity","mean")
                ).reset_index()
                daily.columns = ["날짜","온도","습도"]
                daily["날짜"] = pd.to_datetime(daily["날짜"])

                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                ax1.plot(daily["날짜"], daily["온도"], color="#e74c3c", linewidth=2,
                         marker="o", markersize=4, label="평균 온도 (°C)")
                ax1.set_xlabel("날짜", fontsize=11)
                ax1.set_ylabel("온도 (°C)", color="#e74c3c", fontsize=11)
                ax1.tick_params(axis="y", labelcolor="#e74c3c")
                ax1.set_title("그래프 1. 날짜별 평균 온도·습도 시계열",
                              fontsize=13, pad=12)
                ax2 = ax1.twinx()
                ax2.plot(daily["날짜"], daily["습도"], color="#3498db", linewidth=2,
                         marker="s", markersize=4, label="평균 습도 (%)")
                ax2.set_ylabel("습도 (%)", color="#3498db", fontsize=11)
                ax2.tick_params(axis="y", labelcolor="#3498db")
                fig.autofmt_xdate(rotation=45)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                # 통계 정보
                t = env_df["temperature"].dropna()
                h = env_df["humidity"].dropna()
                col_s, col_i = st.columns(2)
                col_s.metric("전체 평균 온도", f"{t.mean():.2f} °C",
                             delta=f"범위: {t.min():.1f}~{t.max():.1f} °C")
                col_s.metric("전체 평균 습도", f"{h.mean():.1f} %",
                             delta=f"범위: {h.min():.1f}~{h.max():.1f} %")
                cv_t = abs(t.std()/t.mean()*100) if t.mean() != 0 else 0
                cv_h = abs(h.std()/h.mean()*100) if h.mean() != 0 else 0
                col_i.metric("온도 CV (변동계수)", f"{cv_t:.2f} %",
                             help="CV = (표준편차 ÷ 평균) × 100. 값이 작을수록 데이터가 평균 근처에 모여 있음.")
                col_i.metric("습도 CV (변동계수)", f"{cv_h:.2f} %",
                             help="CV = (표준편차 ÷ 평균) × 100.")
            else:
                st.warning("온도 또는 습도 열을 찾을 수 없습니다.")

            st.markdown("""
            **[캡션]** 온도는 여름철(7~8월)에 25~28°C까지 올라가고 가을로 가며 서서히 내려간다.
            습도는 60~70% 구간을 대체로 유지하며 8월 하순~9월 초에 약간 낮아진다.
            """)

            # ─── 1-2. 시간대별 평균 온도·습도 ───
            st.divider()
            st.subheader("2. 시간대별 평균 온도·습도 (24개 시간대)")

            if "temperature" in env_df.columns and "humidity" in env_df.columns:
                hourly = env_df.groupby("_hour").agg(
                    temperature=("temperature","mean"),
                    humidity=("humidity","mean")
                ).reset_index()

                max_temp_h = hourly.loc[hourly["temperature"].idxmax(), "_hour"]
                min_temp_h = hourly.loc[hourly["temperature"].idxmin(), "_hour"]
                max_hum_h  = hourly.loc[hourly["humidity"].idxmax(), "_hour"]
                min_hum_h  = hourly.loc[hourly["humidity"].idxmin(), "_hour"]
                day_range   = hourly["temperature"].max() - hourly["temperature"].min()

                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                bars = ax1.bar(hourly["_hour"], hourly["temperature"],
                               color="#e74c3c", alpha=0.75, width=0.6, label="평균 온도 (°C)")
                ax1.set_xlabel("시간대 (시)", fontsize=11)
                ax1.set_ylabel("온도 (°C)", color="#e74c3c", fontsize=11)
                ax1.set_xticks(range(0, 24))
                ax1.tick_params(axis="y", labelcolor="#e74c3c")
                ax1.set_title("그래프 2. 시간대별 평균 온도·습도",
                              fontsize=13, pad=12)
                ax2 = ax1.twinx()
                ax2.plot(hourly["_hour"], hourly["humidity"], color="#3498db",
                         marker="o", linewidth=2.5, markersize=6, label="평균 습도 (%)")
                ax2.set_ylabel("습도 (%)", color="#3498db", fontsize=11)
                ax2.tick_params(axis="y", labelcolor="#3498db")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown(
                    f"**최고 온도 시간대**: 오전 {int(max_temp_h)}시 ({hourly['temperature'].max():.2f}°C)  "
                    f"**최저 온도 시간대**: 오전 {int(min_temp_h)}시 ({hourly['temperature'].min():.2f}°C)  "
                    f"**일교차**: {day_range:.2f}°C"
                )
                st.markdown(
                    f"**최고 습도 시간대**: 오전 {int(max_hum_h)}시 ({hourly['humidity'].max():.1f}%)  "
                    f"**최저 습도 시간대**: 오전 {int(min_hum_h)}시 ({hourly['humidity'].min():.1f}%)"
                )
                st.caption("온도(막대, 빨강)는 오전 10시에 가장 높고 새벽 6시에 가장 낮다. 습도(선, 파랑)는 새벽 4시에 최고, 오전 10시에 최저로 온도 패턴과 반대 방향이다.")

            # ─── 1-3. 환경 기초 통계 ───
            st.divider()
            st.subheader("3. 환경 변수 기초 통계")

            st.caption("평균: 데이터의 중심값 / 표준편차: 값들이 평균에서 얼마나 퍼져 있는지 / CV(변동계수): (표준편차÷평균)×100, 단위가 다른 값끼리 비교 가능 / 최소·최대: 데이터 범위.")

            env_stat_cols = []
            labels = {"temperature":"온도 (°C)","humidity":"습도 (%)","ec":"EC","ph":"pH"}
            for key, label in labels.items():
                if key in env_df.columns:
                    s = env_df[key].dropna()
                    if len(s) == 0:
                        continue
                    mean_v  = s.mean()
                    std_v   = s.std()
                    min_v   = s.min()
                    max_v   = s.max()
                    cv_v    = abs(std_v/mean_v*100) if mean_v != 0 else 0
                    env_stat_cols.append({
                        "항목": label,
                        "평균": f"{mean_v:.2f}",
                        "중앙값": f"{s.median():.2f}",
                        "표준편차": f"{std_v:.2f}",
                        "CV (%)": f"{cv_v:.2f}",
                        "최소": f"{min_v:.2f}",
                        "최대": f"{max_v:.2f}",
                    })

            if env_stat_cols:
                st.dataframe(pd.DataFrame(env_stat_cols), use_container_width=True, hide_index=True)

            # ─── 1-4. 전반부 vs 후반부 비교 ───
            st.divider()
            st.subheader("4. 전반부(6/16~8/18) vs 후반부(8/19~10/21) 비교")

            st.caption("측정 기간 중간(약 8월 19일)을 기준으로 나누어 평균과 분산을 비교. 분산비 = 후반부 분산 ÷ 전반부 분산. 1보다 크면 후반부에 변동이 커졌다는 뜻.")

            if len(env_df) > 0:
                mid_date = env_df["_date_dt"].min() + (env_df["_date_dt"].max() - env_df["_date_dt"].min()) / 2
                first_half  = env_df[env_df["_date_dt"] <= mid_date]
                second_half = env_df[env_df["_date_dt"] >  mid_date]

                comp_data = []
                for key, label in labels.items():
                    if key in env_df.columns:
                        f_s = first_half[key].dropna()
                        s_s = second_half[key].dropna()
                        if len(f_s) == 0 or len(s_s) == 0:
                            continue
                        var_ratio = s_s.var()/f_s.var() if f_s.var() > 0 else float("nan")
                        comp_data.append({
                            "항목": label,
                            f"전반부 평균 (n={len(f_s)})": f"{f_s.mean():.2f}",
                            f"후반부 평균 (n={len(s_s)})": f"{s_s.mean():.2f}",
                            "평균 차이": f"{s_s.mean()-f_s.mean():+.2f}",
                            "분산비 (후반/전반)": f"{var_ratio:.2f}",
                            "변화 해석": "증가" if s_s.mean() > f_s.mean() else "감소",
                        })

                if comp_data:
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

                st.markdown("""
                - **온도**: 후반부 평균이 약 0.95°C 낮고, 분산은 약 2.08배로 커졌다. 계절이 가을로 넘어가면서 온도가 내려가고 변동도 커진 흔적이다.
                - **습도**: 평균과 분산 모두 거의 변하지 않아, 습도 환경은 재배 기간 내내 비교적 일정했다.
                - **EC**: 후반부로 갈수록 평균 약 0.33 상승, 분산도 약 1.40배 증가. EC가 점점 높아지는 방향이 분명하다.
                - **pH**: 후반부 평균이 약 0.69 낮고 분산은 오히려 약 0.45배로 줄었다. pH는 알칼리성에서 약산성으로 이동했다.
                """)

            # ─── 1-5. EC·pH 변화 구간 비교 ───
            st.divider()
            st.subheader("5. EC·pH 변화 구간 비교 (6월 초 vs 8월 하순)")

            st.caption("EC와 pH가 재배 기간 동안 어떤 방향으로 변했는지 확인하기 위해 6월 초(6/16~6/25)와 8월 하순(8/20~9/5)의 평균을 비교했다.")

            if "ec" in env_df.columns and "ph" in env_df.columns:
                # 기간을 day 단위로 변환
                env_df_c = env_df.copy()
                env_df_c["_date_only"] = env_df_c["_date_dt"].dt.date

                early_mask = (env_df_c["_date_only"] >= pd.Timestamp("2026-06-16").date()) &                              (env_df_c["_date_only"] <= pd.Timestamp("2026-06-25").date())
                late_mask  = (env_df_c["_date_only"] >= pd.Timestamp("2026-08-20").date()) &                              (env_df_c["_date_only"] <= pd.Timestamp("2026-09-05").date())

                if env_df_c[early_mask]["ec"].notna().any() and env_df_c[late_mask]["ec"].notna().any():
                    ec_early = env_df_c[early_mask]["ec"].dropna().mean()
                    ec_late  = env_df_c[late_mask]["ec"].dropna().mean()
                    ec_change = (ec_late - ec_early) / ec_early * 100 if ec_early != 0 else 0

                    ph_early = env_df_c[early_mask]["ph"].dropna().mean()
                    ph_late  = env_df_c[late_mask]["ph"].dropna().mean()
                    ph_change = ph_late - ph_early

                    ec_data = [
                        {"구간": "6월 초 (6/16~6/25)", "EC 평균": f"{ec_early:.2f}", "pH 평균": f"{ph_early:.2f}"},
                        {"구간": "8월 하순 (8/20~9/5)", "EC 평균": f"{ec_late:.2f}", "pH 평균": f"{ph_late:.2f}"},
                        {"구간": "변화량", "EC 평균": f"{ec_late-ec_early:+.2f} ({ec_change:+.0f}%)", "pH 평균": f"{ph_change:+.2f} pH"},
                    ]
                    st.dataframe(pd.DataFrame(ec_data), use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    - **EC**: 6월 초 {ec_early:.2f} → 8월 하순 {ec_late:.2f}로 약 **{abs(ec_change):.0f}% 상승**. 양분 농도 증가를 의미.
                    - **pH**: 6월 초 {ph_early:.2f} → 8월 하순 {ph_late:.2f}로 약 **{abs(ph_change):.2f}pH 단위 하락**. 알칼리성 → 약산성으로 이동.
                    - 이러한 환경 변화는 식물의 양분 흡수나 생리 상태에 영향을 줄 수 있는 수준의 변화다.
                    """)

            # ─── 1-6. IQR 이상치 확인 ───
            st.divider()
            st.subheader("6. 온도·습도 IQR 이상치 확인")

            st.caption("IQR(사분위범위) = Q3(75% 지점) - Q1(25% 지점). 이상치 경계: Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과. 이 연구에서는 이상치를 삭제하지 않고 그대로 두었다.")

            outlier_info = []
            for key, label in [("temperature","온도"),("humidity","습도")]:
                if key in env_df.columns:
                    s = env_df[key].dropna()
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = env_df[(env_df[key] < lower) | (env_df[key] > upper)]
                    outlier_info.append({
                        "항목": label,
                        "Q1": f"{q1:.2f}",
                        "Q3": f"{q3:.2f}",
                        "IQR": f"{iqr:.2f}",
                        "이상치 하한": f"{lower:.2f}",
                        "이상치 상한": f"{upper:.2f}",
                        "이상치 개수": f"{len(outliers)}개",
                    })

            if outlier_info:
                st.dataframe(pd.DataFrame(outlier_info), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # 탭 2: 개체별 생육 결과
    # ══════════════════════════════════════════
    with tab_growth:
        st.header("🌱  개체별 생육 결과 데이터")

        if growth_df is None or growth_df.shape[0] == 0:
            st.info("생육 결과 CSV를 업로드하세요.")
            st.markdown("""
            **예시 (송도고 실제 열 이름):**
```
개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)
1, 18, 17, 7, 0.19, 0.14, 0.05
2, 32, 31, 126, 0.98, 0.65, 0.33
```
            💡 **생중량(g) 열이 없는 학교**: 지상부질량(g) + 지하부질량(g)으로 자동 계산합니다.
            💡 개체번호는 숫자(1,2,...) 또는 글자(A1, A2,...) 모두 가능합니다.
            """)
        else:
            n_growth = len(growth_df)
            st.success(f"✅ {n_growth}개체 로드 완료")

            # ─── 2-1. 전체 생육 데이터 테이블 ───
            st.subheader("1. 전체 생육 데이터 (81개체)")
            st.dataframe(growth_df, use_container_width=True, hide_index=True)

            # ─── 2-2. 생육 기초 통계 ───
            st.divider()
            st.subheader("2. 생육 항목 기초 통계")

            st.caption("평균: 모든 값을 더해 개수로 나눈 값 / 중앙값: 크기순으로 나열했을 때 한가운데 값 / 표준편차: 값들이 평균에서 얼마나 넓게 퍼져 있는지 / CV(변동계수): (표준편차÷평균)×100, 단위가 다른 항목끼리 비교 가능 / 최소·최대: 데이터 범위 / 최대/최소 배율: 가장 큰 값과 가장 작은 값의 비율.")

            growth_stat = []
            stat_configs = [
                ("잎 수",          "잎 수 (장)",   "잎 수"),
                ("지상부 길이",   "지상부 길이 (mm)", "지상부 길이"),
                ("지하부 길이",   "지하부 길이 (mm)", "지하부 길이"),
                ("생중량",        "생중량 (g)",   "생중량"),
            ]
            for col_disp, table_label, col_key in stat_configs:
                if col_key in growth_df.columns:
                    s = growth_df[col_key].dropna()
                    mean_v  = s.mean()
                    median_v= s.median()
                    std_v   = s.std()
                    min_v   = s.min()
                    max_v   = s.max()
                    cv_v    = abs(std_v/mean_v*100) if mean_v != 0 else 0
                    ratio_v = max_v/min_v if min_v > 0 else float("nan")
                    growth_stat.append({
                        "항목": table_label,
                        "평균": f"{mean_v:.2f}",
                        "중앙값": f"{median_v:.2f}",
                        "분산": f"{std_v**2:.2f}",
                        "표준편차": f"{std_v:.2f}",
                        "CV (%)": f"{cv_v:.2f}",
                        "최소": f"{min_v:.2f}",
                        "최대": f"{max_v:.2f}",
                        "최대/최소": f"{ratio_v:.2f}",
                    })

            if growth_stat:
                st.dataframe(pd.DataFrame(growth_stat), use_container_width=True, hide_index=True)

            st.markdown("""
            - 생중량의 CV가 **89.42%**로 가장 높고, 최대/최소 배율도 **250.25배**로 가장 크다.
            - 81개체의 생중량은 평균 2.48g이지만, 중앙값 1.76g보다 평균이 훨씬 큰 것은 일부 무거운 개체들이 평균을 끌어올렸기 때문이다.
            - 지상부 길이는 CV 49.91%, 최대/최소 14.70배로 네 항목 중 개체 간 차이가 가장 작았다.
            """)

            # ─── 2-3. 생중량 분포 히스토그램 ───
            st.divider()
            st.subheader("3. 생중량(g) 분포 히스토그램")

            if "생중량" in growth_df.columns:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                fw_values = growth_df["생중량"].dropna()
                n_bins = max(10, int(np.sqrt(len(fw_values))))
                ax.hist(fw_values, bins=n_bins, color="#2ecc71", edgecolor="white",
                        alpha=0.85, linewidth=1.2)
                median_val = fw_values.median()
                mean_val   = fw_values.mean()
                ax.axvline(median_val, color="red", linestyle="--", linewidth=2,
                           label=f"중앙값: {median_val:.2f}g")
                ax.axvline(mean_val, color="orange", linestyle="--", linewidth=2,
                           label=f"평균: {mean_val:.2f}g")
                ax.set_xlabel("생중량 (g)", fontsize=11)
                ax.set_ylabel("개체 수", fontsize=11)
                ax.set_title("그래프 4. 생중량(g) 분포 히스토그램", fontsize=13, pad=12)
                ax.legend(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                st.caption("81개체의 생중량을 1g 간격 구간으로 나눈 히스토그램. 0~2g 구간에 가장 많은 개체가 몰려 있고, 3g 이상은 급격히 줄어든다. 붉은 점선은 중앙값 1.76g. 분포가 왼쪽에 몰리고 오른쪽으로 꼬리가 긴 비대칭 모양이다.")

            # ─── 2-4. 이상치 분석 ───
            st.divider()
            st.subheader("4. IQR 기준 이상치 분석")

            st.caption("IQR 기준: Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과인 값. 이상치는 삭제하지 않고 그대로 분석에 포함했다.")

            outlier_rows = []
            for col_key, label in [("잎 수","잎 수"),("지상부 길이","지상부 길이"),
                                     ("지하부 길이","지하부 길이"),("생중량","생중량")]:
                if col_key in growth_df.columns:
                    s = growth_df[col_key].dropna()
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = growth_df[(growth_df[col_key] < lower) | (growth_df[col_key] > upper)]
                    if len(outliers) > 0:
                        ids  = ", ".join([str(int(x)) for x in outliers["개체번호"]])
                        vals = ", ".join([f"{v:.2f}" for v in outliers[col_key]])
                    else:
                        ids  = "-"
                        vals = "-"
                    outlier_rows.append({
                        "항목": label,
                        "이상치 개수": f"{len(outliers)}개",
                        "이상치 개체번호": ids,
                        "이상치 값": vals,
                        "이상치 기준": f"[{lower:.2f}, {upper:.2f}] 밖",
                    })

            if outlier_rows:
                st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)

            st.markdown("""
            - IQR 기준으로 생중량이 큰 쪽(7.14g 초과)에 **3개체(13, 20, 21번)**가 이상치로 확인되었다. 이들은 특히 잘 자란 개체들이다.
            - 생중량이 극단적으로 작은 쪽(0.04g인 36번)은 IQR 하한(−2.94g)보다 크므로 이상치로 분류되지 않았다.
            - 24번 개체는 잎 수(102)와 지상부 길이(126) 모두에서 이상치로 나타났다.
            """)

            # ─── 2-5. 최대·최소 개체 ───
            st.divider()
            st.subheader("5. 생중량 최대·최소 개체")

            if "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
                max_row = growth_df.loc[growth_df["생중량"].idxmax()]
                min_row = growth_df.loc[growth_df["생중량"].idxmin()]

                col_max, col_min = st.columns(2)
                col_max.metric("생중량 최대 개체",
                               value=f"{max_row['생중량']:.2f}g",
                               delta=f"{int(max_row['개체번호'])}번  |  잎 수: {int(max_row['잎 수']) if '잎 수' in max_row else '-'}장")
                col_min.metric("생중량 최소 개체",
                               value=f"{min_row['생중량']:.2f}g",
                               delta=f"{int(min_row['개체번호'])}번  |  잎 수: {int(min_row['잎 수']) if '잎 수' in min_row else '-'}장")

                st.markdown(f"""
                **최대 개체(21번)와 최소 개체(36번)의 생중량 차이는 250배이다.** 같은 스마트팜에서 재배했음에도 이 정도로 차이가 났다.
                """)

            # ─── 2-6. 상관계수 ───
            st.divider()
            st.subheader("6. 생중량과 생육 항목 간 Pearson 상관계수")

            st.caption("Pearson 상관계수(r): -1에서 1 사이 값. 1에 가까울수록 한 변수가 증가할 때 다른 변수도 함께 증가하는 경향이 강함. p값: '관계가 없다는 가정에서는 관찰되기 어려운 결과'인지 판단하는 근거. p < 0.05이면 통계적으로 유의한 관계.")

            if all(c in growth_df.columns for c in ["생중량","잎 수","지상부 길이","지하부 길이"]):
                corr_rows = []
                for col, label in [("잎 수","잎 수"),("지상부 길이","지상부 길이"),("지하부 길이","지하부 길이")]:
                    r_val, p_val = stats.pearsonr(growth_df["생중량"], growth_df[col])
                    strength = "강한 양의 상관" if r_val > 0.7 else ("중간 상관" if r_val > 0.4 else "약한 상관")
                    sig = "유의함 (p<0.001)" if p_val < 0.001 else ("유의함 (p<0.05)" if p_val < 0.05 else "유의하지 않음")
                    corr_rows.append({
                        "생육 항목": label,
                        "Pearson r": f"{r_val:.4f}",
                        "p값": f"{p_val:.6f}",
                        "상관 강도": strength,
                        "통계적 유의성": sig,
                        "해석": "잎 수↑ → 생중량↑ 경향" if r_val > 0 else "관계 모호",
                    })

                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

                top_corr_col = max([("잎 수","잎 수"),("지상부 길이","지상부 길이"),("지하부 길이","지하부 길이")],
                                   key=lambda x: abs(stats.pearsonr(growth_df["생중량"], growth_df[x[0]])[0]))
                top_r = stats.pearsonr(growth_df["생중량"], growth_df[top_corr_col[0]])[0]
                st.markdown(f"""
                **생중량과 가장 강한 상관을 보인 항목은 {top_corr_col[1]}(r = {top_r:.4f}, p < 0.001)였다.**
                """)

    # ══════════════════════════════════════════
    # 탭 3: 분석 결과 · 가설 검증 · 결론
    # ══════════════════════════════════════════
    with tab_analysis:
        st.header("📊  분석 결과 · 가설 검증 · 예측모델 · 결론")

        if growth_df is None or growth_df.shape[0] == 0:
            st.info("생육 결과 CSV를 업로드하면 모든 분석 결과가 표시됩니다.")
        else:
            # ─── 3-1. 연구 가설 ───
            st.subheader("연구 가설")
            st.info("가설: **잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다.**")

            # ─── 3-2. Top·Bottom 그룹 비교 ───
            st.divider()
            st.subheader("1. Top 25% vs Bottom 25% 그룹 비교")

            st.caption("생중량 기준으로 상위 25%(20개체)와 하위 25%(20개체)로 나눈다. Welch t검정: 두 그룹의 평균이 같은지 검정하는 방법. p값이 작을수록 '두 그룹 평균이 같다'는 가정이 기각됨.")

            if "생중량" in growth_df.columns:
                fw = growth_df["생중량"].dropna()
                n_each = int(np.ceil(len(fw) * 0.25))

                # 상위·하위 경계값
                top_threshold    = fw.nlargest(n_each).min()
                bottom_threshold = fw.nsmallest(n_each).max()

                top_group    = growth_df[growth_df["생중량"] >= top_threshold].head(n_each).copy()
                bottom_group = growth_df[growth_df["생중량"] <= bottom_threshold].head(n_each).copy()

                st.markdown(f"""
                | 항목 | 내용 |
                | --- | --- |
                | 전체 개체 수 | {len(growth_df)}개체 |
                | 그룹 비율 | 상위 25% / 하위 25% |
                | 각 그룹 개체 수 | {n_each}개체 (81 × 0.25 = 20.25 → 반올림) |
                | 하위 경계값 | {bottom_threshold:.2f}g |
                | 상위 경계값 | {top_threshold:.2f}g |
                | 중간 그룹 | {len(growth_df) - 2*n_each}개체 (비교에서 제외) |
                """)

                # 그룹 비교 테이블
                st.subheader("그룹별 통계량 비교")

                group_data = []
                for col, label in [("잎 수","잎 수 (장)"),("지상부 길이","지상부 길이 (mm)"),
                                    ("지하부 길이","지하부 길이 (mm)")]:
                    if col in growth_df.columns:
                        t_mean = top_group[col].mean()
                        b_mean = bottom_group[col].mean()
                        t_std  = top_group[col].std()
                        b_std  = bottom_group[col].std()
                        diff   = t_mean - b_mean
                        ratio  = t_mean / b_mean if b_mean != 0 else float("nan")
                        t_stat, p_val = stats.ttest_ind(top_group[col], bottom_group[col], equal_var=False)
                        sig_str = "유의함 (p<0.001)" if p_val < 0.001 else ("유의함 (p<0.05)" if p_val < 0.05 else "유의하지 않음")
                        group_data.append({
                            "항목": label,
                            "Top 평균": f"{t_mean:.2f}",
                            "Bottom 평균": f"{b_mean:.2f}",
                            "평균 차이 (Top-Bottom)": f"{diff:.2f}",
                            "비율 (Top/Bottom)": f"{ratio:.2f}",
                            "Top 표준편차": f"{t_std:.2f}",
                            "Bottom 표준편차": f"{b_std:.2f}",
                            "Welch t검정 p값": f"{p_val:.6f}",
                            "유의성": sig_str,
                        })

                if group_data:
                    st.dataframe(pd.DataFrame(group_data), use_container_width=True, hide_index=True)

                st.markdown(f"""
                **가장 큰 차이**: 통계적으로 가장 유의한(p가 가장 작은) 차이는 **잎 수**(p = {group_data[0]['Welch t검정 p값']} 수준). 
                평균 차이의 절대적 크기는 **지하부 길이**(Top − Bottom = 약 43.52mm)가 가장 크다.
                잎 수의 Top/Bottom 비율은 약 **6.36배**로 네 항목 중 가장 큰 배율 차이를 보였다.
                """)

                # ─── 지상부÷지하부 비율 ───
                st.divider()
                st.subheader("지상부 ÷ 지하부 비율 비교")

                if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                    top_ratio    = (top_group["지상부 길이"] / top_group["지하부 길이"]).mean()
                    bottom_ratio = (bottom_group["지상부 길이"] / bottom_group["지하부 길이"]).mean()

                    col_tr, col_br = st.columns(2)
                    col_tr.metric("Top 그룹 지상부/지하부 비율", f"{top_ratio:.3f}",
                                  help="1보다 작으면 지상부보다 지하부가 더 길다는 의미")
                    col_br.metric("Bottom 그룹 지상부/지하부 비율", f"{bottom_ratio:.3f}",
                                  help="1보다 크면 지상부가 지하부보다 더 길다는 의미")

                    t_ratio, p_ratio = stats.ttest_ind(
                        top_group["지상부 길이"]/top_group["지하부 길이"],
                        bottom_group["지상부 길이"]/bottom_group["지하부 길이"],
                        equal_var=False
                    )

                    st.markdown(f"""
                    - **Top 그룹의 지상부÷지하부 비율 = {top_ratio:.3f}**: 잘 자란 개체는 지상부보다 지하부가 상대적으로 더 길었다(비율이 1보다 작음).
                    - **Bottom 그룹의 지상부÷지하부 비율 = {bottom_ratio:.3f}**: 덜 자란 개체는 지상부와 지하부가 비슷하거나 지상부가 상대적으로 더 긴 편이다.
                    - Welch t검정 **p = {p_ratio:.4f}**로, "두 그룹의 비율이 같다"는 가정에서는 관찰되기 어려운 결과다. 다만 비율 차이 자체는 약 0.76으로, 절대적 크기가 아주 크다고 보기는 어렵다.
                    - 즉 **잘 자란 개체일수록 지상부보다 지하부가 더 발달한 경향**을 보였다.
                    """)

            # ─── 3-3. 그룹 비교 결과 해석 ───
            st.divider()
            st.subheader("2. 그룹 비교 결과 해석")

            if all(c in growth_df.columns for c in ["생중량","잎 수","지상부 길이","지하부 길이"]):
                # 실제 계산된 값 사용
                fw_vals = growth_df["생중량"].dropna()
                n_each2 = int(np.ceil(len(fw_vals) * 0.25))
                top_t = fw_vals.nlargest(n_each2).min()
                bot_t = fw_vals.nsmallest(n_each2).max()
                top_g = growth_df[growth_df["생중량"] >= top_t].head(n_each2)
                bot_g = growth_df[growth_df["생중량"] <= bot_t].head(n_each2)

                t_leaves, p_leaves  = stats.ttest_ind(top_g["잎 수"], bot_g["잎 수"], equal_var=False)
                t_fw,    p_fw       = stats.ttest_ind(top_g["생중량"], bot_g["생중량"], equal_var=False)
                t_root,  p_root     = stats.ttest_ind(top_g["지하부 길이"], bot_g["지하부 길이"], equal_var=False)
                t_shoot, p_shoot    = stats.ttest_ind(top_g["지상부 길이"], bot_g["지상부 길이"], equal_var=False)

                top_avg_leaves = top_g["잎 수"].mean()
                bot_avg_leaves = bot_g["잎 수"].mean()
                top_avg_fw     = top_g["생중량"].mean()
                bot_avg_fw     = bot_g["생중량"].mean()
                top_avg_root   = top_g["지하부 길이"].mean()
                bot_avg_root   = bot_g["지하부 길이"].mean()
                top_avg_shoot  = top_g["지상부 길이"].mean()
                bot_avg_shoot  = bot_g["지상부 길이"].mean()

                st.markdown(f"""
                ### 잎 수 차이
                - **Top 평균 {top_avg_leaves:.2f}장 vs Bottom 평균 {bot_avg_leaves:.2f}장** → 약 {top_avg_leaves/bot_avg_leaves:.1f}배 차이.
                - 앞서 확인한 잎 수−생중량 상관(r = 0.886)과 일관되는 결과로, 잎 수가 많은 개체가 생중량도 큰 경향이 그룹 비교에서도 뚜렷하게 나타났다.
                - Welch t검정 **p = {p_leaves:.2e}**, t = {t_leaves:.2f}.
                - 단, 이 또한 인과관계는 아니다.

                ### 생중량 차이 (가설 직접 검증)
                - **Top 평균 {top_avg_fw:.2f}g vs Bottom 평균 {bot_avg_fw:.2f}g** → 약 {top_avg_fw/bot_avg_fw:.1f}배 차이.
                - 가설 "잎 수 상위 25%는 하위 25%보다 생중량이 클 것이다"는 **데이터로 지지되었다**.
                - Welch t검정 **p = {p_fw:.2e}**.

                ### 지하부 길이 차이
                - **Top 평균 {top_avg_root:.2f}mm vs Bottom 평균 {bot_avg_root:.2f}mm**.
                - **p = {p_root:.2e}**로 유의한 차이를 보였다.
                - 잘 자란 개체일수록 지하부가 더 길었다.

                ### 지상부 길이 차이
                - **Top 평균 {top_avg_shoot:.2f}mm vs Bottom 평균 {bot_avg_shoot:.2f}mm**.
                - **p = {p_shoot:.2e}**.
                """)

            # ─── 3-4. 생중량 예측모델 ───
            st.divider()
            st.subheader("3. 생중량 예측모델 (선형회귀)")

            st.caption("선형회귀: 한 변수(예측 변수)로 다른 변수(반응 변수)를 예측하는 방법. R²: 모델이 데이터 변동을 설명하는 비율, 0~1 사이 값. 1에 가까울수록 설명력이 높음. MAE(평균절대오차): 예측값과 실제값 차이의 평균. 값이 작을수록 예측이 정확.")

            if all(c in growth_df.columns for c in ["생중량","잎 수"]):
                X1 = growth_df[["잎 수"]].values
                y  = growth_df["생중량"].values

                # 모델 1: 잎 수만
                slope1, intercept1, r1, p1, se1 = stats.linregress(growth_df["잎 수"], growth_df["생중량"])
                r2_1 = r1 ** 2
                pred1 = slope1 * growth_df["잎 수"] + intercept1
                mae1  = np.mean(np.abs(pred1 - y))

                # 모델 2: 잎 수 + 지상부 길이
                if "지상부 길이" in growth_df.columns:
                    X2 = growth_df[["잎 수","지상부 길이"]].values
                    sl2, int2, r2, p2, se2 = stats.linregress(
                        growth_df[["잎 수","지상부 길이"]].values, y
                    ) if False else (None, None, None, None, None)
                    # 다중회귀는 sklearn으로
                    lr2 = LinearRegression()
                    lr2.fit(X2, y)
                    pred2 = lr2.predict(X2)
                    r2_2  = r2_score(y, pred2)
                    mae2  = mean_absolute_error(y, pred2)
                    sl2_true  = lr2.coef_[0]
                    sl2_shoot = lr2.coef_[1]
                    int2_true = lr2.intercept_
                    p2_shoot  = None  # scipy로 개별 p값 계산 어려움, 주석으로 처리

                    # 각 계수별 p값 (별도 단순회귀)
                    _, _, _, p_shoot_sl, _ = stats.linregress(growth_df["지상부 길이"], growth_df["생중량"])
                    _, _, _, p_leaf_sl,   _ = stats.linregress(growth_df["잎 수"], growth_df["생중량"])

                    model2_available = True
                else:
                    model2_available = False

                # 모델 결과 표시
                st.subheader("모델 1: 잎 수만 사용한 단순 선형회귀")
                st.code(f"생중량(g) = {slope1:.4f} × 잎 수(장) + ({intercept1:.4f})")
                col_m1a, col_m1b, col_m1c = st.columns(3)
                col_m1a.metric("R² (결정계수)", f"{r2_1:.4f}",
                               help=f"잎 수로 생중량 변동의 약 {r2_1*100:.0f}% 설명 가능")
                col_m1b.metric("MAE (평균절대오차)", f"{mae1:.3f}g",
                               help="예측값과 실제값 차이의 평균 크기")
                col_m1c.metric("Pearson r", f"{r1:.4f}",
                               help=f"p < 0.001 (통계적으로 매우 유의)")

                if model2_available:
                    st.divider()
                    st.subheader("모델 2: 잎 수 + 지상부 길이를 사용한 다중 선형회귀")
                    st.code(f"생중량(g) = {sl2_true:.4f} × 잎 수(장) + {sl2_shoot:.4f} × 지상부 길이(mm) + ({int2_true:.4f})")

                    col_m2a, col_m2b, col_m2c = st.columns(3)
                    col_m2a.metric("R² (결정계수)", f"{r2_2:.4f}")
                    col_m2b.metric("MAE (평균절대오차)", f"{mae2:.3f}g")
                    col_m2c.metric("지상부 길이 p값 (개별)", f"{p_shoot_sl:.3f}",
                                   help="p > 0.05이면 이 데이터에서는 지상부 길이가 생중량 예측에 유의하게 기여하지 못함")

                    st.markdown(f"""
                    - R² 기준: 모델 1 = {r2_1:.4f} vs 모델 2 = {r2_2:.4f} → **실질적으로 비슷한 설명력**.
                    - 지상부 길이 계수는 통계적으로 유의하지 않음(p = {p_shoot_sl:.3f} > 0.05).
                    - 즉 이 데이터에서는 **지상부 길이를 추가해도 예측력 개선 효과가 확인되지 않았다**.
                    """)

                # ─── 5겹 교차검증 ───
                st.divider()
                st.subheader("5겹 교차검증 (모델 일반화 성능 평가)")

                st.caption("5겹 교차검증: 데이터를 5개로 나누어 4개로 학습·1개로 검증을 5번 반복. 모델이 학습 데이터에만 지나치게 맞춰지는 과적합을 확인하는 방법. shuffle=True: 데이터를 섞어서 나눔. random_state=42: 매번 같은 결과가 나오도록 고정.")

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2_1, cv_mae_1 = [], []
                for train_idx, test_idx in kf.split(X1):
                    m = LinearRegression()
                    m.fit(X1[train_idx], y[train_idx])
                    pred = m.predict(X1[test_idx])
                    cv_r2_1.append(r2_score(y[test_idx], pred))
                    cv_mae_1.append(mean_absolute_error(y[test_idx], pred))

                if model2_available:
                    cv_r2_2, cv_mae_2 = [], []
                    for train_idx, test_idx in kf.split(X2):
                        m2 = LinearRegression()
                        m2.fit(X2[train_idx], y[train_idx])
                        pred2_cv = m2.predict(X2[test_idx])
                        cv_r2_2.append(r2_score(y[test_idx], pred2_cv))
                        cv_mae_2.append(mean_absolute_error(y[test_idx], pred2_cv))

                    cv_data = pd.DataFrame([
                        {"모델": "모델 1 (잎 수만)", "CV R²": f"{np.mean(cv_r2_1):.4f}",
                         "CV MAE": f"{np.mean(cv_mae_1):.3f}g", "R² 차이 (CV-LR)": f"{np.mean(cv_r2_1)-r2_1:+.4f}"},
                        {"모델": "모델 2 (잎 수+지상부)", "CV R²": f"{np.mean(cv_r2_2):.4f}",
                         "CV MAE": f"{np.mean(cv_mae_2):.3f}g", "R² 차이 (CV-LR)": f"{np.mean(cv_r2_2)-r2_2:+.4f}"},
                    ])
                    st.dataframe(cv_data, use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    - 학습 데이터 기준 R²: 모델 1 = {r2_1:.4f} vs 모델 2 = {r2_2:.4f} → 비슷.
                    - **5겹 교차검증에서는 오히려 단순회귀(모델 1)가 더 안정적**이었다.
                      R² {np.mean(cv_r2_1):.4f} vs {np.mean(cv_r2_2):.4f}, MAE {np.mean(cv_mae_1):.3f}g vs {np.mean(cv_mae_2):.3f}g.
                    - 지상부 길이는 생중량 예측에 **통계적으로 유의한 기여를 하지 못했다**(p = {p_shoot_sl:.3f}).
                    """)
                else:
                    cv_data = pd.DataFrame([{"모델": "모델 1 (잎 수만)", "CV R²": f"{np.mean(cv_r2_1):.4f}",
                                              "CV MAE": f"{np.mean(cv_mae_1):.3f}g",
                                              "R² 차이 (CV-LR)": f"{np.mean(cv_r2_1)-r2_1:+.4f}"}])
                    st.dataframe(cv_data, use_container_width=True, hide_index=True)

                # ─── 예측 예시 ───
                st.divider()
                st.subheader("예측 예시: 잎 수 25장·60장일 때 생중량 예측값")

                st.caption("지상부 길이를 평균(39.4mm)으로 가정하고 예측값을 계산했다.")

                pred_25 = slope1 * 25 + intercept1
                pred_60 = slope1 * 60 + intercept1

                col_p1, col_p2 = st.columns(2)
                col_p1.metric("잎 수 25장 → 예측 생중량", f"{pred_25:.2f}g",
                              help=f"회귀식: {slope1:.4f}×25 + ({intercept1:.4f}) = {pred_25:.2f}g")
                col_p2.metric("잎 수 60장 → 예측 생중량", f"{pred_60:.2f}g",
                              help=f"회귀식: {slope1:.4f}×60 + ({intercept1:.4f}) = {pred_60:.2f}g")

                # 산점도
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(growth_df["잎 수"], growth_df["생중량"],
                           color="#3498db", alpha=0.6, s=45, edgecolors="white", linewidth=0.5,
                           label="81개체 실제값")
                x_range = np.linspace(growth_df["잎 수"].min()-5, growth_df["잎 수"].max()+5, 200)
                ax.plot(x_range, slope1*x_range + intercept1, color="#e74c3c", linewidth=2.5,
                        label=f"모델 1 회귀선 (R²={r2_1:.4f})")
                if model2_available:
                    shoot_mean = growth_df["지상부 길이"].mean()
                    line2 = np.array([slope1*x + sl2_shoot*shoot_mean + int2_true
                                       for x in x_range])
                    ax.plot(x_range, line2, color="#2ecc71", linewidth=2, linestyle="--",
                            label=f"모델 2 (지상부={shoot_mean:.1f}mm 가정, R²={r2_2:.4f})")
                ax.set_xlabel("잎 수 (장)", fontsize=11)
                ax.set_ylabel("생중량 (g)", fontsize=11)
                ax.set_title("그림 3. 잎 수 vs 생중량 산점도 및 회귀선", fontsize=13, pad=12)
                ax.legend(fontsize=10, loc="upper left")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                - 가로축은 잎 수, 세로축은 생중량이다. 점들이 대체로 우상향 방향에 모여 있어 잎 수가 많을수록 생중량이 증가하는 경향이 관찰된다.
                - 상관계수 r = 0.886으로 네 항목 중 생중량과 가장 강한 양의 상관을 보인다.
                - 다만 이것은 "잎 수가 많아서 무겁게 자랐다"는 인과관계가 아니라, 두 측정값이 함께 증가하는 경향이 있다는 통계적 관계이다.
                - 점들이 모델 1 예측선 주변에 비교적 모여 있으며, 생중량이 큰 개체에서는 예측선이 실제값보다 낮게 잡는 경향이 보인다.
                """)

            # ─── 3-5. 지상부 길이 vs 생중량 산점도 ───
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                st.divider()
                st.subheader("보충: 지상부 길이 vs 생중량 산점도")
                r_shoot, p_shoot2 = stats.pearsonr(growth_df["지상부 길이"], growth_df["생중량"])

                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.scatter(growth_df["지상부 길이"], growth_df["생중량"],
                           color="#9b59b6", alpha=0.6, s=45, edgecolors="white", linewidth=0.5)
                x_s = np.linspace(growth_df["지상부 길이"].min(), growth_df["지상부 길이"].max(), 100)
                sl_sh, int_sh, _, _, _ = stats.linregress(growth_df["지상부 길이"], growth_df["생중량"])
                ax.plot(x_s, sl_sh*x_s + int_sh, color="#e74c3c", linewidth=2,
                        label=f"추세선 (r={r_shoot:.3f})")
                ax.set_xlabel("지상부 길이 (mm)", fontsize=11)
                ax.set_ylabel("생중량 (g)", fontsize=11)
                ax.set_title("그림 4. 지상부 길이와 생중량의 산점도", fontsize=13, pad=12)
                ax.legend(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown(f"""
                가로축은 지상부 길이(mm), 세로축은 생중량(g)이다. 점들의 퍼짐이 넓고 경향선이 비교적 완만하여, 잎 수나 지하부 길이에 비해 **지상부 길이와 생중량의 관계는 약한 편**이다(상관계수 r = {r_shoot:.3f}).
                """)

            # ─── 3-6. 논의: 가설 검토 ───
            st.divider()
            st.subheader("4. 논의: 가설 검토")

            st.markdown("""
            ### 가설: 잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다.

            이 가설은 **데이터로 지지되었다**.

            - Top 25% 평균 생중량 5.93g vs Bottom 25% 0.37g → 약 **16배 차이**.
            - Top 25% 평균 잎 수 72.55장 vs Bottom 25% 11.40장 → 약 **6.4배 차이**.
            - Welch t검정 **p = 4.49×10⁻⁹**로, 두 그룹의 잎 수가 같다는 가정에서는 관찰되기 매우 어려운 결과다.
            - 잎 수와 생중량의 Pearson r = 0.8857(p < 0.001)도 같은 방향을 뒷받침한다.

            **다만**, 이것이 "잎 수가 많아서 생중량이 커진 것"이라는 인과관계를 증명하는 것은 아니다.
            잎 수와 생중량이 함께 커지는 경향은 확인되었지만, 그 배경은 따로 확인이 필요하다.
            """)

            # ─── 3-7. 환경 변화와 생육 차이 연결 가능성 ───
            st.divider()
            st.subheader("5. 환경 변화와 생육 차이의 연결 가능성")

            st.caption("환경 데이터(127일간 시계열)와 생육 데이터(한 시점 측정 추정)를 직접 연결하려면 생육 측정일이 언제인지 알아야 한다. 이 정보가 없으므로 가능성으로만 언급한다.")

            st.markdown("""
            - EC는 6월 초 0.49에서 8월 하순 1.42로 약 **188% 상승**했고, pH는 7.96에서 6.52로 약 **1.44pH 단위 하락**했다.
            - 이러한 환경 변화는 식물의 양분 흡수나 생리 상태에 영향을 줄 수 있는 수준의 변화다.
            - 생육 데이터는 한 시점(측정일 미확인)에 측정된 것으로 보이며, 환경 데이터는 127일간의 시계열이다.
            - 두 데이터를 직접 연결하려면 생육 측정일이 언제인지 알아야 환경 조건과 생육 결과를 짝지을 수 있는데, 이 정보가 없다.
            - 따라서 **환경 변화가 개체별 생육 차이를 일으켰다고 단정할 수 없고, 가능성으로만 언급**한다.
            """)

            # ─── 3-8. 실용적 제안 ───
            st.divider()
            st.subheader("6. 실용적 제안")

            st.markdown("""
            - 잎 수만 사용한 모델 1(R² = 0.7845)이 잎 수+지상부 길이를 사용한 모델 2(R² = 0.7893)와 실질적으로 비슷한 설명력을 보였고, 오히려 교차검증에서는 모델 1이 더 안정적이었다. 즉 **지상부 길이를 추가로 재는 것이 예측에는 큰 도움이 되지 않았다**.
            - 다만 R²가 0.78이라는 것은 여전히 **약 22%의 변동은 잎 수로 설명되지 않는다**는 뜻이므로, 잎 수만으로 생중량을 정확히 예측하는 도구라고 보기는 어렵다.
            - 이 모델은 이 데이터(**송도고 극지식물, 특정 시기, 특정 스마트팜 조건**)에서 얻은 것이다. 다른 품종, 다른 환경, 다른 시기에 그대로 적용된다고 단정할 수 없다.
            """)

            # ─── 3-9. 한계점 ───
            st.divider()
            st.subheader("7. 한계점")

            limits = [
                ("생육 측정일 부재", "생육결과데이터에 측정일(날짜) 열이 없다. 따라서 생육 측정이 정확히 언제 이루어졌는지 알 수 없고, 환경 시계열과 생육 결과를 직접 연결할 수 없다. 이 점은 이 연구에서 가장 중요한 한계 중 하나다."),
                ("개체 위치 정보 부재", "같은 스마트팜 안에서도 개체마다 빛, 통풍, 수분 공급 등이 다를 수 있으나, 이 데이터에는 각 개체의 위치 정보가 없다. 위치 효과를 분석할 수 없다."),
                ("표본 크기", "생육 데이터가 81개체로, 그룹 비교(각 20개체)나 회귀분석은 가능하지만 표본이 아주 크다고 보기는 어렵다. 특히 이상치로 잡힌 소수 개체들의 영향이 결과를 왜곡할 가능성도 배제할 수 없다."),
                ("단일 시점 생육 측정", "생육 데이터가 한 시점 측정인지, 여러 시점 측정인지 확인할 수 없다. 만약 한 시점 측정이라면 성장 과정의 변화를 볼 수 없고, 여러 시점 측정이라면 측정 시점 차이가 결과에 영향을 주었을 수 있다."),
                ("EC·pH 결측", "환경 데이터의 EC와 pH는 각각 194개 결측이 있어 전체 2,461행 중 2,267행만 유효하다. 결측이 무작위가 아니라면 통계 결과에 편향이 있을 수 있다."),
                ("인과관계 아님", "모든 통계 결과(상관, 그룹 차이, 회귀계수)는 변수들 사이의 관계나 차이를 보여줄 뿐, 인과관계를 증명하지 않는다. '잎 수가 많아서 생중량이 크다'고 말할 수 없으며, '잎 수가 많은 개체가 생중량이 큰 경향이 있다'고만 말할 수 있다."),
                ("외부 일반화 불가", "이 분석은 송도고등학교의 특정 스마트팜, 특정 품종(극지식물), 특정 시기(2026년 6~10월) 조건에서 얻은 데이터로 수행되었다. 다른 학교, 다른 품종, 다른 환경에서 같은 결과가 나온다고 단정할 수 없다."),
            ]

            for title, content in limits:
                st.markdown(f"**{title}**")
                st.write(content)

            # ─── 3-10. 결론 ───
            st.divider()
            st.subheader("8. 결론")

            st.markdown("""
            같은 스마트팜에서 재배한 극지식물 81개체를 분석한 결과, 개체별 생육 차이, 특히 생중량 차이는 매우 컸다.

            - 생중량은 최소 0.04g에서 최대 10.01g까지 **250배 차이**가 났고, CV는 **89.42%**로 네 측정 항목 중 가장 높았다.
            - 평균 2.48g, 중앙값 1.76g으로, 일부 무거운 개체가 평균을 끌어올리는 오른쪽 꼬리 분포였다.

            생장 환경은 온도와 습도가 비교적 안정적이었고(하루 일교차 1.52°C, CV 10~12%), EC와 pH는 재배 기간 내내 뚜렷한 방향 변화를 보였다.

            - EC는 6월 초 0.49에서 8월 하순 1.42로 약 **188% 상승**.
            - pH는 7.96에서 6.52로 약 **1.44pH 단위 하락**.
            - 이러한 변화가 생육 차이와 관련되었을 가능성은 있으나, 생육 측정일이 없어 직접적 연결은 확인할 수 없었다.

            가설 "잎 수 상위 25%는 하위 25%보다 생중량이 클 것이다"는 **지지되었다**.

            - Top 그룹 평균 잎 수 72.55장 vs Bottom 그룹 11.40장(p = 4.49×10⁻⁹).
            - 생중량 5.93g vs 0.37g.
            - 잎 수는 생중량과 가장 강한 상관을 보였다(r = 0.886).
            - 지하부 길이도 유의한 차이(p = 6.12×10⁻⁷)를 보였고, 잘 자란 개체는 지상부보다 지하부가 상대적으로 더 긴 경향(비율 0.633 vs 1.389, p = 0.0317)이 있었다.

            예측모델은 잎 수만 사용한 단순회귀(R² = 0.7845, MAE = 0.772g)가 잎 수+지상부 길이를 사용한 다중회귀(R² = 0.7893, MAE = 0.761g)와 실질적으로 비슷한 성능을 보였고, 5겹 교차검증에서는 오히려 단순회귀가 더 안정적이었다(R² 0.7186 vs 0.7075). 지상부 길이는 생중량 예측에 통계적으로 유의한 기여를 하지 못했다(p = 0.190).

            **그러나** 이 결과는 표본 81개체, 단일 스마트팜·단일 시기 데이터에서 얻은 것으로, 생육 측정일과 개체 위치 정보가 없다는 한계가 있다. 통계적 관계가 확인되었어도 인과관계는 아니며, 다른 환경이나 다른 품종에 그대로 적용할 수 있다고 단정할 수 없다.

            다음 재배부터는 측정일과 위치 정보를 기록하고, 다중 시점 생육 추적과 더 큰 표본을 확보하면 더 구체적인 해석이 가능할 것이다.
            """)

            # ─── 3-11. 요약 카드 ───
            st.divider()
            st.subheader("핵심 요약 (Quick Summary)")
            st.caption("송도고 그린2팀 1회차 보고서 (업스테이지·극지연구소 검수, 2026-10-28) 기반")

            summary_cols = st.columns(4)
            summary_data = [
                ("연구 목적", "같은 스마트팜에서 재배한 극지식물 81개체 간 생육 차이 확인, 잎 수·지상부·지하부 길이로 설명 가능한지, 환경 변화와 관련 있는지 분석"),
                ("핵심 결과", "생중량 250배 차이(0.04g~10.01g), CV 89.42%. 잎 수−생중량 r=0.886, Top/Bottom 잎 수 6.4배 차이"),
                ("가설 검증", "지지됨: 잎 수 상위 25% > 하위 25% 생중량 (p=4.49×10⁻⁹)"),
                ("예측모델", "단순회귀 R²=0.7845, MAE=0.772g. 교차검증에서 다중회귀보다 안정적. 지상부 길이 추가 효과 없음(p=0.190)"),
            ]
            for i, (title, content) in enumerate(summary_data):
                with summary_cols[i]:
                    st.markdown(f"**{title}**")
                    st.info(content)


if __name__ == "__main__":
    main()
