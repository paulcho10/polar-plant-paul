#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
- CSV 인코딩(utf-8-sig/cp949) 오류 방지
- 학교마다 다른 열 이름(KeyError) 방지
- 한글 폰트 깨짐 방지
- Streamlit Cloud에서 에러 없이 실행
"""

import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def setup_korean_font():
    """여러 환경에서 한글 폰트를 찾아 설정"""
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/app/.fonts/NanumGothic.ttf",
    ]

    chosen_font = None
    for fp in font_paths:
        if os.path.exists(fp):
            chosen_font = fp
            break

    if chosen_font is None:
        for f in fm.fontManager.ttflist:
            if "Nanum" in f.name or "Noto" in f.name:
                chosen_font = f.fname
                break

    if chosen_font:
        fe = fm.FontEntry(fname=chosen_font, name="KoreanFont")
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams["font.family"] = fe.name
    else:
        plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams["axes.unicode_minus"] = False


def read_csv_with_fallback(uploaded_file):
    """인코딩과 열 이름 다름을 모두 처리하는 CSV 읽기"""
    if uploaded_file is None:
        return None

    for enc in ["utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            df = pd.read_csv(uploaded_file, encoding=enc)
            if df.shape[0] > 0:
                break
        except (UnicodeDecodeError, pd.errors.ParserError):
            uploaded_file.seek(0)
            continue
    else:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        uploaded_file.seek(0)
        df = pd.read_csv(io.StringIO(raw), encoding="utf-8", on_bad_lines="skip")

    uploaded_file.seek(0)
    return df


def normalize_column_name(col):
    """열 이름을 정규화: 공백 제거, 단위 분리, 소문자 통일"""
    s = str(col).strip()
    for unit in ["(장)", "(mm)", "(g)", "(%)", "(degC)", " (degC)", " (%)", " (장)", " (mm)", " (g)"]:
        s = s.replace(unit, "")
    s = s.strip()
    return s


def find_column(df, candidates, normalize=True):
    """여러 후보 이름 중 실제 존재하는 열 찾기 (KeyError 방지)"""
    if df is None or df.shape[1] == 0:
        return None

    for c in candidates:
        if c in df.columns:
            return c
        if normalize:
            norm_cols = {normalize(col): col for col in df.columns}
            if normalize(c) in norm_cols:
                return norm_cols[normalize(c)]
    return None


def preprocess_env(df):
    """환경 데이터: 시간 파싱, 결측 처리, 시간대 추출"""
    if df is None:
        return None

    time_col = find_column(df, ["time", "Time", "datetime", "날짜", "일시", "날짜/시간", "date"])
    if time_col is None:
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if "2026" in sample or "_" in sample and any(c.isdigit() for c in sample):
                time_col = col
                break

    if time_col is None:
        st.warning("시간 열을 찾을 수 없습니다.")
        return None

    df = df.copy()
    df["_time"] = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    df = df.dropna(subset=["_time"])
    df["_hour"] = df["_time"].dt.hour
    df["_date"] = df["_time"].dt.date

    temp_col = find_column(df, ["temperature", "온도", "temp", "온도(degC)"])
    hum_col = find_column(df, ["humidity", "습도", "hum", "습도(%)"])
    ec_col = find_column(df, ["ec", "EC", "ec"])
    ph_col = find_column(df, ["ph", "PH", "pH", "PH"])
    wt_col = find_column(df, ["water_temp", "수온", "water_temp"])
    co2_col = find_column(df, ["co2", "CO2", "co2"])

    result = {"_time": df["_time"], "_hour": df["_hour"], "_date": df["_date"]}

    if temp_col:
        result["온도"] = pd.to_numeric(df[temp_col], errors="coerce")
    if hum_col:
        result["습도"] = pd.to_numeric(df[hum_col], errors="coerce")
    if ec_col:
        result["EC"] = pd.to_numeric(df[ec_col], errors="coerce")
    if ph_col:
        result["pH"] = pd.to_numeric(df[ph_col], errors="coerce")
    if wt_col:
        result["수온"] = pd.to_numeric(df[wt_col], errors="coerce")
    if co2_col:
        result["CO2"] = pd.to_numeric(df[co2_col], errors="coerce")

    return pd.DataFrame(result)


def preprocess_growth(df):
    """생육 데이터: 열 이름 정규화, 단위 분리"""
    if df is None:
        return None

    df = df.copy()

    id_col = find_column(df, ["개체번호", "번호", "ID", "id", "개체번호 "])
    if id_col:
        df["개체번호"] = pd.to_numeric(df[id_col], errors="coerce")

    leaf_col = find_column(df, ["잎 수(장)", "잎 수", "leaf", "잎수", "잎 수 (장)"])
    if leaf_col:
        df["잎 수"] = pd.to_numeric(df[leaf_col], errors="coerce")

    shoot_col = find_column(df, ["지상부 길이(mm)", "지상부 길이", "지상부길이", "지상부 길이 (mm)", "shoot"])
    if shoot_col:
        df["지상부 길이"] = pd.to_numeric(df[shoot_col], errors="coerce")

    root_col = find_column(df, ["지하부길이(mm)", "지하부 길이(mm)", "지하부 길이", "지하부길이", "root"])
    if root_col:
        df["지하부 길이"] = pd.to_numeric(df[root_col], errors="coerce")

    fw_col = find_column(df, ["생중량(g)", "생중량", "fresh_weight", "생중량 (g)", "fw"])
    if fw_col:
        df["생중량"] = pd.to_numeric(df[fw_col], errors="coerce")

    shoot_m_col = find_column(df, ["지상부질량(g)", "지상부질량", "지상부 질량", "shoot_mass"])
    if shoot_m_col:
        df["지상부질량"] = pd.to_numeric(df[shoot_m_col], errors="coerce")

    root_m_col = find_column(df, ["지하부질량(g)", "지하부질량", "지하부 질량", "root_mass"])
    if root_m_col:
        df["지하부질량"] = pd.to_numeric(df[root_m_col], errors="coerce")

    keep = ["개체번호", "잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부질량", "지하부질량"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["개체번호"])
    df = df.sort_values("개체번호").reset_index(drop=True)

    return df


def main():
    setup_korean_font()

    st.set_page_config(page_title="송도고 극지식물 생육 대시보드", layout="wide")

    st.title("극지식물 개체별 생육 차이 분석 대시보드")
    st.caption("같은 스마트팜에서 재배한 극지식물 81개체의 생육 결과와 환경 데이터를 분석합니다.")

    st.sidebar.header("데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
    growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type=["csv"])

    tab_env, tab_growth, tab_analysis = st.tabs([
        "환경 데이터",
        "생육 결과",
        "분석 결과"
    ])

    env_df = None
    growth_df = None

    if env_file:
        raw_env = env_file.read()
        env_df = preprocess_env(pd.read_csv(io.BytesIO(raw_env), encoding="utf-8-sig"))
        if env_df is None:
            env_df = preprocess_env(pd.read_csv(io.BytesIO(raw_env), encoding="cp949"))

    if growth_file:
        raw_growth = growth_file.read()
        growth_df = preprocess_growth(pd.read_csv(io.BytesIO(raw_growth), encoding="utf-8-sig"))
        if growth_df is None or growth_df.shape[0] == 0:
            growth_df = preprocess_growth(pd.read_csv(io.BytesIO(raw_growth), encoding="cp949"))

    # 탭 1: 환경 데이터
    with tab_env:
        st.header("스마트팜 환경 데이터")

        if env_df is None or env_df.shape[0] == 0:
            st.info("환경 데이터 CSV를 업로드하세요.")
            example_text = "time, temperature, humidity, ph, ec, water_temp, co2"
            example_text += "\n"
            example_text += "2026-07-15 06:00:00, 24.5, 60.1, 7.2, 0.5, 23.0, 400"
            st.code(example_text, language="text")
        else:
            st.success(f"{len(env_df):,}행 로드 완료")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("날짜별 평균 온도·습도 시계열")
                if "온도" in env_df.columns and "습도" in env_df.columns:
                    daily = env_df.groupby("_date").agg({"온도": "mean", "습도": "mean"}).reset_index()
                    daily["_date"] = pd.to_datetime(daily["_date"])

                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(daily["_date"], daily["온도"], color="#e74c3c", marker="o", markersize=3, label="온도 (C)")
                    ax1.set_xlabel("날짜")
                    ax1.set_ylabel("온도 (C)", color="#e74c3c")
                    ax1.tick_params(axis="y", labelcolor="#e74c3c")

                    ax2 = ax1.twinx()
                    ax2.plot(daily["_date"], daily["습도"], color="#3498db", marker="s", markersize=3, label="습도 (%)")
                    ax2.set_ylabel("습도 (%)", color="#3498db")
                    ax2.tick_params(axis="y", labelcolor="#3498db")

                    fig.autofmt_xdate()
                    plt.title("날짜별 평균 온도·습도")
                    st.pyplot(fig)
                else:
                    st.warning("온도 또는 습도 열을 찾을 수 없습니다.")

            with col2:
                st.subheader("시간대별 평균 온도·습도")
                if "온도" in env_df.columns and "습도" in env_df.columns:
                    hourly = env_df.groupby("_hour").agg({"온도": "mean", "습도": "mean"}).reset_index()

                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    bars = ax1.bar(hourly["_hour"], hourly["온도"], color="#e74c3c", alpha=0.7, label="온도 (C)")
                    ax1.set_xlabel("시간대 (시)")
                    ax1.set_ylabel("온도 (C)", color="#e74c3c")
                    ax1.set_xticks(range(0, 24))
                    ax1.tick_params(axis="y", labelcolor="#e74c3c")

                    ax2 = ax1.twinx()
                    ax2.plot(hourly["_hour"], hourly["습도"], color="#3498db", marker="o", linewidth=2, label="습도 (%)")
                    ax2.set_ylabel("습도 (%)", color="#3498db")
                    ax2.tick_params(axis="y", labelcolor="#3498db")

                    plt.title("24개 시간대별 평균 온도·습도")
                    st.pyplot(fig)

            st.divider()
            st.subheader("환경 기초 통계")

            stat_cols = []
            if "온도" in env_df.columns:
                t = env_df["온도"].dropna()
                cv_val = abs(t.std()/t.mean()*100) if t.mean() != 0 else 0
                stat_cols.append(("온도 (C)", f"{t.mean():.2f}", f"{t.std():.2f}", f"{t.min():.2f}", f"{t.max():.2f}", f"{cv_val:.1f}"))
            if "습도" in env_df.columns:
                h = env_df["습도"].dropna()
                cv_val = abs(h.std()/h.mean()*100) if h.mean() != 0 else 0
                stat_cols.append(("습도 (%)", f"{h.mean():.2f}", f"{h.std():.2f}", f"{h.min():.2f}", f"{h.max():.2f}", f"{cv_val:.1f}"))
            if "EC" in env_df.columns:
                ec = env_df["EC"].dropna()
                cv_val = abs(ec.std()/ec.mean()*100) if ec.mean() != 0 else 0
                stat_cols.append(("EC", f"{ec.mean():.2f}", f"{ec.std():.2f}", f"{ec.min():.2f}", f"{ec.max():.2f}", f"{cv_val:.1f}"))
            if "pH" in env_df.columns:
                ph = env_df["pH"].dropna()
                cv_val = abs(ph.std()/ph.mean()*100) if ph.mean() != 0 else 0
                stat_cols.append(("pH", f"{ph.mean():.2f}", f"{ph.std():.2f}", f"{ph.min():.2f}", f"{ph.max():.2f}", f"{cv_val:.1f}"))

            if stat_cols:
                table_data = []
                for name, m, s, mn, mx, cv in stat_cols:
                    table_data.append({"항목": name, "평균": m, "표준편차": s, "최소": mn, "최대": mx, "CV(%)": cv})
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            st.divider()
            st.subheader("전반부 vs 후반부 비교")

            if len(env_df) > 0:
                midpoint = env_df["_date"].min() + (env_df["_date"].max() - env_df["_date"].min()) / 2
                first_half = env_df[env_df["_date"] <= midpoint]
                second_half = env_df[env_df["_date"] > midpoint]

                comp_data = []
                for name in ["온도", "습도", "EC", "pH"]:
                    if name in env_df.columns:
                        f_data = first_half[name].dropna()
                        s_data = second_half[name].dropna()
                        var_ratio = (s_data.var()/f_data.var()) if f_data.var() > 0 else 0
                        comp_data.append({
                            "항목": name,
                            "전반부 평균": f"{f_data.mean():.2f}",
                            "후반부 평균": f"{s_data.mean():.2f}",
                            "평균 차이": f"{s_data.mean()-f_data.mean():+.2f}",
                            "분산비(후반/전반)": f"{var_ratio:.2f}"
                        })

                if comp_data:
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # 탭 2: 생육 결과
    with tab_growth:
        st.header("개체별 생육 결과 데이터")

        if growth_df is None or growth_df.shape[0] == 0:
            st.info("생육 결과 CSV를 업로드하세요.")
            example_text = "개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)"
            example_text += "\n"
            example_text += "1, 18, 17, 7, 0.19, 0.14, 0.05"
            st.code(example_text, language="text")
        else:
            st.success(f"{len(growth_df)}개체 로드 완료")

            st.subheader("전체 생육 데이터")
            st.dataframe(growth_df, use_container_width=True)

            st.divider()
            st.subheader("생육 기초 통계")

            stat_items = []
            for col_name, label in [("잎 수", "잎 수 (장)"), ("지상부 길이", "지상부 길이 (mm)"),
                                      ("지하부 길이", "지하부 길이 (mm)"), ("생중량", "생중량 (g)")]:
                if col_name in growth_df.columns:
                    s = growth_df[col_name].dropna()
                    mean_v = s.mean()
                    std_v = s.std()
                    cv_v = abs(std_v/mean_v*100) if mean_v != 0 else 0
                    min_v = s.min()
                    max_v = s.max()
                    ratio_v = max_v/min_v if min_v > 0 else 0
                    stat_items.append({
                        "항목": label,
                        "평균": f"{mean_v:.2f}",
                        "중앙값": f"{s.median():.2f}",
                        "표준편차": f"{std_v:.2f}",
                        "CV(%)": f"{cv_v:.1f}",
                        "최소": f"{min_v:.2f}",
                        "최대": f"{max_v:.2f}",
                        "최대/최소": f"{ratio_v:.1f}"
                    })

            if stat_items:
                st.dataframe(pd.DataFrame(stat_items), use_container_width=True)

            st.divider()
            st.subheader("생중량 분포 히스토그램")
            if "생중량" in growth_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(growth_df["생중량"].dropna(), bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
                median_val = growth_df["생중량"].median()
                ax.axvline(median_val, color="red", linestyle="--", label=f"중앙값: {median_val:.2f}g")
                ax.set_xlabel("생중량 (g)")
                ax.set_ylabel("개체 수")
                ax.legend()
                plt.title("생중량(g) 분포")
                st.pyplot(fig)

            st.divider()
            st.subheader("이상치 분석 (IQR 기준)")
            outlier_data = []
            for col_name, label in [("잎 수", "잎 수"), ("지상부 길이", "지상부 길이"),
                                      ("지하부 길이", "지하부 길이"), ("생중량", "생중량")]:
                if col_name in growth_df.columns:
                    q1 = growth_df[col_name].quantile(0.25)
                    q3 = growth_df[col_name].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = growth_df[(growth_df[col_name] < lower) | (growth_df[col_name] > upper)]
                    outlier_ids = ", ".join(map(str, outliers["개체번호"].tolist())) if len(outliers) > 0 else "-"
                    outlier_vals = ", ".join([f"{v:.2f}" for v in outliers[col_name]]) if len(outliers) > 0 else "-"
                    outlier_data.append({
                        "항목": label,
                        "이상치 개수": len(outliers),
                        "이상치 개체번호": outlier_ids,
                        "이상치 값": outlier_vals
                    })

            if outlier_data:
                st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)

    # 탭 3: 분석 결과
    with tab_analysis:
        st.header("분석 결과")

        if growth_df is None or growth_df.shape[0] == 0:
            st.info("생육 결과 CSV를 업로드하면 분석 결과가 표시됩니다.")
        else:
            # 상관분석
            st.subheader("1. 상관계수 분석 (생중량 vs 생육 항목)")

            if all(c in growth_df.columns for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]):
                corr_data = []
                for col, label in [("잎 수", "잎 수"), ("지상부 길이", "지상부 길이"), ("지하부 길이", "지하부 길이")]:
                    r, p = stats.pearsonr(growth_df["생중량"], growth_df[col])
                    if r > 0.7:
                        interp = "강한 양의 상관"
                    elif r > 0.4:
                        interp = "중간 상관"
                    else:
                        interp = "약한 상관"
                    corr_data.append({
                        "생육 항목": label,
                        "Pearson r": f"{r:.4f}",
                        "p값": f"{p:.6f}",
                        "해석": interp
                    })

                st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
                st.caption("Pearson 상관계수(r): -1~1 사이 값. 1에 가까울수록 한 변수가 증가할 때 다른 변수도 함께 증가하는 경향이 강함. p값이 0.05보다 작으면 통계적으로 유의한 관계.")

            # 그룹 비교
            st.subheader("2. Top 25% vs Bottom 25% 그룹 비교")

            if "생중량" in growth_df.columns:
                fw = growth_df["생중량"].dropna()
                n_top = int(np.ceil(len(fw) * 0.25))
                threshold_top = fw.nlargest(n_top).min()
                threshold_bottom = fw.nsmallest(n_top).max()

                top_group = growth_df[growth_df["생중량"] >= threshold_top].head(n_top)
                bottom_group = growth_df[growth_df["생중량"] <= threshold_bottom].head(n_top)

                st.caption(f"상위 25% 기준값: {threshold_top:.2f}g 이상 / 하위 25% 기준값: {threshold_bottom:.2f}g 이하 / 각 그룹 {n_top}개체")

                group_data = []
                for col, label in [("잎 수", "잎 수 (장)"), ("지상부 길이", "지상부 길이 (mm)"),
                                    ("지하부 길이", "지하부 길이 (mm)")]:
                    if col in growth_df.columns:
                        t_mean = top_group[col].mean()
                        b_mean = bottom_group[col].mean()
                        t_std = top_group[col].std()
                        b_std = bottom_group[col].std()
                        ratio = t_mean / b_mean if b_mean != 0 else 0
                        t_stat, p_val = stats.ttest_ind(top_group[col], bottom_group[col], equal_var=False)
                        if p_val < 0.001:
                            sig = "유의함 (p<0.001)"
                        elif p_val < 0.05:
                            sig = "유의함 (p<0.05)"
                        else:
                            sig = "유의하지 않음"
                        group_data.append({
                            "항목": label,
                            "Top 평균": f"{t_mean:.2f}",
                            "Bottom 평균": f"{b_mean:.2f}",
                            "평균 차이": f"{t_mean-b_mean:.2f}",
                            "비율(Top/Bottom)": f"{ratio:.2f}",
                            "p값(Welch t검정)": f"{p_val:.6f}",
                            "유의성": sig
                        })

                if group_data:
                    st.dataframe(pd.DataFrame(group_data), use_container_width=True)

                if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns:
                    top_ratio = (top_group["지상부 길이"] / top_group["지하부 길이"]).mean()
                    bottom_ratio = (bottom_group["지상부 길이"] / bottom_group["지하부 길이"]).mean()
                    st.metric("Top 그룹 지상부/지하부 비율", f"{top_ratio:.3f}")
                    st.metric("Bottom 그룹 지상부/지하부 비율", f"{bottom_ratio:.3f}")
                    st.caption("비율이 1보다 작으면 지상부보다 지하부가 더 길다는 의미. 잘 자란 개체일수록 지하부가 더 발달하는 경향을 보임.")

            # 예측 모델
            st.subheader("3. 생중량 예측 모델 (선형회귀)")

            if all(c in growth_df.columns for c in ["생중량", "잎 수"]):
                X1 = growth_df[["잎 수"]].values
                y = growth_df["생중량"].values

                slope1, intercept1, r1, p1, se1 = stats.linregress(growth_df["잎 수"], growth_df["생중량"])
                r2_1 = r1 ** 2
                pred1 = slope1 * growth_df["잎 수"] + intercept1
                mae1 = np.mean(np.abs(pred1 - y))

                model_formula = f"모델 1 (단순회귀): 생중량 = {slope1:.4f} x 잎 수 + ({intercept1:.4f})"
                st.code(model_formula)
                st.write(f"**R-squared** = {r2_1:.4f}  (모델이 데이터 변동을 설명하는 비율, 0~1 사이)")
                st.write(f"**MAE** = {mae1:.3f}g  (예측값과 실제값 차이의 평균)")

                info_text = f"R-squared = {r2_1:.4f}는 잎 수로 생중량 변동의 약 {r2_1*100:.0f}%를 설명할 수 있다는 의미. 하지만 약 {(1-r2_1)*100:.0f}%는 잎 수만으로 설명되지 않음."
                st.info(info_text)

                from sklearn.model_selection import KFold
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_absolute_error

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2_1 = []
                cv_mae_1 = []

                for train_idx, test_idx in kf.split(X1):
                    model = LinearRegression()
                    model.fit(X1[train_idx], y[train_idx])
                    pred = model.predict(X1[test_idx])
                    cv_r2_1.append(r2_score(y[test_idx], pred))
                    cv_mae_1.append(mean_absolute_error(y[test_idx], pred))

                st.write(f"**5겹 교차검증** - R-squared: {np.mean(cv_r2_1):.4f}, MAE: {np.mean(cv_mae_1):.3f}g")

                st.subheader("예측 예시")
                col_a, col_b = st.columns(2)
                with col_a:
                    pred_25 = slope1 * 25 + intercept1
                    st.metric("잎 수 25장일 때 예측 생중량", f"{pred_25:.2f}g")
                with col_b:
                    pred_60 = slope1 * 60 + intercept1
                    st.metric("잎 수 60장일 때 예측 생중량", f"{pred_60:.2f}g")

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(growth_df["잎 수"], growth_df["생중량"], color="#3498db", alpha=0.6, label="실제 데이터")
                x_range = np.linspace(growth_df["잎 수"].min(), growth_df["잎 수"].max(), 100)
                ax.plot(x_range, slope1 * x_range + intercept1, color="#e74c3c", linewidth=2, label="회귀선")
                ax.set_xlabel("잎 수 (장)")
                ax.set_ylabel("생중량 (g)")
                ax.legend()
                plt.title("잎 수 vs 생중량 산점도 및 회귀선")
                st.pyplot(fig)

            # 보고서 핵심 결론
            st.divider()
            st.subheader("보고서 핵심 결론 (PDF 기반)")

            conclusions = [
                ("환경 측면", "온도와 습도는 재배 기간 내내 비교적 안정적이었다. 반면 EC와 pH는 재배 기간 내내 뚜렷한 방향 변화를 보였다. EC는 6월 초 0.38에서 8월 하순 1.42로 약 274% 상승, pH는 7.96에서 6.52로 약 1.54pH 단위 하락했다."),
                ("개체별 생육 차이", "생중량은 평균 2.48g, 중앙값 1.76g, CV 89.42%로 네 항목 중 개체 간 차이가 가장 컸다. 생중량이 가장 큰 개체는 21번(10.01g), 가장 작은 개체는 36번(0.04g)으로 최대/최소 배율 250.25배였다."),
                ("상관 관계", "생중량과 가장 강한 상관을 보인 항목은 잎 수(r = 0.8857, p < 0.001)였고, 지하부 길이(r = 0.6055), 지상부 길이(r = 0.3521) 순이었다."),
                ("가설 검증", "가설 '잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다'는 데이터로 지지되었다. Top 평균 72.55장 vs Bottom 11.40장(p = 4.49 x 10^-9), 생중량 5.93g vs 0.37g."),
                ("예측 모델", "잎 수만 사용한 단순회귀(R-squared = 0.7845)가 잎 수+지상부 길이를 사용한 다중회귀(R-squared = 0.7893)와 실질적으로 비슷한 성능을 보였고, 교차검증에서는 오히려 단순회귀가 더 안정적이었다."),
                ("한계점", "생육 측정일과 개체 위치 정보가 없어 환경 변화와 생육 차이를 직접 연결할 수 없었다. 모든 통계 결과는 인과관계가 아닌 관계의 경향을 보여준다. 다른 환경이나 다른 품종에 그대로 적용된다고 단정할 수 없다.")
            ]

            for title, content in conclusions:
                st.markdown(f"**{title}**")
                st.write(content)
                st.divider()


if __name__ == "__main__":
    main()
