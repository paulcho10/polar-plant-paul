import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 한글 폰트 설정 (Streamlit Cloud 대비)
# ─────────────────────────────────────────────
def setup_korean_font():
    # 우선 시스템에 설치된 한글 폰트를 찾는다
    candidates = []
    for f in fm.fontManager.ttflist:
        name = f.name.lower()
        if any(k in name for k in ["noto sans cjk", "noto sans kr", "nanum", "nanumgothic",
                                     "nanummyeongjo", "batang", "dotum", "gulim", "malgun"]):
            candidates.append(f)

    if not candidates:
        # fallback: /usr/share/fonts 어딘가에 있을 수 있는 폰트 직접 스캔
        import glob
        scan_paths = [
            "/usr/share/fonts/truetype",
            "/usr/share/fonts",
        ]
        for p in scan_paths:
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    for fn in files:
                        if fn.lower().endswith(".ttf"):
                            fp = os.path.join(root, fn)
                            try:
                                prop = fm.FontProperties(fname=fp)
                                name_lower = (prop.get_name() or "").lower()
                                if any(k in name_lower for k in ["noto", "nanum", "batang", "dotum", "gulim", "malgun"]):
                                    candidates.append(fm.Font(f.name, f.style, f.variant,
                                                                f.weight, f.stretch, f.filename))
                            except Exception:
                                pass

    if candidates:
        font_path = candidates[0].fname
        fm.fontManager.addfont(font_path)
        fp = fm.FontProperties(fname=font_path)
        font_name = fp.get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
    else:
        # 최후의 수단: 기본 sans-serif
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# ─────────────────────────────────────────────
# 환경 데이터 열 매핑 (학교마다 다른 열 이름 대응)
# ─────────────────────────────────────────────
ENV_COLUMN_ALIASES = {
    "time":       ["time", "일자", "날짜", "측정일시", "timestamp"],
    "temperature":[ "temperature", "온도", "temp", "기온", "air_temp"],
    "humidity":   ["humidity", "습도", "rh"],
    "ec":         ["ec", "EC", "전도도", "electrical_conductivity"],
    "ph":         ["ph", "PH", "산도", "pH"],
    "water_temp": ["water_temp", "수온", "water_temperature"],
    "co2":        ["co2", "CO2", "이산화탄소", "co2_ppm"],
}

def detect_env_columns(df: pd.DataFrame) -> dict:
    """데이터프레임의 실제 열 이름 → 표준 키 매핑을 반환"""
    mapping = {}
    for std_key, aliases in ENV_COLUMN_ALIASES.items():
        matched = None
        for col in df.columns:
            if col.strip() in [a.strip() for a in aliases]:
                matched = col
                break
        if matched:
            mapping[std_key] = matched
    return mapping

def clean_env_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """시간 열을 datetime으로 정규화 (두 형식 모두 처리)"""
    s = df[time_col].astype(str).str.strip()
    dt = pd.to_datetime(s, format="mixed", errors="coerce")
    if dt.isna().sum() == len(dt):
        # '20260715_0' 형식 시도
        dt = pd.to_datetime(s.str.replace("_", " ", regex=False),
                             format="%Y%m%d %H", errors="coerce")
    df = df.copy()
    df["_dt"] = dt
    return df

# ─────────────────────────────────────────────
# 생육 데이터 열 매핑
# ─────────────────────────────────────────────
GROWTH_COLUMN_ALIASES = {
    "개체번호":   ["개체번호", "식물번호", "plant_id", "id", "번호"],
    "잎 수":      ["잎 수(장)", "잎수", "leaf_count", "leaves"],
    "지상부 길이": ["지상부 길이(mm)", "지상부길이(mm)", "shoot_length", "지상부길이"],
    "지하부 길이": ["지하부길이(mm)", "지하부 길이(mm)", "root_length", "지하부길이"],
    "생중량":     ["생중량(g)", "생중량", "fresh_weight", "생체중"],
    "지상부 질량": ["지상부질량(g)", "지상부 질량(g)", "shoot_mass", "지상부질량", "지상부무게"],
    "지하부 질량": ["지하부질량(g)", "지하부 질량(g)", "root_mass", "지하부질량", "지하부무게"],
}

def detect_growth_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    for std_key, aliases in GROWTH_COLUMN_ALIASES.items():
        matched = None
        for col in df.columns:
            if col.strip() in [a.strip() for a in aliases]:
                matched = col
                break
        if matched:
            mapping[std_key] = matched
    return mapping

# ─────────────────────────────────────────────
# 통계 요약 함수
# ─────────────────────────────────────────────
def describe_series(s: pd.Series, label: str) -> str:
    """평균·표준편차·중앙값·최소·최대·개수를 짧게 정리"""
    valid = s.dropna()
    if len(valid) == 0:
        return f"{label}: 데이터가 없습니다"
    return (
        f"**{label}**  "
        f"n={len(valid)}  |  "
        f"평균(Mean)={valid.mean():.2f}  |  "
        f"표준편차(SD)={valid.std():.2f}  |  "
        f"중앙값(Median)={valid.median():.2f}  |  "
        f"최소={valid.min():.2f}  |  "
        f"최대={valid.max():.2f}"
    )

def correlation_text(x: pd.Series, y: pd.Series, xname: str, yname: str) -> str:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return f"{xname}–{yname}: 데이터가 부족합니다"
    r = df.iloc[:, 0].corr(df.iloc[:, 1])
    strength = ("매우 강한" if abs(r) >= 0.8 else
                "강한" if abs(r) >= 0.6 else
                "중간 정도의" if abs(r) >= 0.4 else
                "약한" if abs(r) >= 0.2 else "거의 없는")
    direction = ("양의" if r > 0 else "음의") if abs(r) >= 0.2 else ""
    return (
        f"**{xname} ↔ {yname}**  "
        f"r = {r:.3f}  "
        f"({direction} {strength} 선형 관계, n={len(df)})"
    )

# ─────────────────────────────────────────────
# 페이지 설정 및 사이드바
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 분석 대시보드",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 극지식물 생육 분석 대시보드")
st.caption("송도고등학교 극지식물 개체별 생육 차이 분석  |  환경 데이터 + 생육 결과 데이터 시각화")
st.divider()

with st.sidebar:
    st.header("📂 데이터 업로드")
    env_file  = st.file_uploader("환경 데이터 (CSV)", type=["csv"], key="env_uploader")
    growth_file = st.file_uploader("생육 결과 데이터 (CSV)", type=["csv"], key="growth_uploader")
    st.divider()
    st.caption("※ 인코딩 오류·열 이름 불일치·한글 폰트 깨짐을 자동으로 처리합니다.")
    st.caption("※ school마다 다른 열 이름도 매핑하여 동작합니다.")

# ─────────────────────────────────────────────
# 데이터 로드 헬퍼
# ─────────────────────────────────────────────
def load_csv_flexible(path_or_bytes, encoding_priority=None):
    """utf-8-sig → cp949 순으로 시도하여 CSV 읽기"""
    if encoding_priority is None:
        encoding_priority = ["utf-8-sig", "cp949", "utf-8", "euc-kr"]

    if hasattr(path_or_bytes, "read"):
        raw = path_or_bytes.read()
        for enc in encoding_priority:
            try:
                text = raw.decode(enc)
                df = pd.read_csv(pd.io.common.StringIO(text))
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("CSV 인코딩을 인식할 수 없습니다. utf-8-sig 또는 cp949 형식으로 저장해 주세요.")
    else:
        for enc in encoding_priority:
            try:
                return pd.read_csv(path_or_bytes, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 인코딩을 인식할 수 없습니다.")

# ─────────────────────────────────────────────
# 업로드 데이터가 없을 때 → 내장 샘플 데이터 사용
# ─────────────────────────────────────────────
SAMPLE_ENV_PATH    = "/home/user/outputs/송도고등학교_환경데이터.csv"
SAMPLE_GROWTH_PATH = "/home/user/outputs/송도고등학교_생육결과데이터.csv"

def get_env_df(env_file):
    if env_file is not None:
        return load_csv_flexible(env_file)
    if os.path.exists(SAMPLE_ENV_PATH):
        return pd.read_csv(SAMPLE_ENV_PATH, encoding="utf-8-sig")
    return None

def get_growth_df(growth_file):
    if growth_file is not None:
        return load_csv_flexible(growth_file)
    if os.path.exists(SAMPLE_GROWTH_PATH):
        return pd.read_csv(SAMPLE_GROWTH_PATH, encoding="utf-8-sig")
    return None

# ─────────────────────────────────────────────
# 메인 데이터 로딩
# ─────────────────────────────────────────────
env_df_raw    = get_env_df(env_file)
growth_df_raw = get_growth_df(growth_file)

if env_df_raw is None and growth_df_raw is None:
    st.warning("환경 데이터와 생육 결과 데이터를 모두 업로드하거나, 앱에 내장된 샘플 데이터를 확인하세요.\n(현재 업로드된 파일이 없어 샘플 데이터를 로드 중입니다.)")

# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab_env, tab_growth, tab_corr = st.tabs([
    "🌡️  환경 데이터 시계열",
    "🌱  생육 결과 분석",
    "🔗  환경–생육 상관관계",
])

# ============================================================
# 탭 1 — 환경 데이터 시계열
# ============================================================
with tab_env:
    if env_df_raw is not None:
        env_df = env_df_raw.copy()
        col_map = detect_env_columns(env_df)

        if not col_map:
            st.error("환경 데이터의 열 이름을 인식할 수 없습니다.\n"
                     "다음 열 이름 중 하나가 필요합니다:\n"
                     "`time / temperature / humidity / ec / ph / water_temp / co2`")
        else:
            time_col = col_map.get("time")
            env_df  = clean_env_time(env_df, time_col) if time_col else env_df

            st.subheader("📈 환경 측정값 시계열")
            env_cols_show = []
            labels_env    = []
            for std_key, label in [
                ("temperature", "온도 (°C)"),
                ("humidity",    "습도 (%)"),
                ("ec",          "EC (mS/cm)"),
                ("ph",          "pH"),
                ("water_temp",  "수온 (°C)"),
                ("co2",         "CO₂ (ppm)"),
            ]:
                raw = col_map.get(std_key)
                if raw and raw in env_df.columns:
                    env_cols_show.append(raw)
                    labels_env.append(label)

            if env_cols_show:
                chart_df = env_df.set_index("_dt" if "_dt" in env_df.columns else env_df.columns[0])[env_cols_show]
                st.line_chart(chart_df)

            st.divider()
            st.subheader("📊 환경 변수별 기초 통계")

            stats_cols = st.columns(3)
            for i, (std_key, label) in enumerate([
                ("temperature", "온도 (°C)"),
                ("humidity",    "습도 (%)"),
                ("ec",          "EC (mS/cm)"),
                ("ph",          "pH"),
                ("water_temp",  "수온 (°C)"),
                ("co2",         "CO₂ (ppm)"),
            ]):
                raw = col_map.get(std_key)
                if raw and raw in env_df.columns:
                    s = env_df[raw].astype(float)
                    txt = describe_series(s, label)
                    with stats_cols[i % 3]:
                        st.info(txt)

            with st.expander("📖 통계 용어 간단 설명"):
                st.markdown("""
- **평균(Mean)**: 모든 값을 더해 개수로 나눈 값. 데이터의 '대표값'입니다.
- **표준편차(SD)**: 값이 평균 주변에서 얼마나 퍼져 있는지 나타냅니다. 클수록 변동이 큽니다.
- **중앙값(Median)**: 데이터를 크기순으로 정렬했을 때 정가운데 값. 극단값의 영향이 적습니다.
- **최소/최대**: 가장 작은 값과 가장 큰 값.
                """)
    else:
        st.info("환경 데이터가 아직 업로드되지 않았습니다. 좌측 사이드바에서 CSV 파일을 올려주세요.")

# ============================================================
# 탭 2 — 생육 결과 분석
# ============================================================
with tab_growth:
    if growth_df_raw is not None:
        gdf = growth_df_raw.copy()
        gmap = detect_growth_columns(gdf)

        if not gmap:
            st.error("생육 결과 데이터의 열 이름을 인식할 수 없습니다.\n"
                     "다음 열 이름 중 하나가 필요합니다:\n"
                     "`개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)`")
        else:
            id_col   = gmap["개체번호"]
            leaf_col = gmap.get("잎 수")
            shoot_len_col = gmap.get("지상부 길이")
            root_len_col  = gmap.get("지하부 길이")
            fresh_col     = gmap.get("생중량")
            shoot_mass_col = gmap.get("지상부 질량")
            root_mass_col  = gmap.get("지하부 질량")

            # ── 생중량 열이 없으면 지상부질량 + 지하부질량으로 계산 ──
            if fresh_col is None and shoot_mass_col is not None and root_mass_col is not None:
                gdf["_생중량_calc"] = gdf[shoot_mass_col].astype(float) + gdf[root_mass_col].astype(float)
                fresh_col = "_생중량_calc"
            elif fresh_col is None:
                st.warning("⚠️ 생중량(g) 열이 없고 지상부/지하부 질량 열도 확인되지 않아 생중량 계산을 생략합니다.")

            gdf["_개체라벨"] = gdf[id_col].astype(str)

            n_individuals = len(gdf)
            st.subheader(f"🌱 생육 결과  ({n_individuals}개체)")

            # ── 개별 개체 표 ──
            st.subheader("📋 개체별 생육 측정값")
            display_cols = []
            col_labels   = []
            if leaf_col:          display_cols.append(gdf[leaf_col]); col_labels.append("잎 수 (장)")
            if shoot_len_col:     display_cols.append(gdf[shoot_len_col]); col_labels.append("지상부 길이 (mm)")
            if root_len_col:      display_cols.append(gdf[root_len_col]); col_labels.append("지하부 길이 (mm)")
            if fresh_col:         display_cols.append(gdf[fresh_col]); col_labels.append("생중량 (g)")
            if shoot_mass_col:    display_cols.append(gdf[shoot_mass_col]); col_labels.append("지상부 질량 (g)")
            if root_mass_col:     display_cols.append(gdf[root_mass_col]); col_labels.append("지하부 질량 (g)")

            if display_cols:
                tbl = pd.concat(display_cols, axis=1)
                tbl.columns = col_labels
                tbl.insert(0, "개체번호", gdf["_개체라벨"])
                st.dataframe(tbl, use_container_width=True, hide_index=True)

            # ── 기초 통계 ──
            st.divider()
            st.subheader("📊 생육 변수별 기초 통계")

            growth_stats = [
                (leaf_col,         "잎 수 (장)"),
                (shoot_len_col,    "지상부 길이 (mm)"),
                (root_len_col,     "지하부 길이 (mm)"),
                (fresh_col,        "생중량 (g)"),
                (shoot_mass_col,   "지상부 질량 (g)"),
                (root_mass_col,    "지하부 질량 (g)"),
            ]
            gc = st.columns(3)
            for i, (col, label) in enumerate(growth_stats):
                if col is not None and col in gdf.columns:
                    s = gdf[col].astype(float)
                    with gc[i % 3]:
                        st.info(describe_series(s, label))

            with st.expander("📖 통계 용어 간단 설명"):
                st.markdown("""
- **평균(Mean)**: 모든 개체의 값을 더한 뒤 개체 수로 나눈 값. '보통 어느 정도인지' 보여줍니다.
- **표준편차(SD, Standard Deviation)**: 개체마다 값이 평균에서 얼마나 떨어져 있는지 나타냅니다. SD가 크면 개체 간 차이가 크다는 뜻입니다.
- **중앙값(Median)**: 값을 크기순으로 나열했을 때 정가운데 위치한 값. 한두 개 극단적 개체에 덜 흔들립니다.
- **최소/최대**: 가장 작은 개체 값과 가장 큰 개체 값.
                """)

            # ── 분포 시각화 ──
            st.divider()
            st.subheader("📊 생육 변수 분포 (히스토그램)")
            hist_cols_pick = []
            for col, label in growth_stats:
                if col is not None and col in gdf.columns:
                    hist_cols_pick.append((col, label))
            if hist_cols_pick:
                hist_c = st.columns(min(3, len(hist_cols_pick)))
                for i, (col, label) in enumerate(hist_cols_pick):
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(gdf[col].dropna().astype(float), bins=15,
                            color="#4CAF50", edgecolor="white", alpha=0.8)
                    ax.set_title(f"{label} 분포")
                    ax.set_xlabel(label)
                    ax.set_ylabel("개체 수")
                    fig.tight_layout()
                    with hist_c[i % len(hist_c)]:
                        st.pyplot(fig)
                    plt.close(fig)

            # ── 개체별 레이더 차트 (상위 지표 비교) ──
            st.divider()
            st.subheader("🧬 개체별 생육 프로필")
            st.caption("각 개체번호를 선택하면 여러 생육 변수를 한 번에 비교할 수 있습니다. "
                       "개체번호는 학교 내부에서 붙인 이름이며, 분석 대상자 간 비교를 위한 열쇠로만 사용됩니다.")

            radar_cands = []
            for col, label in growth_stats:
                if col is not None and col in gdf.columns:
                    radar_cands.append((col, label))

            if len(radar_cands) >= 2:
                sel = st.selectbox("비교할 개체 선택", sorted(gdf["_개체라벨"].unique()))
                fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                vals = []
                for col, label in radar_cands:
                    v = float(gdf.loc[gdf["_개체라벨"] == sel, col].values[0])
                    vals.append(v)
                cats = [label for _, label in radar_cands]
                angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
                vals_plot = vals + [vals[0]]
                angles_plot = angles + [angles[0]]
                ax2.plot(angles_plot, vals_plot, color="#2E7D32", linewidth=2)
                ax2.fill(angles_plot, vals_plot, color="#4CAF50", alpha=0.25)
                ax2.set_xticks(angles)
                ax2.set_xticklabels(cats, fontsize=9)
                ax2.set_title(f"개체 {sel} 생육 프로필", pad=20)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
    else:
        st.info("생육 결과 데이터가 아직 업로드되지 않았습니다. 좌측 사이드바에서 CSV 파일을 올려주세요.")

# ============================================================
# 탭 3 — 환경–생육 상관관계
# ============================================================
with tab_corr:
    if env_df_raw is not None and growth_df_raw is not None:
        edf = env_df_raw.copy()
        gdf = growth_df_raw.copy()
        ecol_map = detect_env_columns(edf)
        gcol_map = detect_growth_columns(gdf)

        if not ecol_map or not gcol_map:
            st.error("두 데이터 모두 열 이름이 확인되어야 상관관계를 분석할 수 있습니다.")
        else:
            # 환경 데이터의 시간 기준 평균값 계산
            time_col = ecol_map.get("time")
            if time_col:
                edf = clean_env_time(edf, time_col)

            env_agg_cols = []
            env_agg_labels = []
            for std_key, label in [
                ("temperature", "평균 온도 (°C)"),
                ("humidity",    "평균 습도 (%)"),
                ("ec",          "평균 EC (mS/cm)"),
                ("ph",          "평균 pH"),
                ("water_temp",  "평균 수온 (°C)"),
                ("co2",         "평균 CO₂ (ppm)"),
            ]:
                raw = ecol_map.get(std_key)
                if raw and raw in edf.columns:
                    if "_dt" in edf.columns:
                        daily = edf.set_index("_dt")[raw].resample("D").mean()
                        env_agg_cols.append(daily)
                        env_agg_labels.append(label)
                    else:
                        env_agg_cols.append(edf[raw].astype(float))
                        env_agg_labels.append(label)

            # 생육 데이터의 개체별 평균 프로필 → 전체 평균 (개체 간 비교이므로 개체번호 제외)
            g_cands = []
            g_labels = []
            for col, label in [
                (gcol_map.get("잎 수"),         "평균 잎 수"),
                (gcol_map.get("지상부 길이"),   "평균 지상부 길이 (mm)"),
                (gcol_map.get("지하부 길이"),   "평균 지하부 길이 (mm)"),
                (gcol_map.get("생중량"),        "평균 생중량 (g)"),
                (gcol_map.get("지상부 질량"),   "평균 지상부 질량 (g)"),
                (gcol_map.get("지하부 질량"),   "평균 지하부 질량 (g)"),
            ]:
                if col is not None and col in gdf.columns:
                    g_cands.append(gdf[col].astype(float).mean())
                    g_labels.append(label)

            if g_cands and env_agg_cols:
                st.subheader("🔗 환경 변수 vs 생육 변수 상관관계")

                st.caption("환경 데이터는 전체 기간의 일평균, 생육 데이터는 전체 개체 평균입니다. "
                           "두 데이터를 직접 짝지은 상관(r)이 아니라, 각 변수의 '대표값 수준'을 비교한 참고 지표입니다.")

                corr_c = st.columns(2)
                idx = 0
                for env_series, env_label in zip(env_agg_cols, env_agg_labels):
                    for g_val, g_label in zip(g_cands, g_labels):
                        # env_series가 Series면 전체 평균 사용
                        env_val = env_series.mean() if hasattr(env_series, "mean") else env_series
                        with corr_c[idx % 2]:
                            # 가상의 쌍 만들기: 환경 변수는 기간 평균, 생육은 전체 평균 → r 계산은 불가
                            # 대신 환경 변수의 일별 변동과 생육 평균의 산점도는 의미 없으므로
                            # → 환경 변수들의 상호작용(온도-습도 등)을 산점도로 보여줌
                            pass
                        idx += 1

                # 실제 의미 있는 상관: 환경 변수 간 상관 + 생육 변수 간 상관
                st.divider()
                st.subheader("🌡️  환경 변수 간 상관관계 (전체 기간)")
                env_df_numeric = edf.copy()
                for _, raw in ecol_map.items():
                    if raw in env_df_numeric.columns:
                        env_df_numeric[raw] = pd.to_numeric(env_df_numeric[raw], errors="coerce")
                env_df_numeric = env_df_numeric[list(ecol_map.values())].dropna()

                if env_df_numeric.shape[1] >= 2 and len(env_df_numeric) > 5:
                    env_corr = env_df_numeric.corr()
                    fig3, ax3 = plt.subplots(figsize=(7, 6))
                    im = ax3.imshow(env_corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
                    ax3.set_xticks(range(len(env_corr.columns)))
                    ax3.set_xticklabels(env_corr.columns, fontsize=9)
                    ax3.set_yticks(range(len(env_corr.index)))
                    ax3.set_yticklabels(env_corr.index, fontsize=9)
                    for i in range(len(env_corr.index)):
                        for j in range(len(env_corr.columns)):
                            v = env_corr.values[i, j]
                            ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                                     fontsize=9,
                                     color="white" if abs(v) > 0.6 else "black")
                    ax3.set_title("환경 변수 간 Pearson 상관계수 (r)")
                    plt.colorbar(im, ax=ax3, shrink=0.8)
                    fig3.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)

                with st.expander("📖 상관계수 r 해석"):
                    st.markdown("""
- **상관계수 r (Pearson correlation coefficient)**: 두 변수가 함께 움직이는 정도를 -1 ~ +1 사이 숫자로 나타냅니다.
- r이 **+1에 가까우면** 두 변수가 함께 증가/감소하는 '양의 상관'.
- r이 **-1에 가까우면** 한 변수가 증가할 때 다른 변수가 감소하는 '음의 상관'.
- r이 **0에 가까우면** 선형 관계가 거의 없다고 봅니다.
- 상관 계수가 높아도 '원인-결과'를 뜻하지는 않습니다.
                    """)

                st.divider()
                st.subheader("🌱  생육 변수 간 상관관계 (개체 간)")
                g_df_num = gdf.copy()
                g_cols_used = []
                for col in [
                    gcol_map.get("잎 수"),
                    gcol_map.get("지상부 길이"),
                    gcol_map.get("지하부 길이"),
                    gcol_map.get("생중량"),
                    gcol_map.get("지상부 질량"),
                    gcol_map.get("지하부 질량"),
                ]:
                    if col is not None and col in g_df_num.columns:
                        g_df_num[col] = pd.to_numeric(g_df_num[col], errors="coerce")
                        g_cols_used.append(col)

                g_df_num = g_df_num[g_cols_used].dropna()
                if g_df_num.shape[1] >= 2 and len(g_df_num) > 3:
                    g_corr = g_df_num.corr()
                    fig4, ax4 = plt.subplots(figsize=(7, 6))
                    im2 = ax4.imshow(g_corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
                    g_labels_short = [c for c in g_cols_used]
                    ax4.set_xticks(range(len(g_labels_short)))
                    ax4.set_xticklabels(g_labels_short, fontsize=8)
                    ax4.set_yticks(range(len(g_labels_short)))
                    ax4.set_yticklabels(g_labels_short, fontsize=8)
                    for i in range(len(g_labels_short)):
                        for j in range(len(g_labels_short)):
                            v = g_corr.values[i, j]
                            ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                                     fontsize=9,
                                     color="white" if abs(v) > 0.6 else "black")
                    ax4.set_title("생육 변수 간 Pearson 상관계수 (r)")
                    plt.colorbar(im2, ax=ax4, shrink=0.8)
                    fig4.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)

                st.divider()
                st.subheader("🌡️ × 🌱  환경 변수 vs 생육 지표 (참고 scatter)")
                st.caption("환경 일평균과 생육 전체 평균을 각각 하나의 점으로 시각화한 참고 그래프입니다. "
                           "개체별 환경 노출 데이터가 있으면 더 정밀한 상관 분석이 가능합니다.")

                scatter_pairs = []
                for env_raw in ecol_map.values():
                    if env_raw in edf.columns:
                        ev = pd.to_numeric(edf[env_raw], errors="coerce").dropna().mean()
                        for g_raw, g_label in [(gcol_map.get("생중량"), "평균 생중량 (g)"),
                                                (gcol_map.get("잎 수"), "평균 잎 수"),
                                                (gcol_map.get("지상부 길이"), "평균 지상부 길이 (mm)")]:
                            if g_raw is not None and g_raw in gdf.columns:
                                scatter_pairs.append((env_raw, ev, g_label, gdf[g_raw].astype(float).mean()))

                if scatter_pairs:
                    fig5, axes5 = plt.subplots(1, min(3, len(scatter_pairs)), figsize=(15, 4))
                    if not hasattr(axes5, "flat"):
                        axes5 = [axes5]
                    for i, (env_name, env_val, g_label, g_val) in enumerate(scatter_pairs[:3]):
                        ax = axes5[i] if len(scatter_pairs) > 1 else axes5
                        ax.scatter([env_val], [g_val], s=120, color="#2E7D32", zorder=5)
                        ax.set_xlabel(f"{env_name} (전체 평균)")
                        ax.set_ylabel(g_label)
                        ax.set_title(f"{env_name} vs {g_label}")
                        ax.grid(alpha=0.3)
                    fig5.tight_layout()
                    st.pyplot(fig5)
                    plt.close(fig5)
            else:
                st.info("환경 데이터 또는 생육 데이터가 충분하지 않아 상관 분석을 생략합니다.")
    else:
        if env_df_raw is None:
            st.info("환경 데이터가 아직 업로드되지 않았습니다. 좌측 사이드바에서 CSV 파일을 올려주세요.")
        if growth_df_raw is None:
            st.info("생육 결과 데이터가 아직 업로드되지 않았습니다. 좌측 사이드바에서 CSV 파일을 올려주세요.")

st.divider()
st.caption("※ 본 대시보드는 교육 목적의 탐색적 분석 도구입니다. "
           "개체 간 비교는 동일 조건에서 재배된 개체들의 상대적 차이를 이해하는 데 도움을 줍니다.")
