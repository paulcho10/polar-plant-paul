"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
====================================================================
사용법:
  1. 터미널에서 streamlit run main.py 실행
  2. 웹 브라우저에서 두 CSV 파일을 업로드
  3. 환경 데이터 + 생육 결과 데이터 분석 결과 확인

호환성:
  - 인코딩: utf-8-sig → cp949 → utf-8 자동 시도
  - 열 이름이 학교마다 달라도 자동 매핑
  - 한글 폰트 깨짐 방지:
      · matplotlib 폰트 매니저를 try_read_cache=False로 리로드
      · 실제 TTF 파일 경로를 FontProperties(fname=...)로 모든 텍스트에 적용
  - Streamlit Cloud 포함 어떤 환경에서도 그래프 한글 깨지지 않음
====================================================================
"""

import io, os, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties, FontEntry

import streamlit as st

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
# 0. 한글 폰트 설정 — 그래프 한글 깨짐 완전 방지
# ─────────────────────────────────────────────────────────────

_KR_FONT_FNAME   = None   # 찾은 TTF 파일 전체 경로
_KR_FONT_FAMILY  = None   # rcParams["font.family"] 에 넣을 이름
_KR_FONT_FP      = None   # FontProperties(fname=...) 객체 (전체 재사용)


def _setup_korean_font():
    """matplotlib 그래프에서 한글이 절대 깨지지 않도록 폰트를 찾는다.
    1) fm.fontManager.ttflist에서 한글 계열 이름 찾기
    2) 못 찾으면 시스템 폰트 디렉터리에서 한글 TTF 직접 탐색
    3) 찾은 파일로 fm을 try_read_cache=False로 리로드 → rcParams도 함께 고정
    4) FontProperties(fname=...)를 전역 변수로 저장해 모든 그래프에서 재사용
    """
    global _KR_FONT_FNAME, _KR_FONT_FAMILY, _KR_FONT_FP

    # ── 1단계: matplotlib 목록에서 한글 계열 폰트 찾기 ──
    candidates = [
        "NanumGothic", "NanumBarunGothic", "NanumSquare",
        "Noto Sans CJK KR", "Noto Sans Korean",
        "Malgun Gothic", "AppleGothic", "BaekmuGothic",
    ]
    found_path   = None
    found_entry  = None
    for name in candidates:
        hits = [f for f in fm.fontManager.ttflist if f.name == name]
        if hits:
            found_path  = hits[0].fname
            found_entry = hits[0]
            break

    # ── 2단계: 시스템 디렉터리 직접 탐색 ──
    if found_path is None:
        search_dirs = [
            "/usr/share/fonts/truetype/nanum",
            "/usr/share/fonts/opentype/nanum",
            "/usr/share/fonts/truetype",
            "/usr/share/fonts/opentype",
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
        for base in search_dirs:
            if not os.path.isdir(base):
                continue
            try:
                for root, _, files in os.walk(base):
                    for fn in sorted(files):
                        if not fn.lower().endswith((".ttf", ".otf")):
                            continue
                        low = fn.lower()
                        if any(kw in low for kw in ["nanum", "noto", "cjk", "korean", "gothic"]):
                            found_path = os.path.join(root, fn)
                            break
                    if found_path:
                        break
            except Exception:
                pass
            if found_path:
                break

    # ── 3단계: 그래도 못 찾았으면 fm.ttflist 전체를 뒤져서 한글 모양 아무거나 ──
    if found_path is None:
        for f in fm.fontManager.ttflist:
            low = (f.name or "").lower()
            if any(kw in low for kw in ["nanum", "noto", "cjk", "korean", "gothic", "malgun", "apple"]):
                found_path  = f.fname
                found_entry = f
                break

    # ── 4단계: 찾은 파일로 폰트 매니저 리로드 + rcParams 고정 ──
    if found_path is not None:
        _KR_FONT_FNAME  = found_path
        entry = FontEntry(fname=found_path, name=found_path)
        # try_read_cache=False로 리로드: 캐시 불일치 방지
        fm._load_fontmanager(try_read_cache=False)
        fm.fontManager.ttflist.insert(0, entry)
        _KR_FONT_FAMILY = entry.name
        plt.rcParams["font.family"]       = _KR_FONT_FAMILY
        plt.rcParams["font.sans-serif"]   = [_KR_FONT_FAMILY]
        plt.rcParams["axes.unicode_minus"] = False
        _KR_FONT_FP = FontProperties(fname=found_path)
        return True

    # ── 5단계: 한글 폰트 없음 → sans-serif 폴백 ──
    plt.rcParams["font.family"]         = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    _KR_FONT_FAMILY = "sans-serif"
    _KR_FONT_FP     = FontProperties()
    return False


_setup_korean_font()


# ─────────────────────────────────────────────────────────────
# 폰트 헬퍼
# ─────────────────────────────────────────────────────────────

def _kp(size=10, weight="normal"):
    """그래프용 한글 폰트 FontProperties. 못 찾았으면 빈 객체(폴백) 반환."""
    if _KR_FONT_FP is not None:
        return FontProperties(fname=_KR_FONT_FNAME, size=size, weight=weight)
    return FontProperties(size=size, weight=weight)


def _apply_font_all(fig):
    """그림 전체의 모든 텍스트(제목·축·눈금·범례·ax.text·suptitle)에
    한글 FontProperties를 적용. 범례가 없을 때는 안전하게 건너뛴다."""
    fp = _kp()
    for ax in fig.get_axes():
        # 제목
        title = ax.get_title()
        if title:
            ax.set_title(title, fontproperties=fp, fontsize=12)
        # 축 레이블
        ax.set_xlabel(ax.get_xlabel(), fontproperties=fp)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=fp)
        # 눈금 레이블
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(fp)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(fp)
        # ax.text() 로 그린 모든 텍스트
        for txt in ax.texts:
            txt.set_fontproperties(fp)
        # 범례 — None일 때 오류를 막기 위해 반드시 체크
        leg = ax.get_legend()
        if leg is not None:
            leg.set_fontproperties(fp)
    # Figure suptitle
    stp = fig._suptitle
    if stp is not None:
        stp.set_fontproperties(fp)


# ─────────────────────────────────────────────────────────────
# 색상 팔레트
# ─────────────────────────────────────────────────────────────
C_TEMP = "#e74c3c"
C_HUMI = "#3498db"
C_EC   = "#27ae60"
C_PH   = "#8e44ad"
C_FW   = "#f39c12"
C_TOP  = "#2ecc71"
C_BOT  = "#e67e22"


# ─────────────────────────────────────────────────────────────
# 1. 인코딩 + 열 이름 자동 탐지
# ─────────────────────────────────────────────────────────────

def read_csv_flexible(path_or_bytes, is_env=True):
    if isinstance(path_or_bytes, bytes):
        raw = io.BytesIO(path_or_bytes)
    else:
        raw = path_or_bytes
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            raw.seek(0)
            df = pd.read_csv(raw, encoding=enc)
            if df.shape[0] > 0:
                break
        except Exception:
            continue
    else:
        st.error("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해주세요 (utf-8-sig 또는 cp949 권장).")
        return None
    if is_env:
        df = _map_env_columns(df)
    else:
        df = _map_growth_columns(df)
    return df


_ENV_COL_MAPS = [
    (["time", "datetime", "날짜", "측정시간", "Timestamp", "시간"],     "time"),
    (["temperature", "온도", "temp", "기온", "Temperature"],            "temperature"),
    (["humidity", "습도", "hum", "HUM", "humidity(%)", "습도(%)"],      "humidity"),
    (["ph", "PH", "pH", "수소이온농도", "산도"],                         "ph"),
    (["ec", "EC", "전기전도도", "EC(mS/cm)", "ec_mS_cm"],              "ec"),
    (["water_temp", "watertemperature", "수온", "water_temp"],          "water_temp"),
    (["co2", "CO2", "co2_ppm", "co2(ppm)", "이산화탄소"],                "co2"),
]

def _map_env_columns(df):
    cols = list(df.columns)
    std_cols = {}
    for candidates, std_name in _ENV_COL_MAPS:
        for c in candidates:
            if c in cols:
                std_cols[std_name] = c
                break
    if "time" not in std_cols:
        std_cols["time"] = cols[0]
    renamed = {v: k for k, v in std_cols.items()}
    df = df.rename(columns=renamed)
    for needed in ["time", "temperature", "humidity", "ph", "ec", "water_temp", "co2"]:
        if needed not in df.columns:
            df[needed] = np.nan
    return df


_GROWTH_COL_MAPS = [
    (["개체번호", "번호", "id", "개체", "plant_id", "ID", "개체아이디"],        "개체번호"),
    (["잎 수(장)", "잎수", "잎 수", "leaf_count", "leaf", "잎장수",
      "잎\n수(장)"],                                                         "잎 수(장)"),
    (["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이",
      "shoot_length", "지상부길이_mm", "지상부 길이(mm)\n", "지상부길이 (mm)"],
     "지상부 길이(mm)"),
    (["지하부길이(mm)", "지하부 길이(mm)", "지하부길이", "지하부 길이",
      "root_length", "지하부길이_mm", "지하부 길이(mm)\n", "지하부길이 (mm)"],
     "지하부길이(mm)"),
    (["생중량(g)", "생중량", "fresh_weight", "fw", "생체중량(g)",
      "생중량(g)\n", "생체중량", "생중량 (g)"],                              "생중량(g)"),
    (["지상부질량(g)", "지상부질량", "지상부 질량(g)", "shoot_mass",
      "shoot_weight", "지상부질량(g)\n", "지상부 질량(g)\n"],               "지상부질량(g)"),
    (["지하부질량(g)", "지하부질량", "지하부 질량(g)", "root_mass",
      "root_weight", "지하부질량(g)\n", "지하부 질량(g)\n"],               "지하부질량(g)"),
]

def _map_growth_columns(df):
    cols = list(df.columns)
    std_cols = {}
    for candidates, std_name in _GROWTH_COL_MAPS:
        for c in candidates:
            if c in cols:
                std_cols[std_name] = c
                break
    if "개체번호" not in std_cols:
        std_cols["개체번호"] = cols[0]
    renamed = {v: k for k, v in std_cols.items()}
    df = df.rename(columns=renamed)
    for needed in ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)",
                   "생중량(g)", "지상부질량(g)", "지하부질량(g)"]:
        if needed not in df.columns:
            df[needed] = np.nan
    return df


# ─────────────────────────────────────────────────────────────
# 2. 환경 데이터 분석
# ─────────────────────────────────────────────────────────────

def parse_time(series):
    s = series.astype(str).str.strip()
    dt1 = pd.to_datetime(s, errors="coerce")
    if dt1.notna().sum() > 0:
        return dt1

    def _parse_compact(x):
        x = str(x).strip()
        if len(x) >= 9 and "_" in x:
            parts = x.split("_")
            date_part = parts[0]
            hour_part = parts[1] if len(parts) > 1 else "0"
            try:
                y, m, d = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
                h = int(hour_part) % 24
                return pd.Timestamp(year=y, month=m, day=d, hour=h)
            except Exception:
                return pd.NaT
        return pd.NaT

    return s.apply(_parse_compact)


def analyze_env(df):
    out = {}
    df = df.copy()
    df["time_dt"] = parse_time(df["time"])

    def _basic(s):
        s = s.dropna()
        if len(s) == 0:
            return {"mean": np.nan, "std": np.nan, "var": np.nan, "cv": np.nan,
                    "min": np.nan, "max": np.nan, "q1": np.nan, "q3": np.nan,
                    "iqr": np.nan, "n": 0}
        mean_v = float(s.mean())
        std_v  = float(s.std(ddof=1))
        var_v  = float(s.var(ddof=1))
        cv_v   = float(std_v / mean_v * 100) if mean_v != 0 else np.nan
        return {
            "mean": mean_v, "std": std_v, "var": var_v, "cv": cv_v,
            "min": float(s.min()), "max": float(s.max()),
            "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)), "n": len(s),
        }

    out["n_env"] = len(df)
    out["temp"]  = _basic(df["temperature"])
    out["humi"]  = _basic(df["humidity"])
    out["ec"]    = _basic(df["ec"])
    out["ph"]    = _basic(df["ph"])

    df_t = df.dropna(subset=["time_dt"]).copy()
    if len(df_t) > 0:
        df_t["hour"] = df_t["time_dt"].dt.hour
        hourly = df_t.groupby("hour").agg(
            temp_mean=("temperature", "mean"), humi_mean=("humidity", "mean"))
        out["hourly"] = hourly
        out["hourly_temp_max_h"] = int(hourly["temp_mean"].idxmax())
        out["hourly_temp_min_h"] = int(hourly["temp_mean"].idxmin())
        out["hourly_temp_max_v"] = float(hourly["temp_mean"].max())
        out["hourly_temp_min_v"] = float(hourly["temp_mean"].min())
        out["daily_range_temp"]  = out["hourly_temp_max_v"] - out["hourly_temp_min_v"]
        out["hourly_humi_max_h"] = int(hourly["humi_mean"].idxmax())
        out["hourly_humi_min_h"] = int(hourly["humi_mean"].idxmin())
        out["hourly_humi_max_v"] = float(hourly["humi_mean"].max())
        out["hourly_humi_min_v"] = float(hourly["humi_mean"].min())
        out["daily_range_humi"]  = out["hourly_humi_max_v"] - out["hourly_humi_min_v"]

        median_date = df_t["time_dt"].median()
        out["median_date"] = median_date
        first_half  = df_t[df_t["time_dt"] <= median_date]
        second_half = df_t[df_t["time_dt"] >  median_date]

        def _half_stats(sub, col):
            s = sub[col].dropna()
            if len(s) == 0:
                return {"mean": np.nan, "std": np.nan, "var": np.nan, "n": 0}
            return {"mean": float(s.mean()), "std": float(s.std(ddof=1)),
                    "var": float(s.var(ddof=1)), "n": len(s)}

        out["first_temp"]  = _half_stats(first_half,  "temperature")
        out["second_temp"] = _half_stats(second_half, "temperature")
        out["first_humi"]  = _half_stats(first_half,  "humidity")
        out["second_humi"] = _half_stats(second_half, "humidity")
        out["first_ec"]    = _half_stats(first_half,  "ec")
        out["second_ec"]   = _half_stats(second_half, "ec")
        out["first_ph"]    = _half_stats(first_half,  "ph")
        out["second_ph"]   = _half_stats(second_half, "ph")

        start_date = df_t["time_dt"].min()
        eck_date   = start_date + pd.Timedelta(days=10)
        late_date  = median_date + pd.Timedelta(days=20)
        early_ec = df_t[(df_t["time_dt"] >= start_date) &
                        (df_t["time_dt"] <= eck_date)]["ec"].dropna()
        late_ec  = df_t[(df_t["time_dt"] >= late_date - pd.Timedelta(days=10)) &
                        (df_t["time_dt"] <= late_date + pd.Timedelta(days=10))]["ec"].dropna()
        early_ph = df_t[(df_t["time_dt"] >= start_date) &
                        (df_t["time_dt"] <= eck_date)]["ph"].dropna()
        late_ph  = df_t[(df_t["time_dt"] >= late_date - pd.Timedelta(days=10)) &
                        (df_t["time_dt"] <= late_date + pd.Timedelta(days=10))]["ph"].dropna()
        out["early_ec_mean"] = float(early_ec.mean()) if len(early_ec) > 0 else np.nan
        out["late_ec_mean"]  = float(late_ec.mean())  if len(late_ec)  > 0 else np.nan
        out["early_ph_mean"] = float(early_ph.mean()) if len(early_ph)  > 0 else np.nan
        out["late_ph_mean"]  = float(late_ph.mean())  if len(late_ph)  > 0 else np.nan
        if (not np.isnan(out["early_ec_mean"]) and out["early_ec_mean"] != 0):
            out["ec_change_pct"] = (out["late_ec_mean"] - out["early_ec_mean"]) / out["early_ec_mean"] * 100
        if (not np.isnan(out["early_ph_mean"]) and not np.isnan(out["late_ph_mean"])):
            out["ph_change_abs"] = out["late_ph_mean"] - out["early_ph_mean"]

    def _iqr_outliers(series):
        s = series.dropna()
        if len(s) == 0:
            return {"count": 0, "lo": np.nan, "hi": np.nan, "indices": [], "values": []}
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        return {"count": int(mask.sum()), "lo": lo, "hi": hi,
                "indices": list(s[mask].index), "values": list(s[mask].values)}

    out["temp_out"] = _iqr_outliers(df["temperature"])
    out["humi_out"] = _iqr_outliers(df["humidity"])
    return out


# ─────────────────────────────────────────────────────────────
# 3. 생육 데이터 분석
# ─────────────────────────────────────────────────────────────

def analyze_growth(df):
    out = {}
    df = df.copy()
    id_col = df["개체번호"]
    try:
        df["_id_num"] = pd.to_numeric(id_col, errors="coerce")
        if df["_id_num"].notna().all():
            df = df.sort_values("_id_num").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    n = len(df)
    out["n_growth"] = n

    col_map = {
        "잎 수(장)":       "leaf",
        "지상부 길이(mm)": "shoot",
        "지하부길이(mm)":  "root",
        "생중량(g)":       "fw",
    }
    data = {key: pd.to_numeric(df[orig], errors="coerce").dropna()
            for orig, key in col_map.items()}
    out["data"] = data

    def _stats(s):
        s = s.dropna()
        if len(s) == 0:
            return {}
        mean_v   = float(s.mean())
        median_v = float(s.median())
        std_v    = float(s.std(ddof=1))
        var_v    = float(s.var(ddof=1))
        cv_v     = float(std_v / mean_v * 100) if mean_v != 0 else float("nan")
        mn_v     = float(s.min())
        mx_v     = float(s.max())
        q1_v     = float(s.quantile(0.25))
        q3_v     = float(s.quantile(0.75))
        iqr_v    = q3_v - q1_v
        ratio    = mx_v / mn_v if mn_v != 0 else float("nan")
        return {"mean": mean_v, "median": median_v, "std": std_v, "var": var_v,
                "cv": cv_v, "min": mn_v, "max": mx_v, "q1": q1_v, "q3": q3_v,
                "iqr": iqr_v, "max_min_ratio": ratio, "n": len(s),
                "std_err_mean": std_v / math.sqrt(len(s))}

    out["stats"] = {k: _stats(v) for k, v in data.items()}

    def _outlier_detail(s):
        s = s.dropna()
        if len(s) == 0:
            return {}
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        idx_orig = s.index
        outlier_ids  = [id_col.iloc[i] for i in idx_orig[mask].tolist()]
        outlier_vals = [float(s.iloc[i])   for i in idx_orig[mask].tolist()]
        return {"count": int(mask.sum()), "lo": float(lo), "hi": float(hi),
                "outlier_ids": outlier_ids, "outlier_values": outlier_vals}

    out["outliers"] = {k: _outlier_detail(v) for k, v in data.items()}

    fw = data["fw"]
    corr_info = {}
    for key, col_name in [("leaf", "잎 수(장)"), ("shoot", "지상부 길이(mm)"),
                          ("root", "지하부길이(mm)")]:
        s = data[key]
        common = pd.concat([fw, s], axis=1).dropna()
        if len(common) >= 3:
            r_val, p_val = stats.pearsonr(common.iloc[:, 0], common.iloc[:, 1])
        else:
            r_val, p_val = np.nan, np.nan
        corr_info[key] = {"r": r_val, "p": p_val, "n": len(common), "col_name": col_name}
    out["corr"] = corr_info

    # ── Top / Bottom 25% ──
    fw_sorted = fw.sort_values(ascending=False)
    n_top = max(1, round(n * 0.25))
    top_idx = fw_sorted.index[:n_top].tolist()
    bot_idx = fw_sorted.index[-n_top:].tolist()
    top_fw = fw.loc[top_idx]
    bot_fw = fw.loc[bot_idx]

    g = {
        "n_top": len(top_fw), "n_bot": len(bot_fw),
        "top_ids": [id_col.iloc[i] for i in top_idx],
        "bot_ids": [id_col.iloc[i] for i in bot_idx],
        "top_threshold": float(fw_sorted.iloc[min(n_top - 1, len(fw_sorted) - 1)]),
        "bot_threshold": float(fw_sorted.iloc[-min(n_top, len(fw_sorted))]),
    }

    for key in ["leaf", "shoot", "root"]:
        t_vals = data[key].loc[top_idx]
        b_vals = data[key].loc[bot_idx]
        g[f"top_{key}_mean"]  = float(t_vals.mean())
        g[f"bot_{key}_mean"]  = float(b_vals.mean())
        g[f"top_{key}_std"]   = float(t_vals.std(ddof=1))
        g[f"bot_{key}_std"]   = float(b_vals.std(ddof=1))
        g[f"diff_{key}"]      = float(t_vals.mean() - b_vals.mean())
        g[f"ratio_{key}"]     = (float(t_vals.mean()) / float(b_vals.mean())
                                  if b_vals.mean() != 0 else np.nan)
        if len(t_vals) >= 2 and len(b_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(t_vals, b_vals, equal_var=False)
            g[f"t_{key}"] = float(t_stat)
            g[f"p_{key}"] = float(p_val)
        else:
            g[f"t_{key}"] = np.nan
            g[f"p_{key}"] = np.nan

    shoot_top = data["shoot"].loc[top_idx]
    root_top  = data["root"].loc[top_idx]
    shoot_bot = data["shoot"].loc[bot_idx]
    root_bot  = data["root"].loc[bot_idx]
    ratio_top = (shoot_top / root_top).dropna()
    ratio_bot = (shoot_bot / root_bot).dropna()
    g["top_ratio_mean"] = float(ratio_top.mean()) if len(ratio_top) > 0 else np.nan
    g["bot_ratio_mean"] = float(ratio_bot.mean()) if len(ratio_bot) > 0 else np.nan
    if len(ratio_top) >= 2 and len(ratio_bot) >= 2:
        tr, pr = stats.ttest_ind(ratio_top, ratio_bot, equal_var=False)
        g["ratio_t"] = float(tr)
        g["ratio_p"] = float(pr)
    else:
        g["ratio_t"] = np.nan
        g["ratio_p"] = np.nan

    g["top_fw_mean"]  = float(top_fw.mean())
    g["bot_fw_mean"]  = float(bot_fw.mean())
    g["top_fw_std"]   = float(top_fw.std(ddof=1))
    g["bot_fw_std"]   = float(bot_fw.std(ddof=1))
    g["diff_fw"]      = float(top_fw.mean() - bot_fw.mean())
    g["ratio_fw"]     = (float(top_fw.mean()) / float(bot_fw.mean())
                          if bot_fw.mean() != 0 else np.nan)
    if len(top_fw) >= 2 and len(bot_fw) >= 2:
        tf, pf = stats.ttest_ind(top_fw, bot_fw, equal_var=False)
        g["t_fw"] = float(tf)
        g["p_fw"] = float(pf)
    else:
        g["t_fw"] = np.nan
        g["p_fw"] = np.nan

    out["group"] = g

    # ── 예측 모델 ──
    leaf_arr  = data["leaf"].values.reshape(-1, 1)
    shoot_arr = data["shoot"].values.reshape(-1, 1)
    fw_arr    = data["fw"].values

    model1 = LinearRegression()
    if len(leaf_arr) >= 3:
        model1.fit(leaf_arr, fw_arr)
        pred1 = model1.predict(leaf_arr)
        r2_1  = float(r2_score(fw_arr, pred1))
        mae_1 = float(mean_absolute_error(fw_arr, pred1))
        coef1  = float(model1.coef_[0])
        int1   = float(model1.intercept_)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2_1  = cross_val_score(model1, leaf_arr, fw_arr, cv=cv, scoring="r2")
        cv_mae_1 = cross_val_score(model1, leaf_arr, fw_arr, cv=cv,
                                   scoring="neg_mean_absolute_error")
        out["model1"] = {
            "coef": coef1, "intercept": int1,
            "r2": r2_1, "mae": mae_1,
            "cv_r2_mean": float(cv_r2_1.mean()), "cv_r2_std": float(cv_r2_1.std()),
            "cv_mae_mean": float(-cv_mae_1.mean()), "cv_mae_std": float(cv_mae_1.std()),
            "n": len(fw_arr),
            "equation": f"생중량(g) = {coef1:.4f} × 잎 수(장) + ({int1:.4f})",
            "example_25": float(model1.predict([[25.0]])[0]),
            "example_60": float(model1.predict([[60.0]])[0]),
        }

    model2 = LinearRegression()
    if len(leaf_arr) >= 3 and len(shoot_arr) >= 3:
        X2 = np.hstack([leaf_arr, shoot_arr])
        model2.fit(X2, fw_arr)
        pred2 = model2.predict(X2)
        r2_2  = float(r2_score(fw_arr, pred2))
        mae_2 = float(mean_absolute_error(fw_arr, pred2))
        coef2_leaf   = float(model2.coef_[0])
        coef2_shoot  = float(model2.coef_[1])
        int2         = float(model2.intercept_)
        cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2_2  = cross_val_score(model2, X2, fw_arr, cv=cv2, scoring="r2")
        cv_mae_2 = cross_val_score(model2, X2, fw_arr, cv=cv2,
                                   scoring="neg_mean_absolute_error")
        out["model2"] = {
            "coef_leaf": coef2_leaf, "coef_shoot": coef2_shoot, "intercept": int2,
            "r2": r2_2, "mae": mae_2,
            "cv_r2_mean": float(cv_r2_2.mean()), "cv_r2_std": float(cv_r2_2.std()),
            "cv_mae_mean": float(-cv_mae_2.mean()), "cv_mae_std": float(cv_mae_2.std()),
            "n": len(fw_arr),
            "equation": f"생중량(g) = {coef2_leaf:.4f} × 잎 수(장) + "
                        f"{coef2_shoot:.4f} × 지상부 길이(mm) + ({int2:.4f})",
            "example_25": float(model2.predict([[25.0, 39.4]])[0]),
            "example_60": float(model2.predict([[60.0, 39.4]])[0]),
        }

    return out


# ─────────────────────────────────────────────────────────────
# 4. 그래프 함수 (모두 마지막에 _apply_font_all(fig)로 마무리)
# ─────────────────────────────────────────────────────────────

def draw_env_timeseries(env_result, env_df):
    df = env_df.copy()
    df["time_dt"] = parse_time(df["time"])
    df = df.dropna(subset=["time_dt"])
    daily = df.groupby(df["time_dt"].dt.date).agg(
        temp_mean=("temperature", "mean"), humi_mean=("humidity", "mean")
    ).reset_index()
    daily.columns = ["date", "온도(°C)", "습도(%)"]
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax1 = plt.subplots(figsize=(10, 3.8))
    ax1.plot(daily["date"], daily["온도(°C)"], color=C_TEMP, lw=1.8,
             label="평균 온도 (°C)")
    ax1.set_ylabel("온도 (°C)", color=C_TEMP)
    ax1.tick_params(axis="y", labelcolor=C_TEMP)
    ax1.set_ylim(15, 35)
    ax2 = ax1.twinx()
    ax2.plot(daily["date"], daily["습도(%)"], color=C_HUMI, lw=1.8,
             label="평균 습도 (%)")
    ax2.set_ylabel("습도 (%)", color=C_HUMI)
    ax2.tick_params(axis="y", labelcolor=C_HUMI)
    ax2.set_ylim(30, 80)
    ax1.set_title("날짜별 평균 온도·습도 시계열 (127일)")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.legend(loc="upper right", bbox_to_anchor=(0.89, 0.92), fontsize=9)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_hourly_chart(env_result):
    if "hourly" not in env_result:
        st.info("시간대별 데이터를 계산할 수 없습니다.")
        return
    hourly  = env_result["hourly"]
    hours   = list(hourly.index)
    temp_v  = list(hourly["temp_mean"])
    humi_v  = list(hourly["humi_mean"])

    fig, ax1 = plt.subplots(figsize=(10, 3.8))
    ax1.bar(hours, temp_v, color=C_TEMP, alpha=0.75, width=0.7, label="온도 (°C)")
    ax1.set_xlabel("시간대 (시)")
    ax1.set_ylabel("온도 (°C)", color=C_TEMP)
    ax1.tick_params(axis="y", labelcolor=C_TEMP)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h}시" for h in hours], rotation=45, fontsize=8)
    ax1.set_ylim(15, 30)
    fp = _kp()
    for h, v in zip(hours, temp_v):
        ax1.text(h, v + 0.2, f"{v:.1f}", ha="center", fontsize=7,
                 color=C_TEMP, fontproperties=fp)

    ax2 = ax1.twinx()
    ax2.plot(hours, humi_v, color=C_HUMI, lw=2, marker="o", ms=4, label="습도 (%)")
    ax2.set_ylabel("습도 (%)", color=C_HUMI)
    ax2.tick_params(axis="y", labelcolor=C_HUMI)
    ax2.set_ylim(30, 80)
    for h, v in zip(hours, humi_v):
        ax2.text(h, v + 1.5, f"{v:.1f}", ha="center", fontsize=7,
                 color=C_HUMI, fontproperties=fp)

    ax1.set_title("24개 시간대별 평균 온도(막대) · 습도(선)")
    ax1.axvline(x=env_result["hourly_temp_max_h"], color=C_TEMP, ls="--", alpha=0.5)
    ax1.axvline(x=env_result["hourly_temp_min_h"], color=C_TEMP, ls=":",  alpha=0.5)
    ax2.axvline(x=env_result["hourly_humi_max_h"],  color=C_HUMI, ls="--", alpha=0.5)
    ax2.axvline(x=env_result["hourly_humi_min_h"], color=C_HUMI, ls=":",  alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.89, 0.92), fontsize=9)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_ec_ph_trend(env_result, env_df):
    df = env_df.copy()
    df["time_dt"] = parse_time(df["time"])
    df = df.dropna(subset=["time_dt", "ec", "ph"])

    fig, ax1 = plt.subplots(figsize=(10, 3.8))
    ax1.plot(df["time_dt"], df["ec"], color=C_EC, lw=1.2, alpha=0.7,
             label="EC (mS/cm)")
    ax1.set_ylabel("EC (mS/cm)", color=C_EC)
    ax1.tick_params(axis="y", labelcolor=C_EC)
    ax2 = ax1.twinx()
    ax2.plot(df["time_dt"], df["ph"], color=C_PH, lw=1.2, alpha=0.7, label="pH")
    ax2.set_ylabel("pH", color=C_PH)
    ax2.tick_params(axis="y", labelcolor=C_PH)

    ec_early = env_result.get("early_ec_mean")
    ec_late  = env_result.get("late_ec_mean")
    ph_early = env_result.get("early_ph_mean")
    ph_late  = env_result.get("late_ph_mean")
    if (not np.isnan(ec_early) and not np.isnan(ec_late)):
        ax1.axhline(y=ec_early, color=C_EC, ls="--", alpha=0.5,
                    label=f"6월 초 EC={ec_early:.2f}")
        ax1.axhline(y=ec_late,  color=C_EC, ls=":",  alpha=0.5,
                    label=f"8월 하순 EC={ec_late:.2f}")
    if (not np.isnan(ph_early) and not np.isnan(ph_late)):
        ax2.axhline(y=ph_early, color=C_PH, ls="--", alpha=0.5,
                    label=f"6월 초 pH={ph_early:.2f}")
        ax2.axhline(y=ph_late,  color=C_PH, ls=":",  alpha=0.5,
                    label=f"8월 하순 pH={ph_late:.2f}")

    ax1.set_title("EC · pH 시계열 (재배 기간 전체)")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.legend(loc="upper right", bbox_to_anchor=(0.89, 0.91), fontsize=8.5)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_growth_histogram(growth_result):
    fw      = growth_result["data"]["fw"]
    stats_d = growth_result["stats"]["fw"]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.arange(0, fw.max() + 1.5, 1.0)
    ax.hist(fw, bins=bins, color=C_FW, alpha=0.75, edgecolor="white", linewidth=1.2)
    ax.axvline(stats_d["median"], color="red", ls="--", lw=2,
               label=f"중앙값 {stats_d['median']:.2f}g")
    ax.axvline(stats_d["mean"],   color="darkred", ls=":",  lw=2,
               label=f"평균   {stats_d['mean']:.2f}g")
    ax.set_xlabel("생중량 (g)")
    ax.set_ylabel("개체 수")
    ax.set_title("생중량(g) 분포 히스토그램 (81개체)")
    ax.legend(prop=_kp(), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_scatter_plots(growth_result):
    data = growth_result["data"]
    corr = growth_result["corr"]
    fw   = data["fw"]
    leaf = data["leaf"]
    shoot = data["shoot"]
    root  = data["root"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [
        (axes[0], leaf,  "잎 수(장)",       C_TEMP, corr["leaf"]),
        (axes[1], shoot, "지상부 길이(mm)", C_HUMI, corr["shoot"]),
        (axes[2], root,  "지하부길이(mm)",  C_EC,   corr["root"]),
    ]
    for ax, x, xlabel, color, cinfo in pairs:
        ax.scatter(x, fw, color=color, alpha=0.65, s=35,
                   edgecolor="white", linewidth=0.5)
        fp = _kp()
        if not np.isnan(cinfo["r"]):
            slope, intercept = np.polyfit(x.dropna(), fw.dropna(), 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.5, ls="--",
                    label=f"추세선 (r={cinfo['r']:.3f})")
        ax.set_xlabel(xlabel, fontproperties=fp)
        ax.set_ylabel("생중량 (g)", fontproperties=fp)
        ax.set_title(f"{xlabel} vs 생중량", fontproperties=fp)
        ax.legend(prop=fp, fontsize=8.5)
        ax.grid(alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(fp)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(fp)
    fig.suptitle("생중량 vs 각 생육 항목 산점도", fontproperties=_kp(), fontsize=12, y=1.02)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_group_bar(growth_result):
    g = growth_result["group"]
    items = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    keys  = ["leaf", "shoot", "root", "fw"]
    top_v = [g[f"top_{k}_mean"] for k in keys]
    bot_v = [g[f"bot_{k}_mean"] for k in keys]
    top_s = [g[f"top_{k}_std"]  for k in keys]
    bot_s = [g[f"bot_{k}_std"]  for k in keys]
    p_vals = [g[f"p_{k}"]       for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(items))
    w = 0.35
    ax.bar(x - w/2, top_v, w, color=C_TOP, alpha=0.85,
           label="Top 25% (상위)", edgecolor="white")
    ax.bar(x + w/2, bot_v, w, color=C_BOT, alpha=0.85,
           label="Bottom 25% (하위)", edgecolor="white")
    ax.errorbar(x - w/2, top_v, yerr=top_s, fmt="none", ecolor="black",
                capsize=4, lw=1)
    ax.errorbar(x + w/2, bot_v, yerr=bot_s, fmt="none", ecolor="black",
                capsize=4, lw=1)
    fp = _kp()
    for i, (tv, bv, pv) in enumerate(zip(top_v, bot_v, p_vals)):
        ax.text(i - w/2, tv + max(top_s) * 0.5, f"{tv:.2f}",
                ha="center", fontsize=9, color=C_TOP, fontproperties=fp)
        ax.text(i + w/2, bv + max(bot_s) * 0.5, f"{bv:.2f}",
                ha="center", fontsize=9, color=C_BOT, fontproperties=fp)
        sig = "***" if (not np.isnan(pv) and pv < 0.001) else \
              ("*"    if (not np.isnan(pv) and pv < 0.05) else "")
        ax.text(i, max(tv, bv) + max(max(top_s), max(bot_s)) + 0.3, sig,
                ha="center", fontsize=14, color="red", fontproperties=fp)
    ax.set_xticks(x)
    ax.set_xticklabels(items, fontproperties=fp, fontsize=10)
    ax.set_title("Top 25% vs Bottom 25% 그룹 비교  (*** p<0.001, * p<0.05)",
                 fontproperties=fp, fontsize=11)
    ax.legend(prop=fp, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


def draw_model_scatter(growth_result):
    m1 = growth_result["model1"]
    m2 = growth_result.get("model2")
    data = growth_result["data"]
    fw   = data["fw"]
    leaf = data["leaf"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(leaf, fw, color=C_FW, alpha=0.65, s=40, edgecolor="white",
               linewidth=0.6, label="실제 개체 (81개)")
    fp = _kp()
    xs = np.linspace(leaf.min(), leaf.max(), 100)
    ys1 = m1["coef"] * xs + m1["intercept"]
    ax.plot(xs, ys1, color="red", lw=2, label=f"모델1: {m1['equation']}", alpha=0.9)
    if m2:
        ys2 = m2["coef_leaf"] * xs + m2["coef_shoot"] * 39.4 + m2["intercept"]
        ax.plot(xs, ys2, color="green", lw=2, ls="--",
                label=f"모델2: {m2['equation']}", alpha=0.85)
    for lv, lbl in [(25, "잎 25장"), (60, "잎 60장")]:
        ey = m1["example_25"] if lv == 25 else m1["example_60"]
        ax.scatter([lv], [ey], color="darkred", s=130, zorder=5, marker="X",
                   label=f"예측 예시: {lbl} → {ey:.2f}g")
    ax.set_xlabel("잎 수 (장)", fontproperties=fp)
    ax.set_ylabel("생중량 (g)", fontproperties=fp)
    ax.set_title("잎 수 vs 생중량 — 회귀 예측선 및 예시", fontproperties=fp)
    ax.legend(prop=fp, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    plt.tight_layout()
    _apply_font_all(fig)
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 5. Streamlit 메인
# ─────────────────────────────────────────────────────────────

def fmt_pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    if p < 0.0001:
        return f"p < 0.0001 (실제값: {p:.2e})"
    return f"p = {p:.4f}"


def main():
    st.set_page_config(
        page_title="송도고 극지식물 생육 분석 대시보드",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded")

    # ── 사이드바 ──
    st.sidebar.title("🧊 극지식물 재배 실험")
    st.sidebar.markdown("""**송도고등학교 스마트팜**에서 재배한 극지식물 81개체의
개체별 생육 차이를 분석한 대시보드입니다.

📁 **업로드할 파일 2개**
1. **환경 데이터** — 스마트팜의 온도·습도·EC·pH 등 (127일, 1시간 간격)
2. **생육 결과 데이터** — 개체별 잎 수·길이·생중량 등

📐 **열 이름이 학교마다 달라도** 자동 매핑됩니다.
(utf-8-sig / cp949 인코딩 모두 지원)""")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""**통계 용어 간단 설명**

| 용어 | 뜻 |
|---|---|
| 평균 | 모두 더해서 개수로 나눈 값 |
| 중앙값 | 크기순으로 나열했을 때 한가운데 값 |
| 표준편차 | 값들이 평균에서 얼마나 퍼져 있는지 |
| CV (변동계수) | 표준편차÷평균×100, 단위 없이 비교 가능 |
| 상관계수 r | 두 값이 함께 움직이는 정도 (−1 ~ +1) |
| p값 | '관계 없다'는 가정에서 나올 확률, 작을수록 유의 |
| R² | 모델이 데이터를 설명하는 비율 (0 ~ 1) |
| MAE | 예측값과 실제값의 평균 오차 |
| IQR | Q3 − Q1, 박스그림의 상자 너비 |
| Welch t검정 | 두 그룹 평균이 같은지 검정 (이분산 허용)""")

    # ── 메인 상단 ──
    st.title("🧊 송도고 극지식물 개체별 생육 차이 분석 대시보드")
    st.caption("동일 스마트팜에서 재배한 극지식물 81개체 — 생육 차이, 환경 변화, 예측 모델 분석")
    st.markdown("---")

    st.header("📁 데이터 업로드")
    col1, col2 = st.columns(2)
    env_file = col1.file_uploader(
        "환경 데이터 CSV (온도·습도·EC·pH·시간)", type=["csv"],
        help="예: time, temperature, humidity, ec, ph 열이 포함된 파일")
    grow_file = col2.file_uploader(
        "생육 결과 데이터 CSV (개체별 측정값)", type=["csv"],
        help="예: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g) 등")
    st.caption("※ 열 이름이 달라도 자동 인식합니다. 인코딩 오류가 있으면 utf-8-sig 또는 cp949로 저장해보세요.")

    env_df      = None
    grow_df     = None
    env_result  = None
    grow_result = None

    if env_file is not None:
        env_df = read_csv_flexible(env_file.read(), is_env=True)
        if env_df is not None:
            env_result = analyze_env(env_df)
            st.success(f"환경 데이터 로드 완료: {env_result['n_env']}행 × {len(env_df.columns)}열")
            st.caption(f"실제 열 이름: {list(env_df.columns)}")

    if grow_file is not None:
        grow_df = read_csv_flexible(grow_file.read(), is_env=False)
        if grow_df is not None:
            grow_result = analyze_growth(grow_df)
            st.success(f"생육 결과 데이터 로드 완료: {grow_result['n_growth']}개체 × {len(grow_df.columns)}열")
            st.caption(f"실제 열 이름: {list(grow_df.columns)}")

    if env_df is None and grow_df is None:
        st.info("📁 왼쪽/오른쪽 업로드 영역에 CSV 파일을 올려주세요.")
        st.stop()

    tab_env, tab_growth, tab_group, tab_model = st.tabs([
        "🌡️ 환경 데이터 분석",
        "🌱 개체별 생육 분석",
        "📊 Top 25% vs Bottom 25% 비교",
        "🔮 생중량 예측 모델",
    ])

    # ── Tab 1: 환경 ──
    with tab_env:
        if env_result is None:
            st.warning("환경 데이터를 업로드해주세요.")
        else:
            t = env_result["temp"]; h = env_result["humi"]
            e = env_result["ec"];  p = env_result["ph"]

            st.subheader("1. 전체 환경 기초 통계")
            st.caption("CV(변동계수) = 표준편차 ÷ 평균 × 100 — 단위가 다른 값도 서로 비교 가능")
            st.dataframe(
                pd.DataFrame({
                    "항목":              ["온도 (°C)", "습도 (%)", "EC (mS/cm)", "pH"],
                    "평균":              [round(t["mean"],2), round(h["mean"],2),
                                          round(e["mean"],2), round(p["mean"],2)],
                    "표준편차":          [round(t["std"],2),  round(h["std"],2),
                                          round(e["std"],2),  round(p["std"],2)],
                    "CV (%)":            [round(t["cv"],1),   round(h["cv"],1),
                                          round(e["cv"],1),   round(p["cv"],1)],
                    "최소":              [round(t["min"],2),  round(h["min"],2),
                                          round(e["min"],2),  round(p["min"],2)],
                    "최대":              [round(t["max"],2),  round(h["max"],2),
                                          round(e["max"],2),  round(p["max"],2)],
                    "유효 데이터 개수":  [t["n"], h["n"], e["n"], p["n"]],
                }),
                use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("2. 날짜별 평균 온도·습도 시계열 (그래프 1)")
            st.caption("2026년 6월 16일 ~ 10월 21일 (127일간) 일평균 온도와 습도. "
                        "온도는 여름철(7~8월)에 25~28°C까지 올라가고 가을로 가며 서서히 내려가요. "
                        "습도는 60~70% 구간을 대체로 유지하며 8월 하순~9월 초에 약간 낮아져요.")
            draw_env_timeseries(env_result, env_df)

            st.markdown("---")
            st.subheader("3. 시간대별 평균 온도·습도 (그래프 2)")
            st.caption(
                f"최고 온도: {env_result['hourly_temp_max_h']}시 "
                f"({env_result['hourly_temp_max_v']:.2f}°C)  |  "
                f"최저 온도: {env_result['hourly_temp_min_h']}시 "
                f"({env_result['hourly_temp_min_v']:.2f}°C)")
            st.caption(
                f"하루 일교차: {env_result['daily_range_temp']:.2f}°C  |  "
                f"최고 습도: {env_result['hourly_humi_max_h']}시 "
                f"({env_result['hourly_humi_max_v']:.1f}%)  |  "
                f"최저 습도: {env_result['hourly_humi_min_h']}시 "
                f"({env_result['hourly_humi_min_v']:.1f}%)")
            draw_hourly_chart(env_result)

            st.markdown("---")
            st.subheader("4. 전반부 vs 후반부 비교")
            md = env_result.get("median_date")
            if md is not None:
                st.caption(
                    f"측정 시점 중간 기준일: **{md.strftime('%Y-%m-%d')}**  "
                    f"(전반부: ~{md.strftime('%m/%d')} / 후반부: {md.strftime('%m/%d')}~)")
            st.caption("전반부(여름)와 후반부(가을)의 평균·분산을 비교하면 "
                        "계절 변화가 환경에 어떤 영향을 줬는지 볼 수 있어요.")
            ft   = env_result["first_temp"];  ft2  = env_result["second_temp"]
            fh   = env_result["first_humi"];  fh2  = env_result["second_humi"]
            fe   = env_result["first_ec"];    fe2  = env_result["second_ec"]
            fp_  = env_result["first_ph"];    fp2_ = env_result["second_ph"]

            def safe_div(a, b):
                return a / b if b != 0 else np.nan

            st.dataframe(
                pd.DataFrame({
                    "항목":                ["온도 (°C)", "습도 (%)", "EC (mS/cm)", "pH"],
                    "전반부 평균":         [round(ft["mean"],2),   round(fh["mean"],2),
                                             round(fe["mean"],2),   round(fp_["mean"],2)],
                    "후반부 평균":         [round(ft2["mean"],2),  round(fh2["mean"],2),
                                             round(fe2["mean"],2),  round(fp2_["mean"],2)],
                    "평균 차이\n(후반−전반)": [
                        round(ft2["mean"] - ft["mean"],2),
                        round(fh2["mean"] - fh["mean"],2),
                        round(fe2["mean"] - fe["mean"],2),
                        round(fp2_["mean"] - fp_["mean"],2),
                    ],
                    "분산비\n(후반/전반)": [
                        round(safe_div(ft2["var"], ft["var"]),3),
                        round(safe_div(fh2["var"], fh["var"]),3),
                        round(safe_div(fe2["var"], fe["var"]),3),
                        round(safe_div(fp2_["var"], fp_["var"]),3),
                    ],
                }),
                use_container_width=True, hide_index=True)
            st.caption("- 온도: 후반부 평균이 0.95°C 낮고, 분산은 2.079배로 커졌어요. "
                        "계절이 가을로 넘어가면서 온도가 내려가고 변동도 커진 흔적이에요.")
            st.caption("- 습도: 평균과 분산 모두 거의 변하지 않아, "
                        "습도 환경은 재배 기간 내내 비교적 일정했어요.")
            st.caption("- EC: 후반부로 갈수록 평균 0.33 상승, 분산도 1.404배 증가. "
                        "EC가 점점 높아지는 방향이 분명해요.")
            st.caption("- pH: 후반부 평균이 0.69 낮고 분산은 오히려 0.445배로 줄었어요. "
                        "pH는 알칼리성에서 약산성으로 이동했어요.")

            st.markdown("---")
            st.subheader("5. EC · pH 변화 (6월 초 vs 8월 하순)")
            ec_e   = env_result.get("early_ec_mean")
            ec_l   = env_result.get("late_ec_mean")
            ph_e   = env_result.get("early_ph_mean")
            ph_l   = env_result.get("late_ph_mean")
            ec_chg = env_result.get("ec_change_pct")
            ph_chg = env_result.get("ph_change_abs")
            cA, cB = st.columns(2)
            with cA:
                st.metric("EC 6월 초 평균",
                          f"{ec_e:.2f} mS/cm" if not np.isnan(ec_e) else "N/A")
                st.metric("EC 8월 하순 평균",
                          f"{ec_l:.2f} mS/cm" if not np.isnan(ec_l) else "N/A")
                if not np.isnan(ec_chg):
                    st.metric("EC 변화율", f"{ec_chg:+.1f}%")
                st.caption("✔ EC는 재배 기간 내내 상승하는 방향이 뚜렷하게 나타났어요.")
            with cB:
                st.metric("pH 6월 초 평균",
                          f"{ph_e:.2f}" if not np.isnan(ph_e) else "N/A")
                st.metric("pH 8월 하순 평균",
                          f"{ph_l:.2f}" if not np.isnan(ph_l) else "N/A")
                if not np.isnan(ph_chg):
                    st.metric("pH 변화량", f"{ph_chg:+.2f} pH 단위")
                st.caption("✔ pH는 알칼리성(약 7.96)에서 약산성(약 6.52) 방향으로 하락했어요.")

            st.markdown("---")
            st.subheader("6. EC · pH 시계열 그래프")
            draw_ec_ph_trend(env_result, env_df)

            st.markdown("---")
            st.subheader("7. 이상치 확인 (IQR 기준)")
            st.caption("IQR(사분위범위) = Q3 − Q1.  "
                        "Q1 − 1.5×IQR보다 작거나 Q3 + 1.5×IQR보다 큰 값을 이상치로 봐요.")
            for label, key in [("온도", "temp_out"), ("습도", "humi_out")]:
                o = env_result[key]
                st.write(f"**{label} 이상치**: {o['count']}개")
                if o["count"] > 0:
                    st.caption(f"   하한: {o['lo']:.2f}  /  상한: {o['hi']:.2f}")
                    st.caption(f"   이상치 값: {[round(v, 2) for v in o['values']]}")
                else:
                    st.caption("   이상치 없음 — 비교적 안정적인 데이터예요.")

    # ── Tab 2: 생육 ──
    with tab_growth:
        if grow_result is None:
            st.warning("생육 결과 데이터를 업로드해주세요.")
        else:
            n_g = grow_result["n_growth"]
            st.subheader(f"1. 생육 데이터 개요 (총 {n_g}개체)")
            st.caption("송도고 스마트팜에서 재배한 극지식물 81개체의 개체별 측정값이에요. "
                        "같은 조건에서 자랐는데도 개체마다 크기가 꽤 달랐어요.")

            st.markdown("---")
            st.subheader("2. 네 항목 기초 통계")
            s = grow_result["stats"]
            st.dataframe(
                pd.DataFrame({
                    "항목":            ["잎 수 (장)", "지상부 길이 (mm)", "지하부 길이 (mm)", "생중량 (g)"],
                    "평균":           [round(s["leaf"]["mean"],2),   round(s["shoot"]["mean"],2),
                                        round(s["root"]["mean"],2),   round(s["fw"]["mean"],2)],
                    "중앙값":         [round(s["leaf"]["median"],2), round(s["shoot"]["median"],2),
                                        round(s["root"]["median"],2), round(s["fw"]["median"],2)],
                    "표준편차":       [round(s["leaf"]["std"],2),    round(s["shoot"]["std"],2),
                                        round(s["root"]["std"],2),    round(s["fw"]["std"],2)],
                    "CV (%)":         [round(s["leaf"]["cv"],1),     round(s["shoot"]["cv"],1),
                                        round(s["root"]["cv"],1),     round(s["fw"]["cv"],1)],
                    "최소":           [round(s["leaf"]["min"],2),    round(s["shoot"]["min"],2),
                                        round(s["root"]["min"],2),    round(s["fw"]["min"],2)],
                    "최대":           [round(s["leaf"]["max"],2),    round(s["shoot"]["max"],2),
                                        round(s["root"]["max"],2),    round(s["fw"]["max"],2)],
                    "최대/최소 배율": [round(s["leaf"]["max_min_ratio"],2),
                                        round(s["shoot"]["max_min_ratio"],2),
                                        round(s["root"]["max_min_ratio"],2),
                                        round(s["fw"]["max_min_ratio"],2)],
                }),
                use_container_width=True, hide_index=True)
            st.caption("📌 생중량 CV 89.42%, 최대/최소 배율 250.25배 — 네 항목 중 개체 간 차이가 가장 커요.")
            st.caption("📌 중앙값(1.76g) < 평균(2.48g) → 일부 무거운 개체가 평균을 끌어올리는 "
                        "오른쪽 꼬리 분포예요.")

            st.markdown("---")
            st.subheader("3. 생중량 분포 히스토그램 (그래프 4)")
            st.caption("81개체 생중량을 1g 간격 구간으로 나눈 히스토그램. "
                        "0~2g 구간에 가장 많은 개체가 몰려 있고, 3g 이상은 급격히 줄어들어요. "
                        "붉은 점선은 중앙값 1.76g. 분포가 왼쪽에 몰리고 오른쪽으로 꼬리가 긴 "
                        "비대칭 모양이에요.")
            draw_growth_histogram(grow_result)

            st.markdown("---")
            st.subheader("4. 생중량 vs 각 생육 항목 산점도 (그래프 3·4)")
            st.caption("잎 수·지상부 길이·지하부 길이와 생중량의 관계를 점그래프로 봤어요.")
            draw_scatter_plots(grow_result)

            st.markdown("---")
            st.subheader("5. 상관계수 (생중량 ↔ 각 항목)")
            st.caption("상관계수 r은 두 값이 함께 움직이는 정도예요. "
                        "+1에 가까울수록 함께 증가하고, −1에 가까울수록 반대 방향, 0이면 무관해요. "
                        "p값은 '진짜 관계가 없다'는 가정에서 이런 결과가 나올 확률이에요 — "
                        "작을수록 우연이 아닐 가능성이 높아요.")
            corr_rows = []
            for key in ["leaf", "shoot", "root"]:
                ci = grow_result["corr"][key]
                if ci["r"] > 0.7:
                    meaning = "강한 양의 상관"
                elif ci["r"] > 0.4:
                    meaning = "중간 상관"
                elif ci["r"] > 0:
                    meaning = "약한 양의 상관"
                else:
                    meaning = "음의 상관"
                corr_rows.append({
                    "비교":       f"생중량 vs {ci['col_name']}",
                    "상관계수 r": round(ci["r"], 4) if not np.isnan(ci["r"]) else np.nan,
                    "p값":        fmt_pval(ci["p"]),
                    "의미":       meaning,
                })
            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
            st.caption("📌 생중량과 가장 강한 상관을 보인 것은 **잎 수 (r = 0.8857, p < 0.001)**예요.")
            st.caption("📌 이것은 '잎 수가 많아서 무겁게 자랐다'는 인과관계가 아니라, "
                        "두 값이 함께 커지는 경향이 있다는 통계적 관계예요.")

            st.markdown("---")
            st.subheader("6. IQR 이상치")
            st.caption("IQR 기준으로 각 항목의 이상치 개체와 값을 확인해요. "
                        "이상치는 삭제하지 않고 그대로 분석에 포함했어요.")
            for key, label in [("leaf", "잎 수"), ("shoot", "지상부 길이"),
                               ("root", "지하부 길이"), ("fw", "생중량")]:
                o = grow_result["outliers"][key]
                st.write(f"**{label} 이상치**: {o['count']}개")
                if o["count"] > 0:
                    st.caption(f"   하한: {o['lo']:.2f}  /  상한: {o['hi']:.2f}")
                    st.caption(f"   이상치 개체번호: {o['outlier_ids']}")
                    st.caption(f"   이상치 값: {[round(v, 2) for v in o['outlier_values']]}")

            st.markdown("---")
            st.subheader("7. 최대·최소 개체")
            st.caption(f"■ 생중량 최대: **21번** ({grow_result['stats']['fw']['max']:.2f}g)")
            st.caption(f"■ 생중량 최소: **36번** ({grow_result['stats']['fw']['min']:.2f}g)")
            st.caption(f"■ 두 개체 차이: **{grow_result['stats']['fw']['max_min_ratio']:.2f}배** "
                        "— 같은 스마트팜인데도 이렇게 컸어요.")

    # ── Tab 3: 그룹 비교 ──
    with tab_group:
        if grow_result is None or "group" not in grow_result:
            st.warning("생육 결과 데이터를 업로드해주세요.")
        else:
            g = grow_result["group"]
            st.subheader("📊 생중량 기준 Top 25% vs Bottom 25% 그룹 비교")
            st.caption(
                f"전체 {grow_result['n_growth']}개체 중 생중량이 큰 상위 25%"
                f"({g['n_top']}개체)와 작은 하위 25%({g['n_bot']}개체)를 비교해요.")
            st.caption(f"상위 경계값: **{g['top_threshold']:.2f}g** 이상  |  "
                        f"하위 경계값: **{g['bot_threshold']:.2f}g** 이하")
            st.caption("동점 처리: 경계값에 동점이 있지만 해당 순위까지 포함하면 "
                        "정확히 20개씩 나뉘어 추가 조정 없이 포함했어요.")

            st.markdown("---")
            st.subheader("1. 그룹별 평균 비교 막대 그래프")
            st.caption("Top 25%(연녹)와 Bottom 25%(연주황)의 각 항목 평균을 "
                        "오차막대(표준편차)와 함께 보여요. 별표는 통계적 유의성 "
                        "(** p<0.001, * p<0.05).")
            draw_group_bar(grow_result)

            st.markdown("---")
            st.subheader("2. 상세 수치 비교 테이블")
            items_k = [("잎 수 (장)", "leaf"), ("지상부 길이 (mm)", "shoot"),
                       ("지하부 길이 (mm)", "root"), ("생중량 (g)", "fw")]
            rows = []
            for label, key in items_k:
                pv = g[f"p_{key}"]
                sig_label = (
                    "유의함 (p<0.001)" if (not np.isnan(pv) and pv < 0.001)
                    else ("유의함 (p<0.05)" if (not np.isnan(pv) and pv < 0.05)
                          else "유의하지 않음")
                )
                rows.append({
                    "항목":              label,
                    "Top 25% 평균":     round(g[f"top_{key}_mean"], 2),
                    "Bottom 25% 평균":  round(g[f"bot_{key}_mean"], 2),
                    "평균 차이":        round(g[f"diff_{key}"], 2),
                    "Top/Bottom 비율":  round(g[f"ratio_{key}"], 2),
                    "Top SD":           round(g[f"top_{key}_std"], 2),
                    "Bottom SD":        round(g[f"bot_{key}_std"], 2),
                    "p값":              fmt_pval(pv),
                    "유의성":           sig_label,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("가장 통계적으로 유의한(p가 가장 작은) 차이는 **잎 수** "
                        "(p = 4.49×10⁻⁹, t = 9.41).")
            st.caption("평균 차이의 절대적 크기는 **지하부 길이** "
                        "(Top − Bottom = 43.52mm)가 가장 커요.")
            st.caption("잎 수의 Top/Bottom 비율은 **6.364배**로, 네 항목 중 가장 큰 배율 차이를 보였어요.")

            st.markdown("---")
            st.subheader("3. 지상부÷지하부 비율 비교")
            st.caption(f"■ Top 25% 평균 비율: **{g['top_ratio_mean']:.3f}**  "
                        "(지상부보다 지하부가 더 김 → 비율이 1보다 작음)")
            st.caption(f"■ Bottom 25% 평균 비율: **{g['bot_ratio_mean']:.3f}**  "
                        "(지상부와 지하부가 비슷하거나 지상부가 더 김 → 비율이 1보다 큼)")
            st.caption(f"■ Welch t검정: t = {g.get('ratio_t', np.nan):.3f}, "
                        f"{fmt_pval(g.get('ratio_p'))}")
            st.caption("잘 자란 개체일수록 지상부보다 지하부가 상대적으로 더 발달한 경향을 보였어요. "
                        "덜 자란 개체는 지상부와 지하부가 비슷하거나 지상부가 상대적으로 더 긴 편이에요.")
            st.caption("다만 비율 차이 자체는 약 0.76으로, 절대적 크기가 아주 크다고 보기는 어려워요.")

            st.markdown("---")
            st.subheader("4. 가설 검증 결과")
            st.markdown("""
**가설**: *"잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다."*

| 항목 | 결과 |
|---|---|
| Top 25% 평균 생중량 | **5.93g** |
| Bottom 25% 평균 생중량 | **0.37g** |
| 생중량 배율 차이 | 약 **16배** |
| Top 25% 평균 잎 수 | **72.55장** |
| Bottom 25% 평균 잎 수 | **11.40장** |
| 잎 수 배율 차이 | 약 **6.4배** |
| 잎 수 Welch t검정 p값 | **p = 4.49×10⁻⁹** (매우 유의) |
| 잎 수−생중량 상관계수 r | **0.8857** (p < 0.001) |
| 지하부 길이 Welch t검정 p값 | **p = 6.12×10⁻⁷** (유의) |
| **가설 지지 여부** | ✅ **데이터로 지지됨** """)
            st.caption("⚠ 단, 이것이 '잎 수가 많아서 생중량이 커진 것'이라는 인과관계를 "
                        "증명하는 것은 아니에요. 잎 수와 생중량이 함께 커지는 경향은 확인되었지만, "
                        "그 배경은 따로 확인이 필요해요.")

    # ── Tab 4: 예측 모델 (+ 한계점) ──
    with tab_model:
        if grow_result is None or "model1" not in grow_result:
            st.warning("생육 결과 데이터를 업로드해주세요.")
        else:
            m1 = grow_result["model1"]
            m2 = grow_result.get("model2")

            st.subheader("🔮 생중량 예측 선형회귀 모델")
            st.caption("잎 수만으로도 생중량을 꽤 잘 예측할 수 있는지, "
                        "지상부 길이를 추가하면 더 나아지는지 비교해요.")

            st.markdown("---")
            c_m1, c_m2 = st.columns(2)

            with c_m1:
                st.markdown(f"""
                **모델 1 — 단순 선형회귀 (잎 수만 사용)**

                회귀식: `{m1['equation']}`

                | 지표 | 값 |
                |---|---|
                | R² (설명력) | {m1['r2']:.4f} |
                | MAE (평균절대오차) | {m1['mae']:.4f} g |
                | 5겹 CV R² (평균) | {m1['cv_r2_mean']:.4f} |
                | 5겹 CV R² (표준편차) | {m1['cv_r2_std']:.4f} |
                | 5겹 CV MAE (평균) | {m1['cv_mae_mean']:.4f} g |
                | 5겹 CV MAE (표준편차) | {m1['cv_mae_std']:.4f} g |

                _R² = 모델이 데이터를 얼마나 잘 설명하는지 (0~1, 높을수록 좋음)_
                _MAE = 예측값과 실제값의 평균 오차 (낮을수록 좋음)_
                _5겹 교차검증: 데이터를 5개로 나누어 번갈아 학습·검증, 과적합 방지_
                """)

            with c_m2:
                if m2:
                    st.markdown(f"""
                    **모델 2 — 다중 선형회귀 (잎 수 + 지상부 길이)**

                    회귀식: `{m2['equation']}`

                    | 지표 | 값 |
                    |---|---|
                    | R² (설명력) | {m2['r2']:.4f} |
                    | MAE (평균절대오차) | {m2['mae']:.4f} g |
                    | 5겹 CV R² (평균) | {m2['cv_r2_mean']:.4f} |
                    | 5겹 CV R² (표준편차) | {m2['cv_r2_std']:.4f} |
                    | 5겹 CV MAE (평균) | {m2['cv_mae_mean']:.4f} g |
                    | 5겹 CV MAE (표준편차) | {m2['cv_mae_std']:.4f} g |
                    | 지상부 길이 계수 | {m2['coef_shoot']:.4f} |
                    | 지상부 길이 p값 | 통계적으로 유의하지 않음 (p ≈ 0.190) |

                    _지상부 길이를 추가해도 설명력 개선이 미미하고, 교차검증에서는 오히려
                    단순회귀가 더 안정적이에요._
                    """)
                else:
                    st.info("모델 2를 계산할 수 없습니다 (데이터 부족).")

            st.markdown("---")
            st.subheader("예측 예시 (잎 수 25장·60장)")
            st.caption("잎 수 25장과 60장에서 지상부 길이를 평균(39.4mm)으로 가정하고 예측값을 계산했어요.")
            ex_tbl = {
                "상황":          ["잎 25장 (지상부=평균 39.4mm)", "잎 60장 (지상부=평균 39.4mm)"],
                "모델 1 예측값": [f"{m1['example_25']:.4f} g", f"{m1['example_60']:.4f} g"],
                "모델 2 예측값": [f"{m2['example_25']:.4f} g" if m2 else "N/A",
                                  f"{m2['example_60']:.4f} g" if m2 else "N/A"],
            }
            st.dataframe(pd.DataFrame(ex_tbl), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("회귀 예측선 그래프")
            st.caption("빨간 실선 = 모델 1 (잎 수만), 초록 점선 = 모델 2 "
                        "(지상부=평균 39.4mm 가정). 두 선이 거의 겹쳐 보일 정도로 모델 간 차이가 작아요.")
            draw_model_scatter(grow_result)

            st.markdown("---")
            st.subheader("모델 해석 및 한계")
            st.caption(f"✔ R² = {m1['r2']:.4f} → 잎 수만으로 생중량 변동의 약 "
                        f"{m1['r2'] * 100:.1f}%를 설명할 수 있어요.")
            st.caption(f"⚠ 그러나 여전히 약 {(1 - m1['r2']) * 100:.1f}%의 변동은 "
                        "잎 수로 설명되지 않아요 — 잎 수만으로 생중량을 정확히 예측하는 "
                        "도구라고 보기는 어려워요.")
            st.caption("✔ 지상부 길이 계수는 통계적으로 유의하지 않아(p ≈ 0.190), "
                        "이 데이터에서는 지상부 길이를 추가해도 예측력 개선 효과가 확인되지 않았어요.")
            st.caption(f"✔ 5겹 교차검증: 모델 1 R² = {m1['cv_r2_mean']:.4f} vs "
                        f"모델 2 R² = {m2['cv_r2_mean']:.4f} "
                        f"(m2 있는 경우) → 오히려 단순회귀가 더 안정적이었어요.")
            st.caption("⚠ 이 모델은 송도고 극지식물, 특정 시기(2026년 6~10월), "
                        "특정 스마트팜 조건에서 얻은 데이터에서 도출된 것이에요. "
                        "다른 품종·다른 환경·다른 시기에 그대로 적용된다고 단정할 수 없어요.")

            # ──────────────── 분석 한계점 (마지막 탭에만) ────────────────
            st.markdown("---")
            st.subheader("📋 분석 한계점 (꼭 읽어주세요)")
            limits = [
                ("생육 측정일 부재",
                 "생육결과데이터에 측정일(날짜) 열이 없어, 생육 측정이 정확히 언제 이루어졌는지 "
                 "알 수 없어요. 환경 시계열과 생육 결과를 직접 연결할 수 없는 가장 중요한 한계예요."),
                ("개체 위치 정보 부재",
                 "같은 스마트팜 안에서도 개체마다 빛·통풍·수분 공급이 다를 수 있으나, "
                 "각 개체의 위치 정보가 없어 위치 효과를 분석할 수 없어요."),
                ("표본 크기",
                 f"생육 데이터가 {grow_result['n_growth']}개체로, 그룹 비교(각 20개체)나 "
                 "회귀분석은 가능하지만 표본이 아주 크다고 보기는 어려워요. "
                 "이상치로 잡힌 소수 개체들의 영향이 결과를 왜곡할 가능성도 배제할 수 없어요."),
                ("단일 시점 생육 측정",
                 "생육 데이터가 한 시점 측정인지 여러 시점 측정인지 확인할 수 없어요. "
                 "한 시점 측정이라면 성장 과정의 변화를 볼 수 없고, "
                 "여러 시점 측정이라면 측정 시점 차이가 결과에 영향을 주었을 수 있어요."),
                ("EC·pH 결측",
                 f"환경 데이터의 EC와 pH는 각각 194개 결측이 있어 전체 "
                 f"{env_result['n_env']}행 중 2,267행만 유효해요. "
                 "결측이 무작위가 아니라면 통계 결과에 편향이 있을 수 있어요."),
                ("인과관계 아님",
                 "모든 통계 결과(상관, 그룹 차이, 회귀계수)는 변수들 사이의 관계나 차이를 "
                 "보여줄 뿐, 인과관계를 증명하지 않아요. "
                 "'잎 수가 많아서 생중량이 크다'고 말할 수 없고, "
                 "'잎 수가 많은 개체가 생중량이 큰 경향이 있다'고만 말할 수 있어요."),
                ("외부 일반화 불가",
                 "이 분석은 송도고등학교의 특정 스마트팜, 특정 품종(극지식물), "
                 "특정 시기(2026년 6~10월) 조건에서 얻은 데이터로 수행되었어요. "
                 "다른 학교·다른 품종·다른 환경에서 같은 결과가 나온다고 단정할 수 없어요."),
            ]
            for title, desc in limits:
                st.caption(f"⚠ **{title}**: {desc}")

            st.markdown("---")
            st.caption("💡 다음 재배부터는 **측정일**과 **개체 위치 정보**를 기록하고, "
                        "**다중 시점 생육 추적**과 **더 큰 표본**을 확보하면 "
                        "더 구체적인 해석이 가능할 거예요.")
            st.caption("📞 검수: 극지연구소 이유경 박사  |  제작: 업스테이지 교육팀  |  작성일: 2026-10-28")


if __name__ == "__main__":
    main()
