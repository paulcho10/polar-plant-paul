import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="극지식물 생육 분석", page_icon="🌱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

FONT = "Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"
C_TOP, C_BOT, C_MID = "#27ae60", "#e67e22", "#b2bec3"
C_TEMP, C_HUMI, C_EC, C_PH = "#e74c3c", "#3498db", "#16a085", "#8e44ad"
TOP, BOT, MID = "상위 25%", "하위 25%", "중간"
ALL_ITEMS = ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
UNIT = {"잎 수": "장", "지상부 길이": "mm", "지하부 길이": "mm",
        "생중량": "g", "지상부 질량": "g", "지하부 질량": "g", "잎당 생중량": "g/장"}


# ───────────── 열 이름 표준화 · CSV 읽기 ─────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


@st.cache_data(show_spinner=False)
def load_df(data):
    df = pd.DataFrame()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except Exception:
            pass
    if len(df.columns) == 0:
        return df
    return std_cols(df)


def school_name(*files):
    for f in files:
        if f is not None:
            n = re.split(r"[_\s.\-]", f.name)[0].strip()
            if n:
                return n
    return "우리 학교"


# ───────────── 작은 도우미 함수 ─────────────
def has(df, *cols):
    return df is not None and len(df) > 0 and all(c in df.columns for c in cols)


def vals(df, col):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "계산 불가"
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def fmt_id(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(v)
    except Exception:
        return str(v)


def style(fig, title, h=430):
    fig.update_layout(title=title, height=h, font=dict(family=FONT),
                      margin=dict(l=50, r=50, t=70, b=50),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def table(df):
    st.dataframe(df, width="stretch", hide_index=True)


def why(text):
    st.info("🧭 **이 분석이 가설에 필요한 이유**  \n" + text)


def legend(items):
    st.markdown("**📖 그래프 읽는 법**\n\n" + "\n".join(f"- {i}" for i in items))


def describe(s):
    m = s.mean()
    sd = s.std(ddof=1) if len(s) > 1 else np.nan
    return {"평균": m, "중앙값": s.median(), "표준편차": sd,
            "CV": (sd / m * 100 if m else np.nan), "최소": s.min(), "최대": s.max(), "n": len(s)}


def dual_axis(x, y1, n1, c1, y2, n2, c2, title, bar=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if bar:
        fig.add_trace(go.Bar(x=x, y=y1, name=n1, marker_color=c1, opacity=0.75), secondary_y=False)
        fig.update_yaxes(range=[float(np.nanmin(y1)) - 1, float(np.nanmax(y1)) + 0.5], secondary_y=False)
    else:
        fig.add_trace(go.Scatter(x=x, y=y1, name=n1, mode="lines", line=dict(color=c1, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=y2, name=n2, mode="lines+markers" if bar else "lines",
                             line=dict(color=c2, width=2)), secondary_y=True)
    fig.update_yaxes(title_text=n1, secondary_y=False)
    fig.update_yaxes(title_text=n2, secondary_y=True)
    return style(fig, title)


def hist_fig(s, name, unit):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=15, marker_color="#f39c12", opacity=0.8, name="개체 수"))
    fig.add_vline(x=float(s.mean()), line_dash="dot", line_color="darkred",
                  annotation_text=f"평균 {s.mean():.2f}", annotation_position="top right")
    fig.add_vline(x=float(s.median()), line_dash="dash", line_color="red",
                  annotation_text=f"중앙값 {s.median():.2f}", annotation_position="top left")
    fig.update_xaxes(title_text=f"{name} ({unit})")
    fig.update_yaxes(title_text="개체 수")
    return style(fig, f"{name} 분포", 360)


def box_fig(d, col):
    fig = go.Figure()
    for g, color in [(BOT, C_BOT), (TOP, C_TOP)]:
        fig.add_trace(go.Box(y=vals(d[d["그룹"] == g], col), name=g, boxpoints="all", jitter=0.4,
                             pointpos=0, boxmean=True, marker_color=color, line_color=color))
    fig.update_yaxes(title_text=f"{col} ({UNIT.get(col, '')})")
    return style(fig, f"{col}: 하위 25% vs 상위 25%", 400)


def scatter_fig(d, xcols, ycol, title):
    fig = make_subplots(rows=1, cols=len(xcols), subplot_titles=[f"{x} vs {ycol}" for x in xcols])
    for i, x in enumerate(xcols, start=1):
        sub = d[[x, ycol] + (["그룹"] if "그룹" in d.columns else [])].copy()
        sub[x] = pd.to_numeric(sub[x], errors="coerce")
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub.dropna(subset=[x, ycol])
        if "그룹" in sub.columns:
            for g, color in [(TOP, C_TOP), (MID, C_MID), (BOT, C_BOT)]:
                s2 = sub[sub["그룹"] == g]
                fig.add_trace(go.Scatter(x=s2[x], y=s2[ycol], mode="markers", name=g, legendgroup=g,
                                         showlegend=(i == 1), marker=dict(color=color, size=8, opacity=0.8)),
                              row=1, col=i)
        else:
            fig.add_trace(go.Scatter(x=sub[x], y=sub[ycol], mode="markers", name="개체", showlegend=(i == 1),
                                     marker=dict(color=C_MID, size=8)), row=1, col=i)
        if len(sub) >= 3 and sub[x].nunique() > 1:
            a, b = np.polyfit(sub[x], sub[ycol], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines", name="추세선",
                                     showlegend=(i == 1), line=dict(color="black", dash="dash")),
                          row=1, col=i)
        fig.update_xaxes(title_text=f"{x} ({UNIT.get(x, '')})", row=1, col=i)
        fig.update_yaxes(title_text=f"{ycol} ({UNIT.get(ycol, '')})", row=1, col=i)
    return style(fig, title, 420)


# ───────────── 분석 함수 ─────────────
def make_groups(gd):
    if not has(gd, "잎 수", "생중량"):
        return None
    d = gd.copy()
    d["잎 수"] = pd.to_numeric(d["잎 수"], errors="coerce")
    d["생중량"] = pd.to_numeric(d["생중량"], errors="coerce")
    d = d.dropna(subset=["잎 수", "생중량"]).reset_index(drop=True)
    d["잎당 생중량"] = (d["생중량"] / d["잎 수"].where(d["잎 수"] > 0)).round(4)  # 잎 한 장이 평균 몇 g인가
    n = len(d)
    if n < 8:
        return None
    k = int(np.ceil(n * 0.25))
    order = d["잎 수"].sort_values(kind="mergesort").index
    d["그룹"] = MID
    d.loc[order[:k], "그룹"] = BOT
    d.loc[order[-k:], "그룹"] = TOP
    return d, k


def key_results(gd):
    res = make_groups(gd)
    if res is None:
        return None
    d, k = res
    if d["잎 수"].nunique() < 2 or d["생중량"].nunique() < 2:
        return None
    a = d.loc[d["그룹"] == TOP, "생중량"]
    b = d.loc[d["그룹"] == BOT, "생중량"]
    _, p = stats.ttest_ind(a, b, equal_var=False)
    try:
        _, pm = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        pm = np.nan
    pa = d.loc[d["그룹"] == TOP, "잎당 생중량"].dropna()
    pb = d.loc[d["그룹"] == BOT, "잎당 생중량"].dropna()
    pl_p = stats.ttest_ind(pa, pb, equal_var=False)[1] if len(pa) > 1 and len(pb) > 1 else np.nan
    r, pr = stats.pearsonr(d["잎 수"], d["생중량"])
    rho, _ = stats.spearmanr(d["잎 수"], d["생중량"])
    return {"d": d, "k": k, "n": len(d), "top_fw": a.mean(), "bot_fw": b.mean(),
            "top_leaf": d.loc[d["그룹"] == TOP, "잎 수"].mean(), "bot_leaf": d.loc[d["그룹"] == BOT, "잎 수"].mean(),
            "pl_top": pa.mean(), "pl_bot": pb.mean(), "pl_p": pl_p, "p": p, "pm": pm, "r": r, "pr": pr, "rho": rho,
            "supported": bool(a.mean() > b.mean() and p < 0.05 and (np.isnan(pm) or pm < 0.05))}


def compare_rows(d, items):
    rows = []
    for c in items:
        if c not in d.columns:
            continue
        a = vals(d[d["그룹"] == TOP], c)
        b = vals(d[d["그룹"] == BOT], c)
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        try:
            _, pm = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            pm = np.nan
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"항목": f"{c} ({UNIT.get(c, '')})", "상위 25% 평균": round(a.mean(), 2),
                     "하위 25% 평균": round(b.mean(), 2), "평균 차이": round(a.mean() - b.mean(), 2),
                     "배율(상위÷하위)": round(a.mean() / b.mean(), 2) if b.mean() != 0 else np.nan,
                     "상위 표준편차": round(a.std(ddof=1), 2), "하위 표준편차": round(b.std(ddof=1), 2),
                     "Welch t검정": fp(p), "Mann-Whitney": fp(pm),
                     "효과크기 d": round((a.mean() - b.mean()) / sp, 2) if sp > 0 else np.nan,
                     "유의함?": "예" if p < 0.05 else "아니오"})
    return rows


def fit_line(x, y):
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2, float(np.abs(y - pred).mean())


def cv_scores(x, y, k=5):
    idx = np.random.RandomState(42).permutation(len(x))
    r2s, maes = [], []
    for fold in np.array_split(idx, k):
        train = np.setdiff1d(idx, fold)
        a, b = np.polyfit(x[train], y[train], 1)
        pred = a * x[fold] + b
        ss_tot = float(((y[fold] - y[fold].mean()) ** 2).sum())
        if ss_tot > 0:
            r2s.append(1 - float(((y[fold] - pred) ** 2).sum()) / ss_tot)
        maes.append(float(np.abs(y[fold] - pred).mean()))
    return float(np.mean(r2s)) if r2s else np.nan, float(np.mean(maes))


# ───────────── 업로드 · 데이터 읽기 (한 번만) ─────────────
st.sidebar.header("📁 데이터 올리기")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

env_df, growth_df = None, None
with st.spinner("데이터를 읽는 중이에요..."):
    if env_file is not None:
        env_df = load_df(env_file.getvalue())
    if growth_file is not None:
        growth_df = load_df(growth_file.getvalue())

avail = [c for c in ALL_ITEMS if growth_df is not None and c in growth_df.columns]
st.sidebar.markdown("---")
sel = st.sidebar.multiselect("생육 측정 항목 선택", avail,
                             default=[c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in avail])
for c in ("잎 수", "생중량"):  # 가설에 꼭 필요한 두 항목은 항상 포함
    if c in avail and c not in sel:
        sel.append(c)
school = school_name(env_file, growth_file)

st.title("송도고 극지식물 개체별 생육 차이 분석 대시보드")
st.success("📌 **연구 가설** : 잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다.")
st.caption(f"🏫 {school} · 같은 스마트팜에서 자란 극지식물 개체끼리 비교 · 위 가설을 데이터로 검증합니다")
if env_file is None and growth_file is None:
    st.info("👈 왼쪽 사이드바에서 환경 CSV와 생육 CSV를 올려주세요. 한 파일만 올려도 해당 탭은 동작해요.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 문제정의 · 가설", "② 실험방법", "③ 결과 1 · 환경과 개체 차이",
    "④ 결과 2 · 가설 검정과 예측", "⑤ 논의 · 한계점 · 결론"])


# ───────────── 탭1 ─────────────
def tab_problem():
    st.header("1. 문제 정의")
    st.markdown("""
같은 스마트팜, 같은 물과 빛에서 키워도 **어떤 개체는 크게, 어떤 개체는 작게** 자랍니다.
이 차이를 눈으로 크기를 재지 않고도 **잎 수라는 쉬운 측정값**으로 알아낼 수 있다면,
수확량을 미리 예상하고 잘 안 자라는 개체를 일찍 찾아낼 수 있어요.
""")
    c1, c2 = st.columns(2)
    c1.success("**내 가설**  \n잎 수 상위 25% 개체는 하위 25% 개체보다 생중량이 클 것이다.")
    c2.warning("**귀무가설(반대 가정)**  \n두 그룹의 평균 생중량은 같고, 차이는 우연일 뿐이다.  \n→ 데이터로 이 가정을 **깨뜨릴 수 있는지** 확인합니다.")

    st.subheader("가설로 가는 길 (분석 순서와 이유)")
    st.markdown("""
| 단계 | 하는 일 | 가설에 도움이 되는 이유 |
|---|---|---|
| ① 환경 확인 | 온도·습도·EC·pH 변화 보기 | 환경이 크게 흔들렸다면 개체 차이를 '환경 탓'으로 의심해야 해요. 안정적이면 개체 자체의 차이로 볼 근거가 생겨요. |
| ② 개체 차이 확인 | 생육 값의 평균·표준편차·분포 | 개체들이 거의 똑같이 자랐다면 비교할 의미가 없어요. **차이가 실제로 큰지** 먼저 확인해요. |
| ③ 관계 확인 | 잎 수와 생중량의 상관·산점도 | 잎 수가 늘 때 생중량도 함께 늘어나는지 **전체 개체**로 확인해요. |
| ④ 가설 검정 | 잎 수 상위 25% vs 하위 25%의 생중량 비교 | 가설을 **직접** 시험하는 단계예요. 차이가 우연인지 검정으로 판단해요. |
| ⑤ 예측 | 잎 수로 생중량을 예측하는 직선 | 차이가 '얼마나' 큰지, 실제로 쓸 만한지 숫자로 보여줘요. |
""")



def tab_method():
    st.header("2. 실험 방법")
    st.subheader("2.1 사용한 데이터")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**환경 데이터**")
        if has(env_df):
            st.write(f"- {len(env_df):,}행 · 열: {', '.join(map(str, env_df.columns))}")
            if has(env_df, "time") and env_df["time"].notna().sum() > 0:
                st.write(f"- 기간: {env_df['time'].min():%Y-%m-%d} ~ {env_df['time'].max():%Y-%m-%d}")
        else:
            st.info("환경 CSV를 올리면 여기에 요약이 나와요.")
    with cB:
        st.markdown("**생육 결과 데이터**")
        if has(growth_df):
            st.write(f"- {len(growth_df)}개체 · 분석 항목: {', '.join(avail)}")
            st.caption("개체번호는 그래프 이름표로만 쓰고, 계산에는 넣지 않아요.")
        else:
            st.info("생육 CSV를 올리면 여기에 요약이 나와요.")

    st.subheader("2.2 분석 방법")
    st.markdown("""
1. **그룹 나누기**: 개체를 잎 수 순서로 줄 세워 **잎 수가 가장 적은 25%(하위)** 와 **가장 많은 25%(상위)** 로 나눠요. 가운데 50%는 비교에서 빼요.
   (잎 수로 나눠야 가설과 맞아요. 생중량으로 나누고 생중량을 비교하면 '큰 것끼리 크다'는 당연한 결과가 나와요.)
2. **비교하기**: 두 그룹의 생중량 평균을 비교하고, 그 차이가 우연인지 **Welch t검정**과 **Mann-Whitney 검정**으로 확인해요.
3. **이상치**: 극단값은 지우지 않고 그대로 분석해요. 대신 순위 기반 검정(Mann-Whitney, Spearman)으로 영향을 확인해요.
4. **예측**: 잎 수로 생중량을 예측하는 직선(단순 회귀)을 만들고, 데이터를 5조각으로 나눠 **교차검증**해요.
""")
    with st.expander("📚 통계 용어 짧은 설명"):
        st.markdown("""
- **평균 / 중앙값**: 모두 더해 개수로 나눈 값 / 크기순으로 세웠을 때 한가운데 값
- **표준편차(SD)**: 값들이 평균에서 평균적으로 얼마나 떨어져 있나 (클수록 들쭉날쭉)
- **CV(변동계수)**: 표준편차 ÷ 평균 × 100. 단위가 달라도 흩어진 정도를 비교할 수 있어요.
- **IQR / 이상치**: 가운데 50% 범위의 폭. 그 범위를 1.5배 넘게 벗어난 값을 이상치로 봐요.
- **상관계수 r**: 두 값이 함께 움직이는 정도 (−1 ~ +1). Spearman은 순위로 계산해 극단값에 덜 흔들려요.
- **p값**: '차이가 없다'고 가정했을 때 이런 결과가 나올 확률. 0.05보다 작으면 우연으로 보기 어렵다고 판단해요.
- **효과크기 d**: 평균 차이가 흩어진 정도의 몇 배인가. 0.8 이상이면 큰 차이로 봐요.
- **R²**: 모델이 생중량 변동의 몇 %를 설명하나 (0~1). **MAE**: 예측과 실제의 평균 오차(g).
- **교차검증**: 데이터를 나눠 일부로 만들고 나머지로 시험해 '새 개체에도 통하는지' 확인하는 방법
""")


# ───────────── 탭2 ─────────────
def env_section(ed):
    st.subheader("3.1 스마트팜 환경 (먼저 '같은 환경이었나' 확인)")
    why("개체끼리 생중량이 달랐을 때 원인이 환경이라면 잎 수와의 관계를 믿기 어려워요. 환경이 얼마나 일정했는지 먼저 확인합니다.")
    if not has(ed):
        st.info("환경 CSV를 올리면 이 부분이 표시돼요.")
        return
    cols = [c for c in ["temperature", "humidity", "ec", "ph"] if c in ed.columns]
    names = {"temperature": "온도 (°C)", "humidity": "습도 (%)", "ec": "EC (mS/cm)", "ph": "pH"}
    rows = []
    for c in cols:
        s = vals(ed, c)
        if len(s) < 2:
            continue
        d = describe(s)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        rows.append({"항목": names[c], "평균": round(d["평균"], 2), "표준편차": round(d["표준편차"], 2),
                     "CV(%)": round(d["CV"], 1), "최소": round(d["최소"], 2), "최대": round(d["최대"], 2),
                     "이상치 개수": out, "유효 데이터": d["n"], "빈 값(결측)": int(ed[c].isna().sum())})
    if rows:
        table(pd.DataFrame(rows))
        st.caption("CV가 작을수록 환경이 일정했다는 뜻이에요. 이상치는 센서 순간 튐일 수 있어 지우지 않고 표시만 했어요.")
    else:
        st.info("온도·습도·EC·pH 열을 찾지 못해 기초 통계는 건너뛰었어요.")
    if not (has(ed, "time") and pd.api.types.is_datetime64_any_dtype(ed["time"]) and ed["time"].notna().sum() > 1):
        st.info("시간 열을 읽지 못해 시계열 그래프는 건너뛰었어요.")
        return
    t = ed.dropna(subset=["time"]).copy()
    t["date"] = t["time"].dt.normalize()
    t["hour"] = t["time"].dt.hour

    st.markdown("##### 그래프 1. 날짜별 평균 온도·습도")
    if "temperature" in cols and "humidity" in cols:
        daily = t.groupby("date")[["temperature", "humidity"]].mean()
        show(dual_axis(daily.index, daily["temperature"], "평균 온도 (°C)", C_TEMP,
                       daily["humidity"], "평균 습도 (%)", C_HUMI, "날짜별 평균 온도·습도"))
        legend(["🔴 **빨간 선(왼쪽 축)**: 그날 하루 동안 잰 온도의 평균", "🔵 **파란 선(오른쪽 축)**: 그날 하루 습도의 평균",
                "선이 평평할수록 환경이 안정적이었다는 뜻이고, 갑자기 솟거나 꺼진 구간은 센서 오류나 환경 변화 후보예요.",
                "하루치를 평균 낸 값이라 하루 안의 오르내림은 보이지 않아요(그래프 2에서 확인)."])
    else:
        st.info("온도·습도 열이 모두 있어야 이 그래프를 그릴 수 있어요.")

    st.markdown("##### 그래프 2. 시간대별 평균 온도·습도")
    if "temperature" in cols and "humidity" in cols:
        hr = t.groupby("hour")[["temperature", "humidity"]].mean()
        show(dual_axis(hr.index, hr["temperature"], "온도 (°C)", C_TEMP, hr["humidity"], "습도 (%)", C_HUMI,
                       "하루 24시간 평균 온도(막대)·습도(선)", bar=True))
        m1, m2, m3 = st.columns(3)
        m1.metric("평균 온도 최고 시각", f"{int(hr['temperature'].idxmax())}시", f"{hr['temperature'].max():.2f}°C", delta_color="off")
        m2.metric("평균 온도 최저 시각", f"{int(hr['temperature'].idxmin())}시", f"{hr['temperature'].min():.2f}°C", delta_color="off")
        m3.metric("하루 일교차(평균 기준)", f"{hr['temperature'].max() - hr['temperature'].min():.2f}°C")
        legend(["🟥 **막대**: 그 시각(0~23시)의 평균 온도. 막대 세로축을 0부터가 아니라 잘라서 차이가 크게 보여요.",
                "🔵 **파란 선·점**: 그 시각의 평균 습도. 보통 온도와 반대로 움직여요.",
                "막대 높이 차이가 작으면 하루 중 온도 변화가 작다는 뜻이에요."])
    else:
        st.info("온도·습도 열이 모두 있어야 이 그래프를 그릴 수 있어요.")

    st.markdown("##### 그래프 3. EC · pH 변화")
    if "ec" in cols and "ph" in cols:
        d2 = t.groupby("date")[["ec", "ph"]].mean().dropna(how="all")
        show(dual_axis(d2.index, d2["ec"], "EC (mS/cm)", C_EC, d2["ph"], "pH", C_PH, "날짜별 평균 EC·pH"))
        d0, d1 = t["time"].min(), t["time"].max()
        early = t[t["time"] <= d0 + pd.Timedelta(days=10)]
        late = t[t["time"] >= d1 - pd.Timedelta(days=10)]
        e0, e1, p0, p1 = vals(early, "ec").mean(), vals(late, "ec").mean(), vals(early, "ph").mean(), vals(late, "ph").mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("EC 처음 10일 평균", f"{e0:.2f}")
        k2.metric("EC 마지막 10일 평균", f"{e1:.2f}", f"{e1 - e0:+.2f}")
        k3.metric("pH 처음 10일 평균", f"{p0:.2f}")
        k4.metric("pH 마지막 10일 평균", f"{p1:.2f}", f"{p1 - p0:+.2f}")
        legend(["🟢 **초록 선(왼쪽 축)**: 양액(영양액)의 EC. 값이 클수록 녹아 있는 영양분이 많다는 뜻이에요.",
                "🟣 **보라 선(오른쪽 축)**: 양액의 pH. 7보다 작으면 산성, 크면 알칼리성이에요.",
                "선이 한 방향으로 계속 움직이면 '재배 기간 동안 양액 조건이 바뀌었다'는 뜻이에요.",
                "아래 숫자 카드는 재배 시작 10일과 끝 10일의 평균 비교예요."])
    else:
        st.info("EC·pH 열이 없어 이 그래프는 건너뛰었어요.")


def growth_section(gd, kr):
    st.subheader("3.2 개체별 생육 차이 (비교할 만큼 차이가 큰가?)")
    why("모든 개체가 비슷하게 자랐다면 상위·하위를 나누는 의미가 없어요. 개체 간 차이가 실제로 큰지 확인합니다.")
    if not has(gd):
        st.info("생육 CSV를 올리면 이 부분이 표시돼요.")
        return
    rows = []
    for c in sel:
        s = vals(gd, c)
        if len(s) < 2:
            continue
        d = describe(s)
        rows.append({"항목": f"{c} ({UNIT[c]})", "평균": round(d["평균"], 2), "중앙값": round(d["중앙값"], 2),
                     "표준편차": round(d["표준편차"], 2), "CV(%)": round(d["CV"], 1), "최소": round(d["최소"], 2),
                     "최대": round(d["최대"], 2), "최대÷최소": round(d["최대"] / d["최소"], 1) if d["최소"] > 0 else np.nan})
    if rows:
        table(pd.DataFrame(rows))
        st.caption("CV가 클수록 개체 간 차이가 커요. '평균 > 중앙값'이면 일부 큰 개체가 평균을 끌어올린 오른쪽 꼬리 분포예요.")
    else:
        st.info("선택한 항목에서 계산할 값이 부족해요.")
    st.markdown("##### 그래프 4. 분포 히스토그램")
    for c in [x for x in ["잎 수", "생중량"] if x in sel]:
        s = vals(gd, c)
        if len(s) >= 2:
            show(hist_fig(s, c, UNIT[c]))
    legend(["🟧 **막대**: 값을 일정한 구간으로 나눴을 때 그 구간에 들어온 개체 수 (막대가 높을수록 개체가 많아요)",
            "🔴 **빨간 점선**: 중앙값 / 🟥 **진한 빨강 점선**: 평균. 두 선이 멀수록 분포가 한쪽으로 치우쳤어요.",
            "오른쪽으로 긴 꼬리가 있으면 '아주 크게 자란 소수 개체'가 있다는 뜻이에요."])

    cm_cols = [c for c in sel if c in gd.columns and len(vals(gd, c)) >= 3]
    if len(cm_cols) >= 3:
        st.markdown("##### 그래프 5. 생육 항목끼리의 상관 히트맵")
        cm = gd[cm_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(2)
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm_cols, y=cm_cols, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=cm.values, texttemplate="%{text}"))
        show(style(fig, "항목 간 Spearman 순위상관", 420))
        legend(["🟦 **칸 하나**: 가로·세로 두 항목이 함께 움직이는 정도 (칸 안 숫자가 상관계수)",
                "🟥 **진한 빨강**: +1에 가까움(함께 커짐) / ⬜ **흰색**: 0(관계 없음) / 🟦 **파랑**: 음의 관계",
                "**생중량 줄**에서 빨간 칸이 가장 진한 항목이 생중량과 가장 밀접해요. 잎 수 칸이 진하면 가설의 출발점이 맞아요."])

    st.subheader("3.3 잎 수와 다른 값의 관계 (전체 개체로 먼저 보기)")
    why("가설은 상·하위 극단만 비교해요. 전체 81개체 같은 중간 개체까지 '잎 수가 늘 때 생중량도 느는지' 보면 가설의 방향이 우연이 아닌지 한 번 더 확인할 수 있어요.")
    if kr is None:
        st.info("잎 수·생중량 열이 있어야 산점도를 그릴 수 있어요.")
        return
    d = kr["d"]
    others = [c for c in sel if c not in ("잎 수", "생중량") and c in d.columns]
    st.markdown("##### 그래프 6. 잎 수 vs 생중량 산점도")
    show(scatter_fig(d, ["잎 수"], "생중량", "잎 수와 생중량 (점 하나 = 식물 1개체)"))
    legend(["🟢 **초록 점**: 잎 수 상위 25% 개체 / 🟠 **주황 점**: 하위 25% 개체 / ⚪ **회색 점**: 비교에서 빠진 가운데 개체",
            "⚫ **검은 점선(추세선)**: 점들의 평균적인 방향을 직선으로 그린 것. 오른쪽 위로 올라가면 '잎이 많을수록 무겁다'는 경향이에요.",
            "오른쪽 끝에 홀로 떨어진 점은 이상치 후보예요. 이 한 점 때문에 결과가 달라지는지 아래 Spearman으로 확인해요."])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson 상관계수 r", f"{kr['r']:.3f}")
    c2.metric("Spearman 순위상관 ρ", f"{kr['rho']:.3f}")
    c3.metric("r의 p값", fp(kr["pr"]))
    st.caption("r이 +1에 가까울수록 함께 커져요. 상관은 '같이 움직인다'일 뿐 '잎이 많아서 무겁다'는 인과는 아니에요.")
    if others:
        st.markdown("##### 그래프 7. 다른 항목과 생중량의 산점도")
        show(scatter_fig(d, others, "생중량", "선택한 항목 vs 생중량"))
        legend(["점·추세선의 뜻은 그래프 6과 같아요. 점이 추세선 가까이 모일수록 그 항목이 생중량을 잘 따라간다는 뜻이에요.",
                "점이 넓게 퍼져 있고 추세선이 완만하면 그 항목은 생중량과 관계가 약해요. 잎 수가 더 강한 단서인지 비교해 보세요."])


def group_section(kr):
    st.subheader("3.4 가설 검정: 잎 수 상위 25% vs 하위 25%")
    why("이 단계가 가설을 **직접** 시험해요. 잎 수만으로 나눈 두 그룹의 생중량 평균이 다르고, 그 차이가 우연이 아닌지 확인합니다.")
    if kr is None:
        st.info("잎 수·생중량 열이 있는 생육 CSV(개체 8개 이상)를 올리면 표시돼요.")
        return
    d, k = kr["d"], kr["k"]
    lo_max = d.loc[d["그룹"] == BOT, "잎 수"].max()
    hi_min = d.loc[d["그룹"] == TOP, "잎 수"].min()
    ties = int(((d["그룹"] != BOT) & (d["잎 수"] == lo_max)).sum() + ((d["그룹"] != TOP) & (d["잎 수"] == hi_min)).sum())
    st.markdown(f"전체 **{kr['n']}개체** 중 잎 수가 적은 순 {k}개체(≤ {lo_max:g}장)를 **하위 25%**, "
                f"많은 순 {k}개체(≥ {hi_min:g}장)를 **상위 25%** 로 나눴고, 나머지 {kr['n'] - 2 * k}개체는 제외했어요.")
    if ties > 0:
        st.caption(f"경계 잎 수와 같은 값을 가진 개체가 그룹 밖에 {ties}개 있어요. 동점은 순서대로 잘라 개수를 {k}개씩 맞췄어요.")
    m1, m2, m3 = st.columns(3)
    m1.metric("하위 25% 평균 생중량", f"{kr['bot_fw']:.2f} g")
    m2.metric("상위 25% 평균 생중량", f"{kr['top_fw']:.2f} g", f"{kr['top_fw'] - kr['bot_fw']:+.2f} g")
    m3.metric("Welch t검정", fp(kr["p"]))

    ga = d.loc[d["그룹"] == TOP, "생중량"].values
    gb = d.loc[d["그룹"] == BOT, "생중량"].values
    rng = np.random.RandomState(42)
    diffs = np.array([rng.choice(ga, len(ga)).mean() - rng.choice(gb, len(gb)).mean() for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    st.markdown("##### 그래프 8. 평균 차이의 95% 신뢰구간 (부트스트랩)")
    fig = go.Figure(go.Histogram(x=diffs, nbinsx=40, marker_color="#2980b9", opacity=0.8, name="다시 뽑은 평균 차이"))
    fig.add_vline(x=0, line_color="black", line_dash="dash", annotation_text="차이 0 (가설 반대)", annotation_position="top left")
    fig.add_vline(x=float(ci_lo), line_color="red", line_dash="dot")
    fig.add_vline(x=float(ci_hi), line_color="red", line_dash="dot", annotation_text="95% 구간", annotation_position="top right")
    fig.update_xaxes(title_text="상위 − 하위 평균 생중량 차이 (g)")
    fig.update_yaxes(title_text="횟수 (2,000번 다시 뽑기)")
    show(style(fig, f"평균 차이 95% 신뢰구간: {ci_lo:.2f} ~ {ci_hi:.2f} g", 360))
    legend(["🟦 **파란 막대**: 각 그룹에서 개체를 무작위로 다시 뽑아(2,000번) 평균 차이를 계산했을 때 값이 나온 횟수",
            "🔴 **빨간 점선 두 줄**: 가운데 95%가 들어오는 범위(신뢰구간) / ⚫ **검은 점선**: 차이 0",
            "**검은 선이 빨간 두 선 바깥**에 있으면 '차이가 0일 가능성이 낮다'는 뜻이라 가설을 더 믿을 수 있어요."])

    st.markdown("##### 그래프 9. 개체별 생중량 (잎 수 적은 순)")
    order = d.sort_values("잎 수", kind="mergesort").reset_index(drop=True)
    ids = order["개체번호"].map(fmt_id) if "개체번호" in order.columns else pd.Series(range(1, len(order) + 1)).astype(str)
    order["라벨"] = ids.values
    fig = go.Figure()
    for g, color in [(BOT, C_BOT), (MID, C_MID), (TOP, C_TOP)]:
        s = order[order["그룹"] == g]
        fig.add_trace(go.Bar(x=s["라벨"], y=s["생중량"], name=g, marker_color=color, customdata=s["잎 수"],
                             hovertemplate="개체 %{x}<br>생중량 %{y:.2f}g<br>잎 수 %{customdata}장<extra></extra>"))
    fig.update_xaxes(title_text="개체 (왼쪽일수록 잎 수가 적음)", categoryorder="array",
                     categoryarray=list(order["라벨"]), tickangle=-60)
    fig.update_yaxes(title_text="생중량 (g)")
    show(style(fig, "개체별 생중량: 하위(주황)·상위(초록) 강조", 400))
    legend(["각 **막대 하나 = 식물 1개체**, 막대 높이 = 그 개체의 생중량이에요. 가로 이름표는 개체번호예요.",
            "🟠 **주황 막대**: 잎 수 하위 25% / 🟢 **초록 막대**: 상위 25% / ⚪ **회색**: 비교에서 뺀 중간 개체",
            "왼쪽(잎 적음)에서 오른쪽(잎 많음)으로 갈수록 막대가 높아지면 가설과 같은 방향이에요."])

    st.markdown("##### 그래프 10. 두 그룹의 분포 비교 (상자그림)")
    bc1, bc2 = st.columns(2)
    with bc1:
        show(box_fig(d, "생중량"))
    with bc2:
        show(box_fig(d, "잎당 생중량"))
    st.caption("오른쪽은 '잎 한 장당 생중량(생중량÷잎 수)'이에요. 상위 그룹이 단지 잎이 많아서 무거운 건지, 잎 한 장도 더 무거운지 구분해 줘요.")
    legend(["📦 **상자**: 그룹 가운데 50% 개체가 들어 있는 범위 (아래 끝 25% ~ 위 끝 75%)",
            "➖ **상자 안의 실선**: 중앙값 / ┅ **점선**: 평균 (평균이 중앙값보다 크면 큰 개체가 평균을 끌어올린 거예요)",
            "┬ **위아래 수염**: 일반적인 범위의 끝 / ● **점**: 개체 하나하나. 수염 밖 점은 이상치예요.",
            "두 상자가 **겹치지 않고 멀수록** 두 그룹의 차이가 분명하다는 뜻이에요."])

    st.markdown("##### 표 1. 항목별 비교와 통계 검정")
    rows = compare_rows(d, sel + ["잎당 생중량"])
    if rows:
        cmp_df = pd.DataFrame(rows)
        table(cmp_df)
        st.caption("p < 0.05이면 우연으로 보기 어려운 차이예요. 두 검정이 모두 같은 결론이면 결과가 더 믿을 만해요. "
                   "잎 수는 그룹을 나누는 기준이라 항상 차이가 나요(참고용). 핵심은 **생중량** 행이에요.")
        st.download_button("📥 비교표 CSV 내려받기", cmp_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="그룹비교표.csv", mime="text/csv")
    out = d.drop(columns=[c for c in ["라벨"] if c in d.columns])
    st.download_button("📥 개체별 그룹표 CSV 내려받기", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="개체별_그룹표.csv", mime="text/csv")


def model_section(kr):
    st.subheader("3.5 잎 수로 생중량 예측하기")
    why("그룹 비교는 '차이가 있다'까지만 알려줘요. 잎 수 1장이 늘 때 생중량이 얼마나 느는지, 예측이 얼마나 믿을 만한지까지 보여줘서 가설의 크기와 쓸모를 판단합니다.")
    if kr is None:
        st.info("잎 수·생중량 열이 있는 생육 CSV를 올리면 표시돼요.")
        return
    d = kr["d"]
    x, y = d["잎 수"].values.astype(float), d["생중량"].values.astype(float)
    a, b, r2, mae = fit_line(x, y)
    cvr2, cvmae = cv_scores(x, y) if len(x) >= 10 else (np.nan, np.nan)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기울기 (잎 1장당 g)", f"{a:.4f}")
    c2.metric("R² (설명력)", f"{r2:.3f}")
    c3.metric("MAE (평균 오차)", f"{mae:.2f} g")
    c4.metric("교차검증 R²", "계산 불가" if np.isnan(cvr2) else f"{cvr2:.3f}")
    st.caption(f"예측식: 생중량(g) = {a:.4f} × 잎 수 + ({b:.4f}) · 5겹 교차검증 평균 오차 "
               f"{'계산 불가' if np.isnan(cvmae) else f'{cvmae:.2f} g'}")
    lo, hi = int(x.min()), int(x.max())
    leaf = st.slider("잎 수를 골라 보세요 (장)", lo, hi, int(np.median(x)))
    pred = max(a * leaf + b, 0.0)
    st.metric(f"잎 {leaf}장일 때 예상 생중량", f"{pred:.2f} g")
    xs = np.linspace(x.min(), x.max(), 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="실제 개체", marker=dict(color=C_MID, size=8)))
    fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines", name="회귀선(예측)", line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=a * xs + b + mae, mode="lines", name="+MAE", line=dict(color="red", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=xs, y=a * xs + b - mae, mode="lines", name="−MAE", line=dict(color="red", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=[leaf], y=[pred], mode="markers", name="내가 고른 잎 수",
                             marker=dict(symbol="star", size=18, color="gold", line=dict(color="black", width=1))))
    fig.update_xaxes(title_text="잎 수 (장)")
    fig.update_yaxes(title_text="생중량 (g)")
    show(style(fig, "잎 수 → 생중량 회귀선", 440))
    legend(["⚪ **회색 점**: 실제로 잰 개체 / 🔴 **빨간 실선**: 잎 수로 예측한 생중량(평균적인 경향)",
            "🔴 **빨간 점선 두 줄**: 예측값 ± 평균 오차(MAE). 실제 점이 대체로 이 띠 안에 들어오면 예측이 쓸 만해요.",
            "⭐ **노란 별**: 위 슬라이더로 고른 잎 수의 예상 생중량 (관측 범위 밖은 믿기 어려워 슬라이더를 범위 안으로 제한했어요)",
            "큰 잎 수에서 점이 선 위쪽으로 흩어지면 '잎이 많은 개체일수록 예측이 더 어려운 것'이에요."])


def tab_result1():
    st.header("3. 결과 (1) 환경과 개체 차이")
    st.caption("발표 순서: 환경이 일정했나 → 개체 차이가 실제로 큰가 → 잎 수와 생중량이 함께 움직이나")
    kr = key_results(growth_df) if has(growth_df) else None
    env_section(env_df)
    st.markdown("---")
    growth_section(growth_df, kr)


def tab_result2():
    st.header("3. 결과 (2) 가설 검정과 예측")
    kr = key_results(growth_df) if has(growth_df) else None
    if kr is not None:
        st.markdown("##### 🎯 한눈에 보는 핵심 결과")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("하위 25% 평균 생중량", f"{kr['bot_fw']:.2f} g")
        c2.metric("상위 25% 평균 생중량", f"{kr['top_fw']:.2f} g", f"{kr['top_fw'] / kr['bot_fw']:.1f}배", delta_color="off")
        c3.metric("검정 결과", "가설 지지 ✅" if kr["supported"] else "지지 안 됨 ⚠️")
        c4.metric("잎 수–생중량 r", f"{kr['r']:.2f}")
        st.markdown("---")
    group_section(kr)
    st.markdown("---")
    model_section(kr)


# ───────────── 탭3 ─────────────
def tab_discuss():
    kr = key_results(growth_df) if has(growth_df) else None
    st.header("4. 논의")
    if kr is None:
        st.info("생육 CSV(잎 수·생중량)를 올리면 논의와 결론이 자동으로 채워져요.")
    else:
        word = "지지되었어요 ✅" if kr["supported"] else "이 데이터로는 지지되지 않았어요 ⚠️"
        st.subheader("4.1 가설 검토")
        st.markdown(f"""
**가설: 잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다 → {word}**
- 평균 생중량: 상위 {kr['top_fw']:.2f} g vs 하위 {kr['bot_fw']:.2f} g (상위가 하위의 약 {kr['top_fw'] / kr['bot_fw']:.1f}배)
- 평균 잎 수: 상위 {kr['top_leaf']:.1f}장 vs 하위 {kr['bot_leaf']:.1f}장
- 검정: Welch t검정 {fp(kr['p'])}, Mann-Whitney {fp(kr['pm'])}
- 전체 개체의 상관: Pearson r = {kr['r']:.3f}, Spearman ρ = {kr['rho']:.3f}
""")
        st.subheader("4.2 해석할 때 조심할 점")
        st.markdown("""
- **상관·차이는 인과가 아니에요.** '잎이 많아서 무겁다'가 아니라 '잎이 많은 개체가 무거운 경향이 있다'까지만 말할 수 있어요.
- 잎 수와 생중량은 둘 다 **'식물이 얼마나 잘 자랐나'** 를 반영해요. 그래서 함께 커지는 건 어느 정도 예상된 결과이고, 이 연구의 가치는 '쉬운 잎 수 측정으로 무게를 가늠할 수 있는가'에 있어요.
- 극단값 한두 개가 결과를 만들었을 수 있어서 순위 기반 검정(Spearman, Mann-Whitney)도 함께 확인했어요.
""")
        same = kr["pl_p"] >= 0.05 if not np.isnan(kr["pl_p"]) else None
        if same is not None:
            st.markdown(f"- **잎 한 장당 생중량**은 상위 {kr['pl_top']:.3f} g vs 하위 {kr['pl_bot']:.3f} g ({fp(kr['pl_p'])}). "
                        + ("차이가 크지 않아서, 상위 그룹이 무거운 이유는 **잎 한 장이 더 무거워서라기보다 잎이 많아서**에 가까워요."
                           if same else "잎 한 장 자체도 상위 그룹이 더 무거웠어요."))
    st.subheader("4.3 환경과의 연결 가능성")
    msg = "환경 CSV를 올리면 EC·pH 변화 여부를 여기에서 확인할 수 있어요."
    if has(env_df, "ec", "ph"):
        msg = "EC·pH가 재배 기간 동안 변했다면 개체의 양분 흡수에 영향을 줬을 가능성은 있어요. 다만 생육 측정일이 없어 둘을 직접 연결하지는 못해요."
    st.write(msg)

    st.header("5. 한계점과 계산 방향")
    miss = int(env_df["ec"].isna().sum()) if has(env_df, "ec") else 0
    lim = [
        ("생육 측정일이 없음", "환경 시계열과 개체 결과를 짝지을 수 없어요.", "측정일을 기록하고, 측정 전 일주일 평균 환경을 개체 결과와 함께 비교"),
        ("개체 위치 정보가 없음", "빛·바람 차이 같은 위치 효과를 분리할 수 없어요.", "개체의 줄·칸을 기록하고 위치별 평균 생중량을 비교"),
        ("표본이 작음", f"분석한 개체가 {kr['n'] if kr else '많지 않은'}개라 그룹당 {kr['k'] if kr else '소수'}개체만 비교해요.", "반복 재배로 개체를 늘리고, 평균 차이의 신뢰구간을 계산"),
        ("센서 결측", f"EC·pH에 빈 값이 {miss}개 있어요." if miss else "센서가 못 잰 구간이 있을 수 있어요.", "결측 시각을 확인하고 빈 구간이 특정 시기에 몰렸는지 점검"),
        ("극단만 비교", "상·하위 25%만 보면 가운데 50%의 정보를 버려요.", "전체 개체를 쓰는 회귀(3.5)와 함께 해석하고, 잎 한 장당 생중량(생중량÷잎 수)도 계산"),
        ("인과관계 아님", "관계가 있어도 원인이라고 말할 수 없어요.", "지상부 길이 등 다른 변수를 함께 넣은 다중회귀로 잎 수의 몫을 따로 확인"),
        ("일반화 불가", "한 학교, 한 스마트팜, 한 시기의 결과예요.", "다른 학교 CSV를 같은 대시보드에 올려 결과가 반복되는지 비교"),
    ]
    table(pd.DataFrame(lim, columns=["한계점", "왜 문제인가", "다음 계산·개선 방향"]))

    st.header("6. 결론")
    if kr is None:
        st.info("생육 CSV를 올리면 결론이 자동으로 작성돼요.")
        return
    verdict = "지지되었다" if kr["supported"] else "지지되지 않았다"
    st.markdown(f"""
{school}의 극지식물 {kr['n']}개체를 잎 수 기준으로 나눠 비교한 결과, **'잎 수 상위 25% 개체는 하위 25%보다 생중량이 클 것이다'라는 가설은 {verdict}.**
상위 그룹의 평균 생중량은 {kr['top_fw']:.2f} g, 하위 그룹은 {kr['bot_fw']:.2f} g이었고({fp(kr['p'])}),
전체 개체에서도 잎 수와 생중량의 상관이 r = {kr['r']:.2f}로 같은 방향이었어요.
다만 이는 인과가 아니라 **함께 변하는 경향**이며, 측정일·위치 정보가 없고 한 시기의 데이터라 다른 환경에 그대로 적용하기는 어려워요.
다음 재배에서는 측정일과 위치를 기록하면 환경 변화와 개체 차이를 더 구체적으로 연결할 수 있어요.
""")


def tab_present(kr):
    st.header("7. 발표 도우미")
    st.subheader("🎤 30초 요약 스크립트")
    if kr is None:
        st.info("생육 CSV를 올리면 발표 요약이 자동으로 채워져요.")
    else:
        st.markdown(f"""
> 같은 스마트팜에서 자란 극지식물 {kr['n']}개체의 생육을 비교했습니다.
> 잎 수 상위 25%의 평균 생중량은 **{kr['top_fw']:.2f} g**, 하위 25%는 **{kr['bot_fw']:.2f} g**으로 약 **{kr['top_fw'] / kr['bot_fw']:.1f}배** 차이가 났고,
> 이 차이는 우연으로 보기 어렵습니다({fp(kr['p'])}). 따라서 **가설은 {'지지됩니다' if kr['supported'] else '지지되지 않았습니다'}.**
> 다만 이것은 인과가 아니라 함께 변하는 경향이며, 측정일과 위치 정보가 없다는 한계가 있습니다.
""")
    st.subheader("❓ 예상 질문과 답변")
    qa = [
        ("왜 생중량이 아니라 잎 수로 그룹을 나눴나요?",
         "가설이 '잎 수가 많은 개체가 더 무거운가'이기 때문이에요. 생중량으로 나누고 생중량을 비교하면 당연한 결과가 나와 검정이 되지 않아요."),
        ("잎이 많아서 무거워진 건가요?",
         "그렇게 말할 수 없어요. 잎 수와 생중량은 둘 다 '얼마나 잘 자랐나'를 반영해서 함께 커져요. 원인은 따로 실험해야 알 수 있어요."),
        ("환경 때문에 차이가 난 건 아닌가요?",
         "같은 스마트팜이라 환경은 공통이지만, 생육 측정일과 개체 위치 정보가 없어 환경과 직접 연결하지 못했어요. 한계점으로 적었고 다음 재배에서 기록할 계획이에요."),
        ("개체 수가 적지 않나요?",
         "그룹당 20개체 안팎이라 작은 편이에요. 그래서 t검정 외에 순위 검정과 부트스트랩 신뢰구간까지 확인해 결과가 한 가지 방법에 의존하지 않게 했어요."),
        ("극단적인 개체가 결과를 만든 건 아닌가요?",
         "극단값은 지우지 않고 두었어요. 대신 극단값에 덜 민감한 Spearman 상관과 Mann-Whitney 검정에서도 같은 결론인지 확인했어요."),
        ("이 결과를 어디에 쓸 수 있나요?",
         "잎 수만 세어도 생중량을 대략 예상할 수 있어 수확 전 상태 점검에 쓸 수 있어요. 다만 이 스마트팜·이 시기 데이터라는 범위 안에서만 말할 수 있어요."),
    ]
    for q, a in qa:
        with st.expander("Q. " + q):
            st.write(a)


# ───────────── 탭 실행 (오류가 나도 앱은 멈추지 않게) ─────────────
def safe(fn, *args):
    try:
        fn(*args)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


with tab1:
    safe(tab_problem)
with tab2:
    safe(tab_method)
with tab3:
    safe(tab_result1)
with tab4:
    safe(tab_result2)
with tab5:
    safe(tab_discuss)
    st.markdown("---")
    safe(tab_present, key_results(growth_df) if has(growth_df) else None)
