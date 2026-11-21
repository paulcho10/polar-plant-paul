"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
====================================================================
사용법: streamlit run main.py
호환성: utf-8-sig/cp949 인코딩, 열 이름 달라도 자동 매핑, TTC 한글 폰트 적용
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
from matplotlib.font_manager import FontProperties
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ── 한글 폰트 (환경 TTC 경로 고정 + rcParams 병행) ──────────────
_KR_TTC_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_KR_FONT_FAMILY = "NanumGothic"
_KR_FONT_FP = None

def _setup_korean_font():
    global _KR_FONT_FP, _KR_TTC_PATH
    if not os.path.isfile(_KR_TTC_PATH):
        for alt in [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]:
            if os.path.isfile(alt):
                _KR_TTC_PATH = alt; break
        if _KR_TTC_PATH == "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc":
            for f in fm.fontManager.ttflist:
                if any(k in (f.name or "").lower() for k in
                       ["nanum","noto","cjk","korean","gothic"]):
                    _KR_TTC_PATH = f.fname; break
    _KR_FONT_FP = FontProperties(fname=_KR_TTC_PATH)
    plt.rcParams["font.family"] = _KR_FONT_FAMILY
    plt.rcParams["font.sans-serif"] = [_KR_FONT_FAMILY]
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()

def _kp(size=10, weight="normal"):
    return FontProperties(fname=_KR_TTC_PATH, size=size, weight=weight)

def _apply_font_all(fig):
    fp = _kp()
    for ax in fig.get_axes():
        t = ax.get_title()
        if t: ax.set_title(t, fontproperties=fp, fontsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontproperties=fp)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=fp)
        for lbl in ax.get_xticklabels(): lbl.set_fontproperties(fp)
        for lbl in ax.get_yticklabels(): lbl.set_fontproperties(fp)
        for txt in ax.texts: txt.set_fontproperties(fp)
    stp = fig._suptitle
    if stp is not None: stp.set_fontproperties(fp)
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            try: leg.set_fontproperties(fp)
            except Exception: pass
    for leg in fig.legends:
        try: leg.set_fontproperties(fp)
        except Exception: pass

C_TEMP = "#e74c3c"; C_HUMI = "#3498db"; C_EC = "#27ae60"
C_PH = "#8e44ad"; C_FW = "#f39c12"; C_TOP = "#2ecc71"; C_BOT = "#e67e22"

# ── 인코딩 + 열 이름 자동 탐지 ────────────────────────────────────
def read_csv_flexible(path_or_bytes, is_env=True):
    raw = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    for enc in ["utf-8-sig","cp949","utf-8"]:
        try:
            raw.seek(0); df = pd.read_csv(raw, encoding=enc)
            if df.shape > 0: break
        except Exception: continue
    else:
        st.error("CSV 파일을 읽을 수 없습니다 (utf-8-sig 또는 cp949 권장)."); return None
    return _map_env_columns(df) if is_env else _map_growth_columns(df)

_ENV_COL_MAPS = [
    (["time","datetime","날짜","측정시간","Timestamp","시간"],"time"),
    (["temperature","온도","temp","기온","Temperature"],"temperature"),
    (["humidity","습도","hum","HUM","humidity(%)","습도(%)"],"humidity"),
    (["ph","PH","pH","수소이온농도","산도"],"ph"),
    (["ec","EC","전기전도도","EC(mS/cm)","ec_mS_cm"],"ec"),
    (["water_temp","watertemperature","수온","water_temp"],"water_temp"),
    (["co2","CO2","co2_ppm","co2(ppm)","이산화탄소"],"co2"),
]
def _map_env_columns(df):
    cols = list(df.columns); std = {}
    for cands, sn in _ENV_COL_MAPS:
        for c in cands:
            if c in cols: std[sn] = c; break
    if "time" not in std: std["time"] = cols
    rmap = {v:k for k,v in std.items()}; df = df.rename(columns=rmap)
    for n in ["time","temperature","humidity","ph","ec","water_temp","co2"]:
        if n not in df.columns: df[n] = np.nan
    return df

_GROWTH_COL_MAPS = [
    (["개체번호","번호","id","개체","plant_id","ID","개체아이디"],"개체번호"),
    (["잎 수(장)","잎수","잎 수","leaf_count","leaf","잎장수","잎\n수(장)"],"잎 수(장)"),
    (["지상부 길이(mm)","지상부길이(mm)","지상부 길이","지상부길이",
      "shoot_length","지상부길이_mm","지상부 길이(mm)\n","지상부길이 (mm)"],"지상부 길이(mm)"),
    (["지하부길이(mm)","지하부 길이(mm)","지하부길이","지하부 길이",
      "root_length","지하부길이_mm","지하부 길이(mm)\n","지하부길이 (mm)"],"지하부길이(mm)"),
    (["생중량(g)","생중량","fresh_weight","fw","생체중량(g)",
      "생중량(g)\n","생체중량","생중량 (g)"],"생중량(g)"),
    (["지상부질량(g)","지상부질량","지상부 질량(g)","shoot_mass",
      "shoot_weight","지상부질량(g)\n","지상부 질량(g)\n"],"지상부질량(g)"),
    (["지하부질량(g)","지하부질량","지하부 질량(g)","root_mass",
      "root_weight","지하부질량(g)\n","지하부 질량(g)\n"],"지하부질량(g)"),
]
def _map_growth_columns(df):
    cols = list(df.columns); std = {}
    for cands, sn in _GROWTH_COL_MAPS:
        for c in cands:
            if c in cols: std[sn] = c; break
    if "개체번호" not in std: std["개체번호"] = cols
    rmap = {v:k for k,v in std.items()}; df = df.rename(columns=rmap)
    for n in ["개체번호","잎 수(장)","지상부 길이(mm)","지하부길이(mm)",
              "생중량(g)","지상부질량(g)","지하부질량(g)"]:
        if n not in df.columns: df[n] = np.nan
    return df

# ── 환경 데이터 분석 ──────────────────────────────────────────────
def parse_time(series):
    s = series.astype(str).str.strip()
    dt1 = pd.to_datetime(s, errors="coerce")
    if dt1.notna().sum() > 0: return dt1
    def _compact(x):
        x = str(x).strip()
        if len(x) >= 9 and "_" in x:
            p = x.split("_"); dp = p; hp = p if len(p) > 1 else "0"
            try: return pd.Timestamp(year=int(dp[:4]), month=int(dp[4:6]),
                                       day=int(dp[6:8]), hour=int(hp) % 24)
            except Exception: return pd.NaT
        return pd.NaT
    return s.apply(_compact)

def analyze_env(df):
    out = {}; df = df.copy(); df["time_dt"] = parse_time(df["time"])
    def _basic(s):
        s = s.dropna()
        if len(s) == 0:
            return dict(mean=np.nan,std=np.nan,var=np.nan,cv=np.nan,
                        min=np.nan,max=np.nan,q1=np.nan,q3=np.nan,iqr=np.nan,n=0)
        m=float(s.mean()); sd=float(s.std(ddof=1)); v=float(s.var(ddof=1))
        cv=float(sd/m*100) if m!=0 else np.nan
        return dict(mean=m,std=sd,var=v,cv=cv,min=float(s.min()),max=float(s.max()),
                    q1=float(s.quantile(.25)),q3=float(s.quantile(.75)),
                    iqr=float(s.quantile(.75)-s.quantile(.25)),n=len(s))
    out["n_env"]=len(df)
    out["temp"]=_basic(df["temperature"]); out["humi"]=_basic(df["humidity"])
    out["ec"]=_basic(df["ec"]); out["ph"]=_basic(df["ph"])
    df_t = df.dropna(subset=["time_dt"]).copy()
    if len(df_t) > 0:
        df_t["hour"]=df_t["time_dt"].dt.hour
        hourly=df_t.groupby("hour").agg(temp_mean=("temperature","mean"),humi_mean=("humidity","mean"))
        out["hourly"]=hourly
        out["hourly_temp_max_h"]=int(hourly["temp_mean"].idxmax())
        out["hourly_temp_min_h"]=int(hourly["temp_mean"].idxmin())
        out["hourly_temp_max_v"]=float(hourly["temp_mean"].max())
        out["hourly_temp_min_v"]=float(hourly["temp_mean"].min())
        out["daily_range_temp"]=out["hourly_temp_max_v"]-out["hourly_temp_min_v"]
        out["hourly_humi_max_h"]=int(hourly["humi_mean"].idxmax())
        out["hourly_humi_min_h"]=int(hourly["humi_mean"].idxmin())
        out["hourly_humi_max_v"]=float(hourly["humi_mean"].max())
        out["hourly_humi_min_v"]=float(hourly["humi_mean"].min())
        out["daily_range_humi"]=out["hourly_humi_max_v"]-out["hourly_humi_min_v"]
        md=df_t["time_dt"].median(); out["median_date"]=md
        f1=df_t[df_t["time_dt"]<=md]; f2=df_t[df_t["time_dt"]>md]
        def _h(sub,col):
            s=sub[col].dropna()
            if len(s)==0: return dict(mean=np.nan,std=np.nan,var=np.nan,n=0)
            return dict(mean=float(s.mean()),std=float(s.std(ddof=1)),var=float(s.var(ddof=1)),n=len(s))
        out["first_temp"]=_h(f1,"temperature"); out["second_temp"]=_h(f2,"temperature")
        out["first_humi"]=_h(f1,"humidity"); out["second_humi"]=_h(f2,"humidity")
        out["first_ec"]=_h(f1,"ec"); out["second_ec"]=_h(f2,"ec")
        out["first_ph"]=_h(f1,"ph"); out["second_ph"]=_h(f2,"ph")
        start=df_t["time_dt"].min(); eck=start+pd.Timedelta(days=10); late=md+pd.Timedelta(days=20)
        early_ec=df_t[(df_t["time_dt"]>=start)&(df_t["time_dt"]<=eck)]["ec"].dropna()
        late_ec=df_t[(df_t["time_dt"]>=late-pd.Timedelta(days=10))&(df_t["time_dt"]<=late+pd.Timedelta(days=10))]["ec"].dropna()
        early_ph=df_t[(df_t["time_dt"]>=start)&(df_t["time_dt"]<=eck)]["ph"].dropna()
        late_ph=df_t[(df_t["time_dt"]>=late-pd.Timedelta(days=10))&(df_t["time_dt"]<=late+pd.Timedelta(days=10))]["ph"].dropna()
        out["early_ec_mean"]=float(early_ec.mean()) if len(early_ec)>0 else np.nan
        out["late_ec_mean"]=float(late_ec.mean()) if len(late_ec)>0 else np.nan
        out["early_ph_mean"]=float(early_ph.mean()) if len(early_ph)>0 else np.nan
        out["late_ph_mean"]=float(late_ph.mean()) if len(late_ph)>0 else np.nan
        if not np.isnan(out["early_ec_mean"]) and out["early_ec_mean"]!=0:
            out["ec_change_pct"]=(out["late_ec_mean"]-out["early_ec_mean"])/out["early_ec_mean"]*100
        if not np.isnan(out["early_ph_mean"]) and not np.isnan(out["late_ph_mean"]):
            out["ph_change_abs"]=out["late_ph_mean"]-out["early_ph_mean"]
    def _iqr_out(series):
        s=series.dropna()
        if len(s)==0: return dict(count=0,lo=np.nan,hi=np.nan,indices=,values=)
        q1,q3=float(s.quantile(.25)),float(s.quantile(.75)); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr; mask=(s<lo)|(s>hi)
        return dict(count=int(mask.sum()),lo=lo,hi=hi,indices=list(s[mask].index),values=list(s[mask].values))
    out["temp_out"]=_iqr_out(df["temperature"]); out["humi_out"]=_iqr_out(df["humidity"])
    return out

# ── 생육 데이터 분석 ──────────────────────────────────────────────
def analyze_growth(df):
    out={}; df=df.copy(); id_col=df["개체번호"]
    try:
        df["_id_num"]=pd.to_numeric(id_col,errors="coerce")
        if df["_id_num"].notna().all(): df=df.sort_values("_id_num").reset_index(drop=True)
        else: df=df.reset_index(drop=True)
    except Exception: df=df.reset_index(drop=True)
    n=len(df); out["n_growth"]=n
    col_map={"잎 수(장)":"leaf","지상부 길이(mm)":"shoot","지하부길이(mm)":"root","생중량(g)":"fw"}
    data={k: pd.to_numeric(df[o],errors="coerce").dropna() for o,k in col_map.items()}
    out["data"]=data
    def _stats(s):
        s=s.dropna()
        if len(s)==0: return {}
        m=float(s.mean()); med=float(s.median()); sd=float(s.std(ddof=1))
        v=float(s.var(ddof=1)); cv=float(sd/m*100) if m!=0 else float("nan")
        mn=float(s.min()); mx=float(s.max()); q1=float(s.quantile(.25)); q3=float(s.quantile(.75))
        iqr=q3-q1; ratio=mx/mn if mn!=0 else float("nan")
        return dict(mean=m,median=med,std=sd,var=v,cv=cv,min=mn,max=mx,q1=q1,q3=q3,
                    iqr=iqr,max_min_ratio=ratio,n=len(s),std_err_mean=sd/math.sqrt(len(s)))
    out["stats"]={k:_stats(v) for k,v in data.items()}
    def _od(s):
        s=s.dropna()
        if len(s)==0: return {}
        q1,q3=float(s.quantile(.25)),float(s.quantile(.75)); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr; mask=(s<lo)|(s>hi); idx=s.index
        return dict(count=int(mask.sum()),lo=float(lo),hi=float(hi),
                    outlier_ids=[id_col.iloc[i] for i in idx[mask].tolist()],
                    outlier_values=[float(s.iloc[i]) for i in idx[mask].tolist()])
    out["outliers"]={k:_od(v) for k,v in data.items()}
    fw=data["fw"]; corr_info={}
    for k,cn in [("leaf","잎 수(장)"),("shoot","지상부 길이(mm)"),("root","지하부길이(mm)")]:
        s=data[k]; common=pd.concat([fw,s],axis=1).dropna()
        if len(common)>=3: rv,pv=stats.pearsonr(common.iloc[:,0],common.iloc[:,1])
        else: rv,pv=np.nan,np.nan
        corr_info[k]=dict(r=rv,p=pv,n=len(common),col_name=cn)
    out["corr"]=corr_info
    fw_sorted=fw.sort_values(ascending=False); n_top=max(1,round(n*0.25))
    top_idx=fw_sorted.index[:n_top].tolist(); bot_idx=fw_sorted.index[-n_top:].tolist()
    top_fw=fw.loc[top_idx]; bot_fw=fw.loc[bot_idx]
    g=dict(n_top=len(top_fw),n_bot=len(bot_fw),
           top_ids=[id_col.iloc[i] for i in top_idx],bot_ids=[id_col.iloc[i] for i in bot_idx],
           top_threshold=float(fw_sorted.iloc[min(n_top-1,len(fw_sorted)-1)]),
           bot_threshold=float(fw_sorted.iloc[-min(n_top,len(fw_sorted))]))
    for k in ["leaf","shoot","root"]:
        t=data[k].loc[top_idx]; b=data[k].loc[bot_idx]
        g[f"top_{k}_mean"]=float(t.mean()); g[f"bot_{k}_mean"]=float(b.mean())
        g[f"top_{k}_std"]=float(t.std(ddof=1)); g[f"bot_{k}_std"]=float(b.std(ddof=1))
        g[f"diff_{k}"]=float(t.mean()-b.mean())
        g[f"ratio_{k}"]=(float(t.mean())/float(b.mean()) if b.mean()!=0 else np.nan)
        if len(t)>=2 and len(b)>=2:
            ts,pv=stats.ttest_ind(t,b,equal_var=False)
            g[f"t_{k}"]=float(ts); g[f"p_{k}"]=float(pv)
        else: g[f"t_{k}"]=np.nan; g[f"p_{k}"]=np.nan
    st_=data["shoot"].loc[top_idx]; rt_=data["root"].loc[top_idx]
    sb_=data["shoot"].loc[bot_idx]; rb_=data["root"].loc[bot_idx]
    rT=(st_/rt_).dropna(); rB=(sb_/rb_).dropna()
    g["top_ratio_mean"]=float(rT.mean()) if len(rT)>0 else np.nan
    g["bot_ratio_mean"]=float(rB.mean()) if len(rB)>0 else np.nan
    if len(rT)>=2 and len(rB)>=2:
        tr,pr=stats.ttest_ind(rT,rB,equal_var=False)
        g["ratio_t"]=float(tr); g["ratio_p"]=float(pr)
    else: g["ratio_t"]=np.nan; g["ratio_p"]=np.nan
    g["top_fw_mean"]=float(top_fw.mean()); g["bot_fw_mean"]=float(bot_fw.mean())
    g["top_fw_std"]=float(top_fw.std(ddof=1)); g["bot_fw_std"]=float(bot_fw.std(ddof=1))
    g["diff_fw"]=float(top_fw.mean()-bot_fw.mean())
    g["ratio_fw"]=(float(top_fw.mean())/float(bot_fw.mean()) if bot_fw.mean()!=0 else np.nan)
    if len(top_fw)>=2 and len(bot_fw)>=2:
        tf,pf=stats.ttest_ind(top_fw,bot_fw,equal_var=False)
        g["t_fw"]=float(tf); g["p_fw"]=float(pf)
    else: g["t_fw"]=np.nan; g["p_fw"]=np.nan
    out["group"]=g
    leaf_arr=data["leaf"].values.reshape(-1,1)
    shoot_arr=data["shoot"].values.reshape(-1,1)
    fw_arr=data["fw"].values
    m1=LinearRegression()
    if len(leaf_arr)>=3:
        m1.fit(leaf_arr,fw_arr); p1=m1.predict(leaf_arr)
        cv=KFold(n_splits=5,shuffle=True,random_state=42)
        out["model1"]=dict(coef=float(m1.coef_),intercept=float(m1.intercept_),
            r2=float(r2_score(fw_arr,p1)),mae=float(mean_absolute_error(fw_arr,p1)),
            cv_r2_mean=float(cross_val_score(m1,leaf_arr,fw_arr,cv=cv,scoring="r2").mean()),
            cv_r2_std=float(cross_val_score(m1,leaf_arr,fw_arr,cv=cv,scoring="r2").std()),
            cv_mae_mean=float(-cross_val_score(m1,leaf_arr,fw_arr,cv=cv,
                scoring="neg_mean_absolute_error").mean()),
            cv_mae_std=float(cross_val_score(m1,leaf_arr,fw_arr,cv=cv,
                scoring="neg_mean_absolute_error").std()),
            n=len(fw_arr),
            equation=f"생중량(g) = {m1.coef_:.4f} × 잎 수(장) + ({m1.intercept_:.4f})",
            example_25=float(m1.predict([[25.0]])),
            example_60=float(m1.predict([[60.0]])))
    m2=LinearRegression()
    if len(leaf_arr)>=3 and len(shoot_arr)>=3:
        X2=np.hstack([leaf_arr,shoot_arr]); m2.fit(X2,fw_arr); p2=m2.predict(X2)
        cv2=KFold(n_splits=5,shuffle=True,random_state=42)
        out["model2"]=dict(coef_leaf=float(m2.coef_),coef_shoot=float(m2.coef_),
            intercept=float(m2.intercept_),r2=float(r2_score(fw_arr,p2)),
            mae=float(mean_absolute_error(fw_arr,p2)),
            cv_r2_mean=float(cross_val_score(m2,X2,fw_arr,cv=cv2,scoring="r2").mean()),
            cv_r2_std=float(cross_val_score(m2,X2,fw_arr,cv=cv2,scoring="r2").std()),
            cv_mae_mean=float(-cross_val_score(m2,X2,fw_arr,cv=cv2,
                scoring="neg_mean_absolute_error").mean()),
            cv_mae_std=float(cross_val_score(m2,X2,fw_arr,cv=cv2,
                scoring="neg_mean_absolute_error").std()),
            n=len(fw_arr),
            equation=f"생중량(g) = {m2.coef_:.4f} × 잎 수(장) + "
                     f"{m2.coef_:.4f} × 지상부 길이(mm) + ({m2.intercept_:.4f})",
            example_25=float(m2.predict([[25.0,39.4]])),
            example_60=float(m2.predict([[60.0,39.4]])))
    return out

# ── 그래프 함수 ───────────────────────────────────────────────────
def draw_env_timeseries(env_result, env_df):
    df=env_df.copy(); df["time_dt"]=parse_time(df["time"]); df=df.dropna(subset=["time_dt"])
    daily=df.groupby(df["time_dt"].dt.date).agg(temp_mean=("temperature","mean"),
        humi_mean=("humidity","mean")).reset_index()
    daily.columns=["date","온도(°C)","습도(%)"]; daily["date"]=pd.to_datetime(daily["date"])
    fig,ax1=plt.subplots(figsize=(10,3.8))
    ax1.plot(daily["date"],daily["온도(°C)"],color=C_TEMP,lw=1.8,label="평균 온도 (°C)")
    ax1.set_ylabel("온도 (°C)",color=C_TEMP); ax1.tick_params(axis="y",labelcolor=C_TEMP)
    ax1.set_ylim(15,35)
    ax2=ax1.twinx()
    ax2.plot(daily["date"],daily["습도(%)"],color=C_HUMI,lw=1.8,label="평균 습도 (%)")
    ax2.set_ylabel("습도 (%)",color=C_HUMI); ax2.tick_params(axis="y",labelcolor=C_HUMI)
    ax2.set_ylim(30,80)
    ax1.set_title("날짜별 평균 온도·습도 시계열 (127일)")
    ax1.grid(True,alpha=0.3); fig.autofmt_xdate()
    fig.legend(loc="upper right",bbox_to_anchor=(0.89,0.92),fontsize=9,prop=_kp())
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_hourly_chart(env_result):
    if "hourly" not in env_result:
        st.info("시간대별 데이터를 계산할 수 없습니다."); return
    hourly=env_result["hourly"]; hours=list(hourly.index)
    temp_v=list(hourly["temp_mean"]); humi_v=list(hourly["humi_mean"])
    fig,ax1=plt.subplots(figsize=(10,3.8))
    ax1.bar(hours,temp_v,color=C_TEMP,alpha=0.75,width=0.7,label="온도 (°C)")
    ax1.set_xlabel("시간대 (시)"); ax1.set_ylabel("온도 (°C)",color=C_TEMP)
    ax1.tick_params(axis="y",labelcolor=C_TEMP); ax1.set_ylim(15,30)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h}시" for h in hours],rotation=45,fontsize=8)
    fp=_kp()
    for h,v in zip(hours,temp_v):
        ax1.text(h,v+0.2,f"{v:.1f}",ha="center",fontsize=7,color=C_TEMP,fontproperties=fp)
    ax2=ax1.twinx()
    ax2.plot(hours,humi_v,color=C_HUMI,lw=2,marker="o",ms=4,label="습도 (%)")
    ax2.set_ylabel("습도 (%)",color=C_HUMI); ax2.tick_params(axis="y",labelcolor=C_HUMI)
    ax2.set_ylim(30,80)
    for h,v in zip(hours,humi_v):
        ax2.text(h,v+1.5,f"{v:.1f}",ha="center",fontsize=7,color=C_HUMI,fontproperties=fp)
    ax1.set_title("24개 시간대별 평균 온도(막대) · 습도(선)")
    ax1.axvline(x=env_result["hourly_temp_max_h"],color=C_TEMP,ls="--",alpha=0.5)
    ax1.axvline(x=env_result["hourly_temp_min_h"],color=C_TEMP,ls=":",alpha=0.5)
    ax2.axvline(x=env_result["hourly_humi_max_h"],color=C_HUMI,ls="--",alpha=0.5)
    ax2.axvline(x=env_result["hourly_humi_min_h"],color=C_HUMI,ls=":",alpha=0.5)
    ax1.grid(axis="y",alpha=0.3)
    fig.legend(loc="upper right",bbox_to_anchor=(0.89,0.92),fontsize=9,prop=_kp())
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_ec_ph_trend(env_result, env_df):
    df=env_df.copy(); df["time_dt"]=parse_time(df["time"])
    df=df.dropna(subset=["time_dt","ec","ph"])
    fig,ax1=plt.subplots(figsize=(10,3.8))
    ax1.plot(df["time_dt"],df["ec"],color=C_EC,lw=1.2,alpha=0.7,label="EC (mS/cm)")
    ax1.set_ylabel("EC (mS/cm)",color=C_EC); ax1.tick_params(axis="y",labelcolor=C_EC)
    ax2=ax1.twinx()
    ax2.plot(df["time_dt"],df["ph"],color=C_PH,lw=1.2,alpha=0.7,label="pH")
    ax2.set_ylabel("pH",color=C_PH); ax2.tick_params(axis="y",labelcolor=C_PH)
    ee=env_result.get("early_ec_mean"); el_=env_result.get("late_ec_mean")
    pe=env_result.get("early_ph_mean"); pl_=env_result.get("late_ph_mean")
    if not np.isnan(ee) and not np.isnan(el_):
        ax1.axhline(y=ee,color=C_EC,ls="--",alpha=0.5,label=f"6월 초 EC={ee:.2f}")
        ax1.axhline(y=el_,color=C_EC,ls=":",alpha=0.5,label=f"8월 하순 EC={el_:.2f}")
    if not np.isnan(pe) and not np.isnan(pl_):
        ax2.axhline(y=pe,color=C_PH,ls="--",alpha=0.5,label=f"6월 초 pH={pe:.2f}")
        ax2.axhline(y=pl_,color=C_PH,ls=":",alpha=0.5,label=f"8월 하순 pH={pl_:.2f}")
    ax1.set_title("EC · pH 시계열 (재배 기간 전체)")
    ax1.grid(True,alpha=0.3); fig.autofmt_xdate()
    fig.legend(loc="upper right",bbox_to_anchor=(0.89,0.91),fontsize=8.5,prop=_kp())
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_growth_histogram(growth_result):
    fw=growth_result["data"]["fw"]; sd=growth_result["stats"]["fw"]
    fig,ax=plt.subplots(figsize=(8,3.8))
    bins=np.arange(0,fw.max()+1.5,1.0)
    ax.hist(fw,bins=bins,color=C_FW,alpha=0.75,edgecolor="white",linewidth=1.2)
    # 수정: ax.axvline()에서 fontproperties 인자 제거 (ax.axvline은 fontproperties를 받지 않음)
    ax.axvline(sd["median"],color="red",ls="--",lw=2,
               label=f"중앙값 {sd['median']:.2f}g")
    ax.axvline(sd["mean"],color="darkred",ls=":",lw=2,
               label=f"평균   {sd['mean']:.2f}g")
    ax.set_xlabel("생중량 (g)",fontproperties=_kp())
    ax.set_ylabel("개체 수",fontproperties=_kp())
    ax.set_title("생중량(g) 분포 히스토그램 (81개체)",fontproperties=_kp())
    ax.legend(prop=_kp(),fontsize=9); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_scatter_plots(growth_result):
    d=growth_result["data"]; c=growth_result["corr"]
    fw=d["fw"]; leaf=d["leaf"]; shoot=d["shoot"]; root=d["root"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    pairs=[(axes,leaf,"잎 수(장)",C_TEMP,c["leaf"]),
           (axes,shoot,"지상부 길이(mm)",C_HUMI,c["shoot"]),
           (axes,root,"지하부길이(mm)",C_EC,c["root"])]
    for ax,x,xl,color,cinfo in pairs:
        ax.scatter(x,fw,color=color,alpha=0.65,s=35,edgecolor="white",linewidth=0.5)
        fp=_kp()
        if not np.isnan(cinfo["r"]):
            sl,ic=np.polyfit(x.dropna(),fw.dropna(),1)
            xs=np.linspace(x.min(),x.max(),100)
            ax.plot(xs,sl*xs+ic,color="black",lw=1.5,ls="--",
                    label=f"추세선 (r={cinfo['r']:.3f})")
        ax.set_xlabel(xl,fontproperties=fp)
        ax.set_ylabel("생중량 (g)",fontproperties=fp)
        ax.set_title(f"{xl} vs 생중량",fontproperties=fp)
        ax.legend(prop=fp,fontsize=8.5); ax.grid(alpha=0.3)
        for lbl in ax.get_xticklabels(): lbl.set_fontproperties(fp)
        for lbl in ax.get_yticklabels(): lbl.set_fontproperties(fp)
    fig.suptitle("생중량 vs 각 생육 항목 산점도",fontproperties=_kp(),fontsize=12,y=1.02)
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_group_bar(growth_result):
    g=growth_result["group"]
    items=["잎 수(장)","지상부 길이(mm)","지하부길이(mm)","생중량(g)"]
    keys=["leaf","shoot","root","fw"]
    tv=[g[f"top_{k}_mean"] for k in keys]; bv=[g[f"bot_{k}_mean"] for k in keys]
    ts=[g[f"top_{k}_std"]  for k in keys]; bs=[g[f"bot_{k}_std"]  for k in keys]
    pv=[g[f"p_{k}"]        for k in keys]
    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.arange(len(items)); w=0.35
    ax.bar(x-w/2,tv,w,color=C_TOP,alpha=0.85,label="Top 25% (상위)",edgecolor="white")
    ax.bar(x+w/2,bv,w,color=C_BOT,alpha=0.85,label="Bottom 25% (하위)",edgecolor="white")
    ax.errorbar(x-w/2,tv,yerr=ts,fmt="none",ecolor="black",capsize=4,lw=1)
    ax.errorbar(x+w/2,bv,yerr=bs,fmt="none",ecolor="black",capsize=4,lw=1)
    fp=_kp()
    for i,(tvv,bvv,pvv) in enumerate(zip(tv,bv,pv)):
        ax.text(i-w/2,tvv+max(ts)*0.5,f"{tvv:.2f}",ha="center",fontsize=9,
                color=C_TOP,fontproperties=fp)
        ax.text(i+w/2,bvv+max(bs)*0.5,f"{bvv:.2f}",ha="center",fontsize=9,
                color=C_BOT,fontproperties=fp)
        sig="***" if (not np.isnan(pvv) and pvv<0.001) else \
            ("*" if (not np.isnan(pvv) and pvv<0.05) else "")
        ax.text(i,max(tvv,bvv)+max(max(ts),max(bs))+0.3,sig,
                ha="center",fontsize=14,color="red",fontproperties=fp)
    ax.set_xticks(x); ax.set_xticklabels(items,fontproperties=fp,fontsize=10)
    ax.set_title("Top 25% vs Bottom 25% 그룹 비교  (*** p<0.001, * p<0.05)",
                 fontproperties=fp,fontsize=11)
    ax.legend(prop=fp,fontsize=10); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)

def draw_model_scatter(growth_result):
    m1=growth_result["model1"]; m2=growth_result.get("model2")
    d=growth_result["data"]; fw=d["fw"]; leaf=d["leaf"]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(leaf,fw,color=C_FW,alpha=0.65,s=40,edgecolor="white",
               linewidth=0.6,label="실제 개체 (81개)")
    fp=_kp()
    xs=np.linspace(leaf.min(),leaf.max(),100)
    ys1=m1["coef"]*xs+m1["intercept"]
    ax.plot(xs,ys1,color="red",lw=2,label=f"모델1: {m1['equation']}",alpha=0.9)
    if m2:
        ys2=m2["coef_leaf"]*xs+m2["coef_shoot"]*39.4+m2["intercept"]
        ax.plot(xs,ys2,color="green",lw=2,ls="--",
                label=f"모델2: {m2['equation']}",alpha=0.85)
    for lv,lbl in [(25,"잎 25장"),(60,"잎 60장")]:
        ey=m1["example_25"] if lv==25 else m1["example_60"]
        ax.scatter([lv],[ey],color="darkred",s=130,zorder=5,marker="X",
                   label=f"예측 예시: {lbl} → {ey:.2f}g")
    ax.set_xlabel("잎 수 (장)",fontproperties=fp)
    ax.set_ylabel("생중량 (g)",fontproperties=fp)
    ax.set_title("잎 수 vs 생중량 — 회귀 예측선 및 예시",fontproperties=fp)
    ax.legend(prop=fp,fontsize=8.5,loc="upper left"); ax.grid(alpha=0.3)
    for lbl in ax.get_xticklabels(): lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels(): lbl.set_fontproperties(fp)
    plt.tight_layout(); _apply_font_all(fig); st.pyplot(fig); plt.close(fig)
