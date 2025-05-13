import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter
import warnings
import shap

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../../data/processed/lendingclub_features_for_rf.parquet"
FEATURE_PATH = "../../data/processed/features_final_list_rf.csv"

# --------------------------------------------------
# 1) 데이터 로드 & 필터
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], errors="coerce")

# 결측항 제거  
print(f"issue_d NaT 개수: {df['issue_d'].isna().sum()}")
print(f"last_pymnt_d NaT 개수: {df['last_pymnt_d'].isna().sum()}")
df = df[df['issue_d'].notna() & df['last_pymnt_d'].notna()]  # NaT 제거


# --------------------------------------------------
# COVID‑19 노출 변수 정의
#   - cut‑off 날짜 : 2020‑03‑01
#   - 비‑부도(Charged Off/Default 아님) 대출  →  issue_d + term ≤ cut‑off
#   - 부도 대출                              →  last_pymnt_d ≤ cut‑off
# --------------------------------------------------
cutoff = pd.to_datetime("2020-03-01")

# term이 '36 months' 같은 문자열이면 숫자(월수)만 추출
df["term"] = (
    df["term"].astype(str)
      .str.extract(r"(\d+)")[0]
      .astype(int)
)

# 만기일 = issue_d + term_m × 30일 (월 단위 근사)
df["maturity_d"] = df["issue_d"] + pd.to_timedelta(df["term"] * 30, unit="D")

# 부도(event==1)  →  마지막 상환일이 cutoff **이전(포함)** 이면 코로나 노출
# 그 외            →  만기일이 cutoff **이전(포함)** 이면 코로나 노출
df["covid_exposure"] = np.where(
    df["E"] == 1,
    df["last_pymnt_d"] >= cutoff,
    df["maturity_d"]   >= cutoff
).astype(int)

# Calculate survival duration:
# If event occurred (event == 1), use duration until event (e.g. charged off)
# Else, use duration until end of observation (censored)
# T / event 결측 & 음수 제거
df = df.dropna(subset=["T", "E"])
df = df[df["T"] >= 0]

# --------------------------------------------------
# 2) 특징 행렬 / 타깃
# --------------------------------------------------
features = pd.read_csv(FEATURE_PATH)["feature"].tolist()
features = [f for f in features if f in df.columns]
features = [f for f in features if f not in ["T", "E"] and str(df[f].dtype) not in ["object", "datetime64[ns]"]]

# issue_month 인코딩: XGBoost 학습을 위한 시간 변수 추가
df["issue_month"] = df["issue_d"].dt.to_period("M")
df["issue_month_encoded"] = df["issue_month"].astype("category").cat.codes

## features.append("issue_month_encoded")  # 제거: 실험 목적상 issue_month_encoded를 feature에서 제외

X = df[features]


# ▼▼▼ 추가 코드 ▼▼▼  
print("▶ 데이터 분포 확인:")
print(f"- 전체 데이터: {len(df)}")
print(f"- COVID 노출 그룹: {df['covid_exposure'].sum()}")
print(f"- issue_d < cutoff: {(df['issue_d'] < cutoff).sum()}")
print(f"- 최소 issue_d: {df['issue_d'].min()}, 최대 issue_d: {df['issue_d'].max()}")

df = df[df['covid_exposure'] == 1]
X = df[features].copy()

# --------------------------------------------------
# 3) 시간순 분할
# --------------------------------------------------
# 훈련 데이터 인덱스 마스크: issue_d < cutoff & covid_exposure == 0
train_mask = (df['issue_d'] < cutoff).reindex(X.index).fillna(False)

# Debug: inspect mask components
print(f"▶ Total rows: {len(df)}")
print(f"▶ Rows with issue_d < cutoff ({cutoff.date()}): {int((df['issue_d'] < cutoff).sum())}")

# 훈련/테스트 분할
X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
y_train = df.loc[train_mask, ["T", "E"]]
y_test  = df.loc[~train_mask, ["T", "E"]]
print(X_train.dtypes.value_counts()) # category 타입과 숫자형(int/float)만 존재해야 함


# Fallback if no training data with covid_exposure == 0
if X_train.shape[0] == 0:
    print("⚠️ No non-exposed training data; using time-based split only.")
    train_mask = df["issue_d"] < cutoff
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train = df.loc[train_mask, ["T", "E"]]
    y_test  = df.loc[~train_mask, ["T", "E"]]

# --------------------------------------------------
# 4) Nelson-Aalen 기반 Hazard Rate 추정
# --------------------------------------------------
naf = NelsonAalenFitter()
df["issue_month"] = df["issue_d"].dt.to_period("M")
monthly_hazards = {}
monthly_events = {}
monthly_ci = {}

for month, group_df in df.groupby("issue_month"):
    if len(group_df) < 100:
        continue
    try:
        naf.fit(group_df["T"], event_observed=group_df["E"])
        cum_hazard = naf.cumulative_hazard_
        inst_hazard = cum_hazard.diff().fillna(0)
        monthly_hazards[str(month)] = inst_hazard.mean().values[0]

        # 추가 기능 1: 월별 부도 사건 수 저장
        monthly_events[str(month)] = int(group_df["E"].sum())

        # 추가 기능 2: 신뢰구간 저장 (마지막 시점 기준)
        ci_df = naf.confidence_interval_
        if not ci_df.empty:
            last_ci = ci_df.iloc[-1]
            monthly_ci[str(month)] = (last_ci[0], last_ci[1])
    except Exception as e:
        print(f"⚠️ {month} 월 hazard 계산 오류: {e}")
        continue

# hazard 평균값 시계열 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.plot(monthly_hazards.keys(), monthly_hazards.values(), marker="o", label="NA estimated hazard")
plt.title("Monthly Nelson-Aalen estimated Hazard Rate")
plt.xlabel("issued month")
plt.ylabel("avg Hazard")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print("✅ NA 기반 월별 hazard 추정 완료")

# --------------------------------------------------
# 5-2) 월별 누적 SHAP 분석 및 위험 추정 저장
# --------------------------------------------------

window_results = {}
shap_top_records = []
monthly_shap_dynamic = []
month_labels = []

unique_months = sorted(df["issue_month"].unique())

for month in unique_months:
    # 누적 학습 데이터 생성
    df_window = df[df["issue_month"] <= month]
    if df_window.shape[0] < 300:
        continue
    if df_window.shape[0] > 200000:
        df_window = df_window.sample(n=20000, random_state=42)
    # 특징 행렬 및 타깃 구성
    X_window = df_window[features]
    y_window = df_window[["T", "E"]]
    y_lower = np.where(y_window["E"] == 1, y_window["T"], -np.inf)
    y_upper = y_window["T"]

    # DMatrix 구성
    import xgboost as xgb
    dtrain_window = xgb.DMatrix(data=X_window, label=y_upper)
    dtrain_window.set_float_info("label_lower_bound", y_lower)
    dtrain_window.set_float_info("label_upper_bound", y_upper)

    params = {
        "objective": "survival:aft",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "random_state": 42,
        "nthread": -1,
        "verbosity": 1
    }

    # 모델 학습
    model_window = xgb.train(
        params=params,
        dtrain=dtrain_window,
        num_boost_round=100
    )

    # SHAP 계산 (학습에 사용된 누적 데이터 전체에 대해)
    shap_input = X_window.sample(n=min(20000, len(X_window)), random_state=42)
    import shap
    explainer_window = shap.TreeExplainer(model_window, data=shap_input, feature_perturbation="interventional", approximate=True)
    shap_values_window = explainer_window.shap_values(shap_input)
    shap_df_window = pd.DataFrame(shap_values_window, columns=features)
    monthly_shap_dynamic.append(shap_df_window.mean())
    # 월별 SHAP 평균값 저장 (wide-form)
    monthly_shap_dynamic.append(shap_df_window.mean())
    month_labels.append(str(month))

    # 월별 SHAP 상위 10개 저장 (long-form)
    top_features = shap_df_window.abs().mean().sort_values(ascending=False).head(10)
    for feature, value in top_features.items():
        shap_top_records.append({
            "issue_month": str(month),
            "feature": feature,
            "mean_abs_shap": value
        })

# 결과 저장
monthly_shap_df = pd.DataFrame(monthly_shap_dynamic, index=month_labels)
monthly_shap_df.index.name = "issue_month"
monthly_shap_df.to_csv("../../reports/monthly_shap_dynamic.csv")
print("📁 월별 누적 SHAP 평균값 저장 완료: monthly_shap_dynamic.csv")

shap_top_df = pd.DataFrame(shap_top_records)
shap_top_df.to_csv("../../reports/monthly_top10_shap_longform.csv", index=False)
print("📁 월별 SHAP 상위 10개 변수 저장 완료: monthly_top10_shap_longform.csv")

# --------------------------------------------------
# 5-1) Nelson-Aalen hazard와 SHAP 기반 변수 기여도의 상관성 분석
# --------------------------------------------------

from scipy.stats import pearsonr

# 전체 SHAP 중요도 기준 상위 3개 변수 선택
global_mean_abs = shap_df[features].abs().mean()
top3_features = global_mean_abs.sort_values(ascending=False).head(3).index.tolist()

# 해당 변수들의 월별 평균값 합계 (monthly_shap 기준)
monthly_shap_top3_sum = monthly_shap[top3_features].sum(axis=1)

# 공통 월만 사용하여 monthly_hazards와 정렬
common_months = monthly_shap_top3_sum.index.intersection(pd.PeriodIndex(monthly_hazards.keys(), freq="M"))
hazard_series = pd.Series(monthly_hazards).astype(float)
hazard_series.index = pd.PeriodIndex(hazard_series.index, freq="M")

aligned_hazard = hazard_series[common_months]
aligned_shap = monthly_shap_top3_sum[common_months]

# 상관계수 계산
r, p = pearsonr(aligned_hazard.values, aligned_shap.values)
print(f"📊 Nelson-Aalen hazard와 SHAP Top-3 합계의 Pearson 상관계수: r = {r:.3f}, p = {p:.3f}")

# --------------------------------------------------
# 6) 월별 누적 모델 기반 C-index 및 IBS 저장
# --------------------------------------------------

from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
from scipy.stats import norm
import xgboost as xgb
import numpy as np
import pandas as pd

cindex_records = []
ibs_records = []

for i, month in enumerate(month_labels):
    df_window = df[df["issue_month"] <= month]
    if df_window.shape[0] < 300:
        continue

    X_window = df_window[features]
    y_window = df_window[["T", "E"]]

    X_window = X_window.copy()
    y_window = y_window.copy()

    # C-index
    cidx = concordance_index(
        y_window["T"], 
        -monthly_shap_dynamic[i].values @ X_window[monthly_shap_dynamic[i].index].T.values,  # linear SHAP proxy score
        y_window["E"]
    )
    cindex_records.append({"issue_month": month, "c_index": cidx})

    # IBS (생존확률 기반)
    y_sksurv = Surv.from_arrays(event=y_window["E"].astype(bool), time=y_window["T"])
    # Use the model from last training iteration for prediction
    # For this, we need to re-train or store model_window from above; assuming model_window is last trained model
    # But model_window is overwritten in the loop, so to use correct model, we can re-train or skip
    # For simplicity, use model_window from last iteration (month_labels[-1])
    # So we re-train here for each month to get model_window
    y_lower = np.where(y_window["E"] == 1, y_window["T"], -np.inf)
    y_upper = y_window["T"]
    dtrain_window = xgb.DMatrix(data=X_window, label=y_upper)
    dtrain_window.set_float_info("label_lower_bound", y_lower)
    dtrain_window.set_float_info("label_upper_bound", y_upper)
    params = {
        "objective": "survival:aft",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "random_state": 42,
        "nthread": -1,
        "verbosity": 1
    }
    model_window = xgb.train(
        params=params,
        dtrain=dtrain_window,
        num_boost_round=100
    )
    pred = model_window.predict(xgb.DMatrix(X_window))
    sigma = params["aft_loss_distribution_scale"]
    t_min = y_window["T"].min()
    t_max = y_window["T"].max()
    times = np.linspace(t_min, t_max * 0.999, 50)

    estimate = np.zeros((len(pred), len(times)))
    for j, t in enumerate(times):
        estimate[:, j] = 1 - norm.cdf(np.log(t), loc=np.log(pred), scale=sigma)

    ibs = integrated_brier_score(y_sksurv, y_sksurv, estimate, times)
    ibs_records.append({"issue_month": month, "ibs": ibs})

# Save results
cindex_df = pd.DataFrame(cindex_records)
cindex_df.to_csv("../../reports/monthly_cindex.csv", index=False)

ibs_df = pd.DataFrame(ibs_records)
ibs_df.to_csv("../../reports/monthly_ibs.csv", index=False)
print("📁 월별 C-index 및 IBS 저장 완료")