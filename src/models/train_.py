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

for month, group_df in df.groupby("issue_month"):
    if len(group_df) < 100:
        continue
    try:
        naf.fit(group_df["T"], event_observed=group_df["E"])
        cum_hazard = naf.cumulative_hazard_
        # 순간 hazard ≈ 누적 hazard 차분
        inst_hazard = cum_hazard.diff().fillna(0)
        monthly_hazards[str(month)] = inst_hazard.mean().values[0]
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
# 5) XGBoost AFT 모델 학습 & SHAP 분석
# --------------------------------------------------
import xgboost as xgb


# AFT용 label 구성
df_model = df.loc[X.index]
y_lower = np.where(df_model["E"] == 1, df_model["T"], -np.inf)
y_upper = df_model["T"]

# DMatrix 구성
dtrain = xgb.DMatrix(data=X, label=y_upper)
dtrain.set_float_info("label_lower_bound", y_lower)
dtrain.set_float_info("label_upper_bound", y_upper)

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

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100
)

# SHAP 계산
explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional", approximate=True)
shap_values = explainer.shap_values(X)

# 월별 평균 SHAP 계산
X["issue_month"] = df["issue_d"].dt.to_period("M")
shap_df = pd.DataFrame(shap_values, columns=features)
shap_df["issue_month"] = X["issue_month"]
monthly_shap = shap_df.groupby("issue_month")[features].mean()
print("✅ XGBoost AFT 기반 월별 SHAP 계산 완료")

# --------------------------------------------------
# 6) 테스트 성능
# --------------------------------------------------

# ✅ 6-1. Concordance Index 계산
from lifelines.utils import concordance_index

c_index = concordance_index(y_test["T"], -model.predict(xgb.DMatrix(X_test)), y_test["E"])
print(f"Concordance Index (C-index): {c_index:.4f}")

# ✅ 6-2. Integrated Brier Score 계산 (scikit-survival 필요)
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

# (1) scikit-survival 형식으로 데이터 변환
y_train_sksurv = Surv.from_arrays(event=y_train["E"].astype(bool), time=y_train["T"])
y_test_sksurv  = Surv.from_arrays(event=y_test["E"].astype(bool), time=y_test["T"])

# (2) AFT 모델의 예측값 사용
predicted = model.predict(xgb.DMatrix(X_test))

# (3) IBS 계산 (예: 테스트 기간 내 분위수 기반 시점 설정)
times = np.percentile(y_test["T"], np.linspace(10, 90, 50))
ibs = integrated_brier_score(y_train_sksurv, y_test_sksurv, predicted, times)
print(f"Integrated Brier Score (IBS): {ibs:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# 시각화: 각 그룹에서 top feature들의 SHAP 총합 비중 (비율 기반 중요도)
monthly_df = monthly_shap
mean_abs = monthly_df.abs().sum()
mean_abs = mean_abs / mean_abs.sum()  # Normalize to sum=1
top10 = mean_abs.sort_values(ascending=False).head(10)

# SHAP 중요도 바 플롯
plt.figure(figsize=(6, 4))
sns.barplot(x=top10.values, y=top10.index)
plt.title(f"Top 10 features SHAP importance")
plt.xlabel("SHAP importance")
plt.ylabel("feature")
plt.tight_layout()
plt.show()

# 시각화: Top 10 feature들의 월별 SHAP 평균값 추이
plt.figure(figsize=(12, 6))
for col in top10.index:
    sns.lineplot(x=monthly_df.index.astype(str), y=monthly_df[col], label=col)
plt.title(f"Monthly SHAP average for Top 10 features")
plt.xlabel("issue_month")
plt.ylabel("Average SHAP value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()