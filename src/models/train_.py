from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter
import warnings
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../../data/processed/lendingclub_features_for_rf.parquet"
FEATURE_PATH = "../../data/processed/features_final_list_rf.csv"

# --------------------------------------------------
# 1) 데이터 로드 & 필터
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], errors="coerce")

# ▼▼▼ 추가 코드 ▼▼▼  
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

exclude_types = ["object", "datetime64[ns]"]
features = [col for col in features if str(df[col].dtype) not in exclude_types]

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
# 5) Random Survival Forest 모델 학습 & 평가 (covid_exposure 그룹별)
# --------------------------------------------------

print("▶ 훈련 가능 데이터 현황:")
print(f"- 총 행 수: {len(df)}")
print(f"- issue_d ≥ 2019-05-01: {len(df[df['issue_d'] >= '2015-01-01'])}")
print(f"- covid_exposure=0: {len(df[df['covid_exposure']==0])}")

# Removed the original loop over "early" and "late" subgroups for model training and SHAP calculation

from sklearn.utils import resample
import shap

print("\n🔍 전체 데이터 기반 SHAP 분석")
print(f"X shape: {X.shape}, df[['T', 'E']] shape: {df[['T', 'E']].shape}")
shap_runs = []
from collections import Counter
for i in range(10):  # bootstrap iterations
    X_bs, y_bs = resample(X, df[["T", "E"]], replace=True, random_state=42 + i)
    y_surv_bs = Surv.from_arrays(event=y_bs["E"].astype(bool), time=y_bs["T"])
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                               min_samples_leaf=15, max_features="sqrt",
                               n_jobs=-1, random_state=42 + i)
    rsf.fit(X_bs, y_surv_bs)
    print(f"✅ 부트스트랩 {i+1}/10 학습 완료")
    shap_runs.append(rsf.feature_importances_)
    if i == 0:
        top_idx_list = []
    top_idx = np.argsort(rsf.feature_importances_)[::-1][:5]
    top_idx_list.extend(top_idx)

top_5_idx = [item[0] for item in Counter(top_idx_list).most_common(5)]
top_5_features = [features[i] for i in top_5_idx]
print(f"🔝 전체 기준 Top 5 features:", top_5_features)

rsf_final = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                 min_samples_leaf=15, max_features="sqrt",
                                 n_jobs=-1, random_state=999)
y_final = Surv.from_arrays(event=df["E"].astype(bool), time=df["T"])
rsf_final.fit(X, y_final)
print("✅ 최종 RSF 학습 완료")


    # Restrict X to top 5 features for SHAP computation
X_top5 = X[top_5_features].copy()

# ▼▼▼ SHAP 계산 최적화를 위한 다운샘플링 (E 비율 유지) ▼▼▼
from sklearn.model_selection import train_test_split

X_top5["T"] = df["T"]
X_top5["E"] = df["E"]

# Stratified downsampling to 20,000 rows maintaining event proportion
X_top5_sampled, _ = train_test_split(
    X_top5,
    train_size=20000,
    stratify=X_top5["E"],
    random_state=999
)

y_top5_sampled = Surv.from_arrays(
    event=X_top5_sampled["E"].astype(bool),
    time=X_top5_sampled["T"]
)

# Drop T, E from features for SHAP
X_top5_sampled = X_top5_sampled.drop(columns=["T", "E"])

explainer = shap.TreeExplainer(rsf_final)
shap_values = explainer.shap_values(X_top5_sampled)
print("✅ SHAP 계산 완료 (샘플 20,000건 기준)")

df["issue_month"] = df["issue_d"].dt.to_period("M")
shap_df = pd.DataFrame(shap_values, columns=top_5_features)
shap_df["issue_month"] = df["issue_month"].values
mean_by_month = shap_df.groupby("issue_month")[top_5_features].mean()
print("✅ 월별 SHAP 평균값 계산 완료")
top_features_by_month = {"all": mean_by_month}

# 저장 또는 시각화용 결과 준비됨

# --------------------------------------------------
# 6) 테스트 성능
# --------------------------------------------------
# (Removed CoxPHFitter test performance and variable selection saving as per instructions)

import matplotlib.pyplot as plt
import seaborn as sns

# 시각화: 월별 top feature들의 SHAP 평균값 추이
monthly_df = top_features_by_month["all"]
plt.figure(figsize=(12, 6))
for col in monthly_df.columns:
    sns.lineplot(data=monthly_df[col], label=col)
plt.title(f"월별 SHAP 평균값 추이")
plt.xlabel("issue_month")
plt.ylabel("평균 SHAP 값")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 시각화: 각 그룹에서 top feature들의 SHAP 총합 비중 (비율 기반 중요도)
mean_abs = monthly_df.abs().sum()
mean_abs = mean_abs / mean_abs.sum()  # Normalize to sum=1
plt.figure(figsize=(6, 4))
sns.barplot(x=mean_abs.values, y=mean_abs.index)
plt.title(f"Top 5 변수 SHAP 비중")
plt.xlabel("SHAP 비중")
plt.ylabel("변수명")
plt.tight_layout()
plt.show()