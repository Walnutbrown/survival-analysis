import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. NA 누적위험 증가량
na_delta = pd.read_csv("../../reports/monthly_na_delta.csv")
plt.figure(figsize=(12, 5))
plt.plot(na_delta["month"], na_delta["delta"], marker='o')
plt.title("Monthly Increase in NA Cumulative Hazard")
plt.xlabel("Month")
plt.ylabel("Δ NA Hazard")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. SHAP 평균값 (Top 5)
shap_mean = pd.read_csv("../../reports/monthly_shap_dynamic.csv", index_col=0)
top5_features = shap_mean.abs().mean().sort_values(ascending=False).head(5).index
plt.figure(figsize=(12, 5))
for feature in top5_features:
    plt.plot(shap_mean.index, shap_mean[feature], label=feature)
plt.title("Top 5 SHAP Mean Values Over Time")
plt.xlabel("Month")
plt.ylabel("Mean SHAP Value")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. SHAP 표준편차 (Top 5)
shap_std = pd.read_csv("../../reports/monthly_shap_dynamic_std.csv", index_col=0)
plt.figure(figsize=(12, 5))
for feature in top5_features:
    plt.plot(shap_std.index, shap_std[feature], label=feature)
plt.title("Top 5 SHAP Standard Deviation Over Time")
plt.xlabel("Month")
plt.ylabel("SHAP Std (Uncertainty)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. SHAP Longform Lineplot (Top 10)
shap_long = pd.read_csv("../../reports/monthly_top10_shap_longform.csv")
top10_features_long = shap_long.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(10).index
shap_long_top10 = shap_long[shap_long["feature"].isin(top10_features_long)]

plt.figure(figsize=(14, 6))
sns.lineplot(data=shap_long_top10, x="issue_month", y="mean_abs_shap", hue="feature")
plt.title("Top 10 SHAP Mean Contribution per Feature Over Time")
plt.xlabel("Month")
plt.ylabel("Mean Absolute SHAP Value")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

# 5. C-index
cindex = pd.read_csv("../../reports/monthly_cindex.csv")
plt.figure(figsize=(12, 5))
plt.plot(cindex["issue_month"], cindex["c_index"], marker='o')
plt.title("C-index Over Time")
plt.xlabel("Month")
plt.ylabel("C-index")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. IBS
ibs = pd.read_csv("../../reports/monthly_ibs.csv")
plt.figure(figsize=(12, 5))
plt.plot(ibs["issue_month"], ibs["ibs"], marker='o', color='green')
plt.title("Integrated Brier Score (IBS) Over Time")
plt.xlabel("Month")
plt.ylabel("IBS")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()