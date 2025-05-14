import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, logistic, gumbel_r
import pandas as pd

# 데이터 불러오기
df = pd.read_parquet("../../data/processed/lendingclub_features_for_rf.parquet")
print("✅ Data loaded:", df.shape)

# T 컬럼 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df["T"], bins=50, color="skyblue", edgecolor="black", density=True)
plt.title("Distribution of T (Observed Time)")
plt.xlabel("T")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# x 범위 설정
x = np.linspace(-5, 5, 1000)

# 분포 정의 (평균 0, 스케일 1)
pdf_normal = norm.pdf(x, loc=0, scale=1)
pdf_logistic = logistic.pdf(x, loc=0, scale=1)
pdf_extreme = gumbel_r.pdf(x, loc=0, scale=1)  # Gumbel (minimum) 분포

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_normal, label="Normal", color="blue", linewidth=2)
plt.plot(x, pdf_logistic, label="Logistic", color="green", linewidth=2)
plt.plot(x, pdf_extreme, label="Extreme (Gumbel)", color="red", linewidth=2)

plt.title("XGBoost AFT Distributions (Mean=0, Scale=1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()