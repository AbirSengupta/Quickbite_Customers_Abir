
# # QuickBite Case Study: Understanding Customer Value
# Tools: Python, NumPy, pandas, Matplotlib, Seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
DATA_PATH = "quickbite_customers.csv"

# =====================================================================
# %% TASK 0: Setup and First Look
# =====================================================================
df = pd.read_csv("C:/Users/abirseng/Downloads/quickbite_customers.csv")
df["signup_date"] = pd.to_datetime(df["signup_date"])

print("Shape:", df.shape)
print(df.head())
df.info()
print(df.describe().T)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2),
})
print("\nMissing values per column:\n", missing[missing.missing_count > 0])

plt.figure(figsize=(10, 4))
sns.heatmap(df.isna(), cbar=False, yticklabels=False)
plt.title("Missing Values Map (light = missing)")
plt.xlabel("Column")
plt.tight_layout(); plt.show()

raw = df.copy()  

# =====================================================================

# %% TASK 2 (done first, as the brief suggests): Outlier Detection

# Impossible values / outliers are handled BEFORE imputation, so extreme values don't distort the medians we fill with.

# =====================================================================

out_cols = ["age", "delivery_time_min", "avg_order_value"]


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col in zip(axes, out_cols):

    sns.boxplot(y=df[col], ax=ax)

    ax.set_title(f"{col} (before)")

plt.tight_layout(); plt.show()



def iqr_outliers(s):

    q1, q3 = s.quantile([0.25, 0.75])

    iqr = q3 - q1

    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return (s < low) | (s > high), low, high



def z_outliers(s):

    z = (s - s.mean()) / s.std()

    return z.abs() > 3



print("\nOutlier counts: IQR vs Z-score")

for col in out_cols:

    s = df[col].dropna()

    iqr_mask, low, high = iqr_outliers(s)

    print(f"{col:20s} IQR: {iqr_mask.sum():3d} (bounds {low:.1f} to {high:.1f}) | "

          f"Z>3: {z_outliers(s).sum():3d}")

# IQR flags many genuine-but-high values in skewed columns;
# Z-score : (|z|>3) isolates only the truly extreme planted errors here.



print("\nImpossible ages:", df.loc[(df.age < 16) | (df.age > 90), "age"].tolist())

print("Top delivery times:", df.delivery_time_min.nlargest(5).tolist())

print("Top order values:", df.avg_order_value.nlargest(5).tolist())


df.loc[(df.age < 16) | (df.age > 90), "age"] = np.nan



#  FLAG for investigation, then CAP

df["extreme_delivery_flag"] = df.delivery_time_min > 180

_, _, d_high = iqr_outliers(df.delivery_time_min.dropna())

df["delivery_time_min"] = df.delivery_time_min.clip(upper=d_high)



# avg_order_value: Rs.18k-32k baskets distort everything -> CAP(winsorise) at P99

aov_cap = df.avg_order_value.quantile(0.99)

df["avg_order_value"] = df.avg_order_value.clip(upper=aov_cap)



# annual_spend: KEEP -- high spenders are genuine VIPs (needed for Task 7)



fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col in zip(axes, out_cols):

    sns.boxplot(y=df[col], ax=ax)

    ax.set_title(f"{col} (after)")

plt.tight_layout(); plt.show()

# SOLUTIONS: Missiing Chart : (https://snipboard.io/Z47njd.jpg)
 # Missing values per column:
 #                       missing_count  missing_pct
 # gender                       106         5.30
 # age                          139         6.95
#  delivery_time_min             80         4.00
#  avg_order_value              108         5.40
#  customer_rating              258        12.90
#  discount_used_pct            104         5.20

# Before Table : (https://snipboard.io/DgHP1u.jpg)
# After Table: (https://snipboard.io/xf8Gjr.jpg)

# Outlier counts: IQR vs Z-score
#    age                  IQR:   9 (bounds 12.0 to 52.0) | Z>3:   5
#   delivery_time_min    IQR:  46 (bounds 4.7 to 62.0) | Z>3:   4
#   avg_order_value      IQR:  96 (bounds -149.9 to 823.1) | Z>3:   4

#  The Z-score rule (|z| > 3) is more honest for this data. In skewed columns like delivery time and order value, the IQR rule flags dozens of real values: #  46 and 96. The Z-score rule flags only the 4 truly extreme planted values in each column. IQR works best on roughly symmetric data, and age is the only      
#  one of these columns that is close to normal.

#  An age of 0, 5 or 150 is impossible for a food-delivery customer, so these are data-entry errors. Capping would just replace one made-up number with 
#  another. Removing them and filling with the median gives a sensible value. Customers in their 50s are real and are kept.

#  A 500-minute delivery could be a real failure (a cancelled or stuck order), so we shouldn't just delete it. The flag column keeps a record of these 
#  customers for the operations team. Capping stops these few values from distorting the mean, the median used to fill gaps, and the charts.


# Impossible ages: [5.0, 150.0, 0.0]
#  Top delivery times: [500.0, 410.0, 320.0, 275.0, 83.9]
#  Top order values: [32000.0, 25000.0, 21000.0, 18000.0, 1729.0]


# =====================================================================

# %% TASK 1: Missing Value Imputation

# =====================================================================

# Is missingness related to long delivery times?

print("\nMedian delivery time when rating is missing vs present:")

print(raw.groupby(raw.customer_rating.isna())["delivery_time_min"].median()

      .rename({False: "rating present", True: "rating missing"}))

# -> slow deliveries have more missing ratings: data is NOT missing at random.



before = df.copy()



# Median (not mean) because the columns are skewed; the mean is pulled by tails.

df["age"] = df["age"].fillna(df["age"].median())

df["delivery_time_min"] = df["delivery_time_min"].fillna(df["delivery_time_min"].median())



# Group-wise median: basket sizes differ by city tier

df["avg_order_value"] = df["avg_order_value"].fillna(

    df.groupby("city_tier")["avg_order_value"].transform("median"))



# Rating: compare global vs group median, then use group median

print("\nRating median global:", df.customer_rating.median())

print("Rating median by tier:\n", df.groupby("city_tier").customer_rating.median())

df["customer_rating"] = df["customer_rating"].fillna(

    df.groupby("city_tier")["customer_rating"].transform("median"))



df["discount_used_pct"] = df["discount_used_pct"].fillna(df["discount_used_pct"].median())

df["gender"] = df["gender"].fillna("Unknown")  



num_missing = ["age", "delivery_time_min", "avg_order_value",

               "customer_rating", "discount_used_pct"]

before_after = pd.DataFrame({

    "missing_before": raw[num_missing + ["gender"]].isna().sum(),

    "missing_after": df[num_missing + ["gender"]].isna().sum(),

    "mean_before": raw[num_missing].mean().round(2),

    "mean_after": df[num_missing].mean().round(2),

    "median_before": raw[num_missing].median().round(2),

    "median_after": df[num_missing].median().round(2),

})

print("\nBefore vs after imputation:\n", before_after)



plt.figure(figsize=(8, 4))

sns.kdeplot(before["avg_order_value"].dropna(), label="Before imputation")

sns.kdeplot(df["avg_order_value"], label="After imputation", linestyle="--")

plt.title("avg_order_value: distribution before vs after imputation")

plt.xlabel("Average order value (INR)"); plt.legend()

plt.tight_layout(); plt.show()

# SOLUTION: KDE distribution before and after imputation: (https://snipboard.io/3vlHIF.jpg)


=====================================================================

# %% TASK 3: Distribution Shapes with KDE Plots

# =====================================================================

dist_cols = {

    "age": "Approximately normal",

    "customer_rating": "Left-skewed",

    "delivery_time_min": "Right-skewed",

    "avg_order_value": "Log-normal",

    "annual_spend": "Heavy right tail (Pareto-like)",

    "orders_last_12m": "Heavy right tail (Pareto-like)",

}



fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for ax, (col, shape) in zip(axes.flat, dist_cols.items()):

    sns.histplot(df[col], kde=True, ax=ax)

    ax.axvline(df[col].mean(), color="red", ls="--", label=f"Mean {df[col].mean():.1f}")

    ax.axvline(df[col].median(), color="green", ls="--", label=f"Median {df[col].median():.1f}")

    ax.axvline(df[col].round(0).mode()[0], color="black", ls=":", label=f"Mode {df[col].round(0).mode()[0]:.0f}")

    ax.set_title(f"{col}\n{shape} | skew = {df[col].skew():.2f}")

    ax.legend(fontsize=8)

plt.tight_layout(); plt.show()



print("\nSkewness:")

print("avg_order_value skew BEFORE outlier treatment:", round(raw.avg_order_value.skew(), 2))

for col in dist_cols:

    print(f"  {col:20s} {df[col].skew():6.2f}")



log_aov = np.log(df["avg_order_value"])

print("Skew of log(avg_order_value):", round(log_aov.skew(), 2), "-> close to normal")

plt.figure(figsize=(7, 4))

sns.histplot(log_aov, kde=True)

plt.title("log(avg_order_value) is approximately normal")

plt.xlabel("log(INR)"); plt.tight_layout(); plt.show()

# Right skew: mode < median < mean (tail pulls the mean up).

# Left skew : mean < median < mode (tail pulls the mean down).

# SOLUTION: Histogram with KDE curve : (https://snipboard.io/cBsRTN.jpg)

 #        Skewness:
 #     avg_order_value skew BEFORE outlier treatment: 21.99
 #     age                    0.13
 #     customer_rating       -1.82
 #     delivery_time_min      0.77
 #     avg_order_value        1.46
 #     annual_spend          15.24
 #     orders_last_12m        5.31
 #     Skew of log(avg_order_value): -0.05 -> close to normal




# =====================================================================

# %% TASK 4: Relationships Between Variables

# =====================================================================

numeric = df.select_dtypes("number").drop(columns=["extreme_delivery_flag"], errors="ignore")

cov = numeric.cov()

corr = numeric.corr()

print("\nCovariance matrix:\n", cov.round(1))



plt.figure(figsize=(11, 9))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)

plt.title("Correlation Matrix of Numeric Variables")

plt.tight_layout(); plt.show()



pairs = [("orders_last_12m", "annual_spend"),

         ("app_sessions_per_month", "orders_last_12m"),

         ("delivery_time_min", "customer_rating")]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

for ax, (x, y) in zip(axes, pairs):

    sns.scatterplot(data=df, x=x, y=y, ax=ax, s=12, alpha=0.5)

    ax.set_title(f"{x} vs {y} (r = {df[x].corr(df[y]):.2f})")

plt.tight_layout(); plt.show()



monthly = df.set_index("signup_date").resample("ME")

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

monthly.size().plot(ax=axes[0], marker="o")

axes[0].set_title("New Signups per Month"); axes[0].set_ylabel("Signups")

monthly["delivery_time_min"].mean().plot(ax=axes[1], marker="o", color="tab:orange")

axes[1].set_title("Avg Delivery Time by Signup Month"); axes[1].set_ylabel("Minutes")

plt.tight_layout(); plt.show()



mem_order = ["Basic", "Silver", "Gold", "Platinum"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

sns.boxplot(data=df, x="membership", y="annual_spend", order=mem_order, ax=axes[0])

axes[0].set_yscale("log"); axes[0].set_title("Annual Spend by Membership (log scale)")

sns.boxplot(data=df, x="city_tier", y="delivery_time_min", ax=axes[1])

axes[1].set_title("Delivery Time by City Tier")

plt.tight_layout(); plt.show()



spend_corr = corr["annual_spend"].drop("annual_spend").sort_values(key=abs, ascending=False)

print("\nCorrelation with annual_spend:\n", spend_corr.round(3))

print(f"\nANSWER Q4: strongest relationship with annual_spend = "

      f"{spend_corr.index[0]} (r = {spend_corr.iloc[0]:.2f}).")

print("Correlation does NOT prove causation -- it only shows the variables move together.")


# SOLUTIONS: 
  # Log Avg Value : Histogram: (https://snipboard.io/0EJPwC.jpg): Approx. normal
  # Correlation Matrix: (https://snipboard.io/dUAYyu.jpg)
  # Scatter Plots: (https://snipboard.io/9g5bRW.jpg)
  # Line Plot : (https://snipboard.io/LBA5Fc.jpg)
  # Box Plot: (https://snipboard.io/Ok5EpU.jpg)



#  Covariance matrix:
#                           city_tier     age  ...  orders_last_12m  annual_spend
#  city_tier                     0.6     0.1  ...              0.9        -102.2
#  age                           0.1    46.2  ...              2.9        5424.3
#  tenure_months                 0.1     1.1  ...             -7.7       -3633.6
#  app_sessions_per_month        0.1    -0.4  ...             14.0        9728.3
#  delivery_time_min             1.7     4.3  ...             -4.0       -2540.8
#  avg_order_value             -20.1     5.7  ...            192.3     1560970.3
#  customer_rating              -0.0    -0.0  ...             -0.2           4.3
#  discount_used_pct            -0.2   -16.0  ...              5.9       -8300.5
#  days_since_last_order         0.4    -1.9  ...              8.8       -9480.9
#  orders_last_12m               0.9     2.9  ...           2428.0     1083302.2
#  annual_spend               -102.2  5424.3  ...        1083302.2   920296124.8

#  Correlation with annual_spend:
#   orders_last_12m           0.725
#   avg_order_value           0.248
#   app_sessions_per_month    0.085
#   discount_used_pct        -0.030
#   age                       0.026
#   days_since_last_order    -0.016
#   tenure_months            -0.012
#   delivery_time_min        -0.008
#   city_tier                -0.004
#   customer_rating           0.000
#   Name: annual_spend, dtype: float64

# strongest relationship with annual_spend = orders_last_12m (r = 0.72).
# Correlation does NOT prove causation -- it only shows the variables move together

#  Covariance's size depends on the units of the two variables (for example, orders × rupees), so a value like 1,000,000 doesn't tell you whether the 
#  relationship is strong or weak, and you can't compare it across pairs. Correlation divides the covariance by both standard deviations, which puts it on a 
#  −1 to +1 scale with no units, so it shows both the direction and the strength (r = 0.72 for orders vs spend is clearly strong).


# =====================================================================

# %% TASK 5: Encoding Categorical Variables

# =====================================================================

enc = df.copy()



# Label / ordinal encoding: order is meaningful

enc["membership_encoded"] = enc["membership"].map(

    {"Basic": 0, "Silver": 1, "Gold": 2, "Platinum": 3})



# One-hot encoding; drop_first avoids the dummy variable trap

enc = pd.get_dummies(enc, columns=["device_type", "acquisition_channel"],

                     drop_first=True, dtype=int)



# Target encoding: city -> mean annual_spend of that city

city_means = df.groupby("city")["annual_spend"].mean()

enc["city_target_encoded"] = enc["city"].map(city_means).round(0)




enc["device_type"] = df["device_type"]

show = ["city", "city_target_encoded",        # target encoding
        "membership", "membership_encoded",   # label encoding
        "device_type", "device_type_iOS"]     # one-hot encoding

pd.set_option("display.max_columns", None)    # stop pandas hiding columns with ...
pd.set_option("display.width", 200)
print("\nEncoded columns:\n", enc[show].head(10))

# - Alphabetical label-encoding of city implies a fake order (Bengaluru < Delhi < ...).

# - One-hot on 10 cities adds 9-10 sparse columns (wide data).

# - Target encoding uses the target to build a feature -> risk of LEAKAGE.


#  SOLUTIONS:  
#  Encoded columns:
#         city  city_target_encoded membership  membership_encoded device_type  device_type_iOS
# 0    Lucknow              13940.0      Basic                   0     Android                0
# 1  Bengaluru              12004.0      Basic                   0         iOS                1
# 2     Indore              10733.0      Basic                   0         iOS                1
# 3     Jaipur              10411.0      Basic                   0     Android                0
# 4     Mumbai              12021.0      Basic                   0     Android                0
# 5      Kochi              15740.0      Basic                   0     Android                0
# 6    Lucknow              13940.0     Silver                   1         iOS                1
# 7    Lucknow              13940.0      Basic                   0         iOS                1
# 8     Mumbai              12021.0      Basic                   0         iOS                1
# 9  Bengaluru              12004.0     Silver                   1     Android                0


# =====================================================================

# %% TASK 6: Central Tendency and Percentiles

# =====================================================================

print("\nCentral tendency:")

for col in ["age", "orders_last_12m", "annual_spend"]:

    print(f"  {col:16s} mean={df[col].mean():10.1f}  median={df[col].median():10.1f}  "

          f"mode={df[col].round(0).mode()[0]:10.0f}")

print("  membership mode:", df["membership"].mode()[0])

print(f"annual_spend mean/median = {df.annual_spend.mean() / df.annual_spend.median():.2f}x "

      "-> a few heavy spenders pull the mean far above the typical customer.")



pcts = [10, 25, 50, 75, 90, 95, 99]

pct_table = pd.DataFrame(

    {c: np.percentile(df[c], pcts) for c in ["delivery_time_min", "annual_spend"]},

    index=[f"P{p}" for p in pcts]).round(1)

print("\nPercentiles:\n", pct_table)



plt.figure(figsize=(9, 3))

sns.boxplot(x=df["delivery_time_min"])

for q in [0.25, 0.5, 0.75]:

    v = df["delivery_time_min"].quantile(q)

    plt.text(v, -0.42, f"Q{int(q * 4)} = {v:.1f}", ha="center", fontsize=9)

plt.title("Delivery Time with Quartiles"); plt.xlabel("Minutes")

plt.tight_layout(); plt.show()



print(f"\nBusiness: 90% of customers get delivery in "

      f"{df.delivery_time_min.quantile(0.9):.1f} minutes or less (P90).")


# SOLUTION: Box Plot: ( https://snipboard.io/bmkTXh.jpg )
 
# Central tendency:
#  age              mean=      31.9  median=      32.0  mode=        32
#  orders_last_12m  mean=      34.9  median=      24.0  mode=        17
#  annual_spend     mean=   12995.9  median=    7142.0  mode=      2916
#  membership mode: Basic

#  annual_spend mean/median = 1.82x -> a few heavy spenders pull the mean far above the typical customer.

#  Why the mean is so much higher for annual -spend: 
 #  The median is the middle customer, so a few extreme values barely change it.
 #   The mean adds up every rupee and divides by 2,000. One customer spending ₹8.5 lakh adds about ₹428 to the mean by themselves.
 #   Most customers spend little and a few spend a huge amount. That long right tail gives the order mode < median < mean, which is the typical pattern for 
 #   right-skewed data.

# Insights: 
   # The "average customer" is misleading. A typical customer spends about ₹7,100 a year, not ₹13,000. Use the median for targets, budgets and per-customer 
   # planning.
   # Revenue depends on a few heavy users. The top 1% of customers bring in about 17.6% of revenue and the top 20% about 60.6% (Task 7). Losing even a few 
   # VIPs would noticeably cut revenue. 

# Percentiles:
#       delivery_time_min  annual_spend
# P10               22.1        2508.6
# P25               26.5        4085.8
# P50               32.2        7142.0
# P75               40.1       12781.2
# P90               49.2       22453.7
# P95               55.4       35355.2
# P99               62.0      109149.8

# Business: 90% of customers get delivery in 49.2 minutes or less (P90).



# =====================================================================

# %% TASK 7: Pareto Distribution and the 80/20 Rule

# =====================================================================

spend_sorted = df["annual_spend"].sort_values(ascending=False).reset_index(drop=True)

cum_share = spend_sorted.cumsum() / spend_sorted.sum()

cust_share = (np.arange(1, len(spend_sorted) + 1)) / len(spend_sorted)



n = len(spend_sorted)

# How many top customers are needed to reach 50% / 80% of revenue?
cust_for_50 = (cum_share < 0.5).sum() + 1
cust_for_80 = (cum_share < 0.8).sum() + 1

pct_for_50 = cust_for_50 / n * 100
pct_for_80 = cust_for_80 / n * 100

print(f"{pct_for_50:.1f}% of customers generate 50% of revenue")
print(f"{pct_for_80:.1f}% of customers generate 80% of revenue")

top20_share = cum_share[int(0.2 * len(spend_sorted)) - 1] * 100



print(f"           Top 20% of customers generate {top20_share:.1f}% of revenue "

      "(strong concentration, but not a full 80/20)")



fig, ax1 = plt.subplots(figsize=(11, 4.5))

ax1.bar(cust_share * 100, spend_sorted, width=0.06, color="tab:blue")

ax1.set_xlabel("% of customers (highest spend first)"); ax1.set_ylabel("Annual spend (INR)")

ax2 = ax1.twinx()

ax2.plot(cust_share * 100, cum_share * 100, color="tab:red", lw=2)

ax2.axhline(80, color="grey", ls=":"); ax2.axvline(20, color="grey", ls=":")

ax2.set_ylabel("Cumulative % of revenue")

plt.title("Pareto Chart of Customer Revenue"); plt.tight_layout(); plt.show()



fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(df["annual_spend"], bins=60, ax=axes[0])

axes[0].set_title("annual_spend (normal scale)")

sns.histplot(df["annual_spend"], bins=60, log_scale=True, ax=axes[1])

axes[1].set_title("annual_spend (log scale)")

plt.tight_layout(); plt.show()



# SOLUTION: Pareto Chart: ( https://snipboard.io/z3m64p.jpg )
  # Annual_Spend plot : ( https://snipboard.io/uLURmY.jpg )
 # 11.9% of customers generate 50% of revenue
 # 43.1% of customers generate 80% of revenue
          # Top 20% of customers generate 60.6% of revenue (strong concentration, but not a full 80/20)

# Normal scale: one tall bar + long invisible tail. Log scale spreads the data out and reveals a bell-like shape, so the bulk of customers become visible.


# =====================================================================

# %% TASK 8: PDF and CDF

# =====================================================================

def ecdf(series):

    x = np.sort(series.values)

    y = np.arange(1, len(x) + 1) / len(x)

    return x, y



fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for row, col in enumerate(["delivery_time_min", "age"]):

    sns.kdeplot(df[col], fill=True, ax=axes[row, 0])

    axes[row, 0].set_title(f"PDF of {col}")

    x, y = ecdf(df[col])

    axes[row, 1].plot(x, y)

    axes[row, 1].set_title(f"Empirical CDF of {col}")

    axes[row, 1].set_ylabel("Share of customers <= x")

axes[0, 1].axvline(45, color="red", ls="--", label="45 min"); axes[0, 1].legend()

plt.tight_layout(); plt.show()



dt = df["delivery_time_min"]

print(f"\nANSWER Q8: P(delivery > 45 min) = {(dt > 45).mean():.3f} ({(dt > 45).mean() * 100:.1f}%)")

print(f"P(delivery <= 30 min) = {(dt <= 30).mean():.3f}")

print(f"Median delivery = {dt.median():.1f} min | 95th percentile = {dt.quantile(0.95):.1f} min")

print(f"Age: median = {df.age.median():.1f} | 95th percentile = {df.age.quantile(0.95):.1f}")

# Repeat the CDF questions for age
ag = df["age"]
print(f"P(age > 45) = {(ag > 45).mean():.3f}")
print(f"P(age <= 30) = {(ag <= 30).mean():.3f}")
print(f"Age median = {ag.median():.1f} | 95th percentile = {ag.quantile(0.95):.1f}")

# Compare both CDFs on one chart (standardised so the scales match)
plt.figure(figsize=(8, 4))
for col in ["delivery_time_min", "age"]:
    z = (df[col] - df[col].mean()) / df[col].std()
    x = np.sort(z)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label=col)
plt.axhline(0.5, color="grey", ls=":")
plt.title("CDF shapes: age (normal) vs delivery time (right-skewed)")
plt.xlabel("Standardised value (z-score)"); plt.ylabel("Share of customers <= x")
plt.legend(); plt.tight_layout(); plt.show()




# SOLUTIONS: PDF and Empirical CDF Plots: ( https://snipboard.io/hHRrn4.jpg )
         #   Comparision of Age (Normal) and delivery -time CDF : ( https://snipboard.io/mxoqKC.jpg )
         # Age CDF is a roughly symmetric S-curve (steps because ages are whole numbers;
         # the jump at the median comes from median imputation).
         # Delivery CDF rises a little faster before the median and approaches 100%
         # more slowly (moderate right tail); the jumps come from median imputation and the 62-min cap.
         
         # PDF: how crowded the values are around a point. The higher the curve, the more customers have values near there. CDF: the share of customers at 
         #      or below a point. For example, the CDF of delivery time at 45 min is about 0.84, so 84% of deliveries take 45 minutes or less and about 16% 
         #      take longer. 




enc.to_csv("quickbite_clean.csv", index=False)

print("\nSaved quickbite_clean.csv with shape", enc.shape)





