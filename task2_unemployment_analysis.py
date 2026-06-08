# ============================================================
#  TASK 2: Unemployment Analysis with Python
#  CodeAlpha Data Science Internship
#  Author: [Your Name] | GitHub: CodeAlpha_UnemploymentAnalysis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Create Realistic Dataset ───────────────────────────────
# (Since original dataset link is in the internship portal,
#  we recreate a realistic India unemployment dataset with
#  Covid-19 impact — swap with pd.read_csv("your_file.csv")
#  when you have the actual dataset)

print("=" * 60)
print("    UNEMPLOYMENT ANALYSIS — CodeAlpha")
print("=" * 60)

np.random.seed(42)

regions = [
    "Haryana", "Rajasthan", "Delhi", "Himachal Pradesh",
    "Bihar", "Uttar Pradesh", "Tripura", "Jharkhand",
    "Andhra Pradesh", "Assam", "Kerala", "Tamil Nadu",
    "Maharashtra", "Karnataka", "Gujarat", "West Bengal",
    "Punjab", "Odisha", "Goa", "Telangana",
]

dates = pd.date_range(start="2019-01-01", end="2021-12-01", freq="MS")
rows = []

for region in regions:
    base = np.random.uniform(5, 15)
    for date in dates:
        # Covid surge: April–June 2020 and April–May 2021
        if date >= pd.Timestamp("2020-04-01") and date <= pd.Timestamp("2020-06-01"):
            rate = base + np.random.uniform(10, 25)
        elif date >= pd.Timestamp("2021-04-01") and date <= pd.Timestamp("2021-05-01"):
            rate = base + np.random.uniform(5, 15)
        elif date >= pd.Timestamp("2020-07-01") and date <= pd.Timestamp("2020-12-01"):
            rate = base + np.random.uniform(2, 8)
        else:
            rate = base + np.random.uniform(-2, 3)
        rows.append({
            "Date": date,
            "Region": region,
            "Estimated Unemployment Rate (%)": round(max(0, rate), 2),
            "Estimated Employed": int(np.random.uniform(5e6, 50e6)),
            "Estimated Labour Participation Rate (%)": round(np.random.uniform(35, 55), 2),
            "Area": np.random.choice(["Rural", "Urban"], p=[0.6, 0.4]),
        })

df = pd.DataFrame(rows)

# ── If you have the actual CSV, replace above with: ──────────
# df = pd.read_csv("Unemployment in India.csv")
# df["Date"] = pd.to_datetime(df["Date"])
# ─────────────────────────────────────────────────────────────

print("\n📊 Dataset Shape:", df.shape)
print("\n🔍 First 5 rows:\n", df.head())
print("\n📈 Basic Statistics:\n", df.describe())
print("\n❓ Missing Values:\n", df.isnull().sum())

# ── 2. Data Cleaning ──────────────────────────────────────────
df.dropna(inplace=True)
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Month_Name"] = df["Date"].dt.strftime("%b")

col = "Estimated Unemployment Rate (%)"

# ── 3. National Average Over Time ────────────────────────────
national = df.groupby("Date")[col].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(national["Date"], national[col], color="#2c3e50",
        linewidth=2, marker="o", markersize=3, label="National Avg")
ax.fill_between(national["Date"], national[col], alpha=0.15, color="#3498db")

# Annotate Covid waves
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
           alpha=0.2, color="#e74c3c", label="Covid Wave 1")
ax.axvspan(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-06-30"),
           alpha=0.2, color="#e67e22", label="Covid Wave 2")
ax.axvline(pd.Timestamp("2020-03-25"), color="#e74c3c",
           linestyle="--", linewidth=1.2, label="Lockdown Start")

ax.set_title("India Unemployment Rate — Jan 2019 to Dec 2021",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
plt.tight_layout()
plt.savefig("unemployment_national_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ National trend chart saved.")

# ── 4. Region-wise Average Unemployment ──────────────────────
region_avg = df.groupby("Region")[col].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(region_avg.index, region_avg.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(region_avg))),
               edgecolor="white")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{bar.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")
plt.title("Average Unemployment Rate by Region (2019–2021)",
          fontsize=13, fontweight="bold")
plt.ylabel("Avg Unemployment Rate (%)")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("unemployment_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Region-wise chart saved.")

# ── 5. Urban vs Rural Comparison ─────────────────────────────
area_trend = df.groupby(["Date", "Area"])[col].mean().reset_index()

plt.figure(figsize=(13, 5))
colors_area = {"Urban": "#e74c3c", "Rural": "#2ecc71"}
for area, grp in area_trend.groupby("Area"):
    plt.plot(grp["Date"], grp[col], label=area,
             color=colors_area[area], linewidth=2)
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
            alpha=0.15, color="red", label="Covid Wave 1")
plt.title("Urban vs Rural Unemployment Rate Over Time",
          fontsize=13, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("unemployment_urban_vs_rural.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Urban vs Rural chart saved.")

# ── 6. Monthly Heatmap (Seasonal Patterns) ───────────────────
pivot = df.pivot_table(values=col, index="Year",
                        columns="Month_Name", aggfunc="mean")
month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

plt.figure(figsize=(13, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, cbar_kws={"label": "Unemployment Rate (%)"})
plt.title("Monthly Heatmap of Unemployment Rate (Seasonal Trends)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("unemployment_monthly_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Monthly heatmap saved.")

# ── 7. Covid Impact — Before vs During vs After ───────────────
def covid_phase(date):
    if date < pd.Timestamp("2020-03-01"):
        return "Pre-Covid (2019)"
    elif date <= pd.Timestamp("2020-12-31"):
        return "During Covid (2020)"
    else:
        return "Post-Covid (2021)"

df["Phase"] = df["Date"].apply(covid_phase)
phase_avg = df.groupby("Phase")[col].mean()
phase_order = ["Pre-Covid (2019)", "During Covid (2020)", "Post-Covid (2021)"]
phase_avg = phase_avg.reindex(phase_order)

plt.figure(figsize=(8, 5))
bars = plt.bar(phase_avg.index, phase_avg.values,
               color=["#2ecc71", "#e74c3c", "#f39c12"],
               edgecolor="white", width=0.5)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{bar.get_height():.2f}%", ha="center", fontsize=11,
             fontweight="bold")
plt.title("Covid-19 Impact on Unemployment Rate",
          fontsize=13, fontweight="bold")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("unemployment_covid_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Covid impact chart saved.")

# ── 8. Correlation Analysis ───────────────────────────────────
num_df = df[[col, "Estimated Labour Participation Rate (%)", "Month"]].copy()
plt.figure(figsize=(6, 4))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("unemployment_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Correlation heatmap saved.")

# ── 9. Key Insights ──────────────────────────────────────────
print("\n" + "=" * 60)
print("   📌 KEY INSIGHTS")
print("=" * 60)
for phase in phase_order:
    print(f"  {phase}: {phase_avg[phase]:.2f}% avg unemployment")

highest = region_avg.idxmax()
lowest  = region_avg.idxmin()
print(f"\n  🔴 Highest Unemployment Region: {highest} ({region_avg[highest]:.2f}%)")
print(f"  🟢 Lowest  Unemployment Region: {lowest}  ({region_avg[lowest]:.2f}%)")
print(f"\n  📈 Overall Avg: {df[col].mean():.2f}%")
print(f"  📉 Min Rate  : {df[col].min():.2f}%")
print(f"  📈 Max Rate  : {df[col].max():.2f}%")

print("\n" + "=" * 60)
print("  ✅ TASK 2 COMPLETE — All outputs saved!")
print("=" * 60)
