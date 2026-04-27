#   INDIA AIR QUALITY ANALYSIS AND PREDICTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 55)
print("   INDIA AIR QUALITY ANALYSIS AND PREDICTION")
print("=" * 55)

# LOAD DATA
df = pd.read_csv("India_Air_Quality_Insights_Dataset.csv")

# CLEAN DATA
df.replace("NA", np.nan, inplace=True)

df["pollutant_min"] = pd.to_numeric(df["pollutant_min"], errors="coerce")
df["pollutant_max"] = pd.to_numeric(df["pollutant_max"], errors="coerce")
df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# DATA INFO
print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Summary ---")
print(df.describe())

print("\n--- First 5 Rows ---")
print(df.head())

# BASIC ANALYSIS
print("\n" + "=" * 55)
print("   BASIC ANALYSIS")
print("=" * 55)

pollutant_mean = df.groupby("pollutant_id")["pollutant_avg"].mean()
print("\nMost Dominant Pollutant:", pollutant_mean.idxmax())
print(pollutant_mean.sort_values(ascending=False))

city_mean = df.groupby("city")["pollutant_avg"].mean()
print("\nMost Polluted City:", city_mean.idxmax())
print(city_mean.sort_values(ascending=False).head(10))

# PREP DATA FOR GRAPHS
top_cities = city_mean.sort_values(ascending=False).head(10)
pollutant_avg_by_type = pollutant_mean
state_mean = df.groupby("state")["pollutant_avg"].mean().sort_values(ascending=False).head(10)

# GRAPH 1 - BAR
plt.figure()
plt.bar(top_cities.index, top_cities.values)
plt.title("Top Cities")
plt.xticks(rotation=45)
plt.figtext(0.5, 0.01, "Bar Chart", ha="center")
plt.show()

# GRAPH 2 - LINE
plt.figure()
plt.plot(pollutant_avg_by_type.index, pollutant_avg_by_type.values, marker="o")
plt.title("Pollutant Type Avg")
plt.xticks(rotation=45)
plt.figtext(0.5, 0.01, "Line Graph", ha="center")
plt.show()

# GRAPH 3 - SCATTER
plt.figure()
plt.scatter(df["pollutant_min"], df["pollutant_max"])
plt.title("Min vs Max")
plt.xlabel("Min")
plt.ylabel("Max")
plt.figtext(0.5, 0.01, "Scatter Plot", ha="center")
plt.show()

# GRAPH 4 - HISTOGRAM
plt.figure()
plt.hist(df["pollutant_avg"], bins=30)
plt.title("Distribution")
plt.figtext(0.5, 0.01, "Histogram", ha="center")
plt.show()

# GRAPH 5 - PIE
pollutant_sum = df.groupby("pollutant_id")["pollutant_avg"].sum()
plt.figure()
plt.pie(pollutant_sum.values, labels=pollutant_sum.index, autopct="%1.1f%%")
plt.title("Pollutant Share")
plt.figtext(0.5, 0.01, "Pie Chart", ha="center")
plt.show()

# GRAPH 6 - HEATMAP
plt.figure()
corr = df[["pollutant_min", "pollutant_max", "pollutant_avg"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Heatmap")
plt.figtext(0.5, 0.01, "Heatmap", ha="center")
plt.show()

# GRAPH 7 - BOXPLOT (FIXED ERROR HERE)
top_pollutants = df["pollutant_id"].value_counts().head(6).index
df_box = df[df["pollutant_id"].isin(top_pollutants)]

plt.figure()
sns.boxplot(x="pollutant_id", y="pollutant_avg", data=df_box, color="skyblue")
plt.title("Box Plot (Outliers)")
plt.figtext(0.5, 0.01, "Box Plot", ha="center")
plt.show()

# GRAPH 8 - HORIZONTAL BAR
plt.figure()
plt.barh(state_mean.index, state_mean.values)
plt.title("Top States")
plt.figtext(0.5, 0.01, "Horizontal Bar Chart", ha="center")
plt.show()

# MACHINE LEARNING
print("\n" + "=" * 55)
print("   LINEAR REGRESSION MODEL RESULTS")
print("=" * 55)

X = df[["pollutant_min", "pollutant_max"]]
y = df["pollutant_avg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

print("\nActual vs Predicted:")
print(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]}))

# REGRESSION GRAPH
plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Regression Plot")
plt.figtext(0.5, 0.01, "Regression Plot", ha="center")
plt.show()

print("\n" + "=" * 55)
print("   ANALYSIS COMPLETE!")
print("=" * 55)
