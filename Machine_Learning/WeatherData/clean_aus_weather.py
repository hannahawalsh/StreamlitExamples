"""
Clean the Australian Weather Data Set found here:
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
"""
### Imports 
import pandas as pd 
import numpy as np 



### Load in the data (saved to same directory as this file)
print("\nCleaning Australian Weather Data")
aus = pd.read_csv("weatherAUS.csv", parse_dates=["Date"],
                  infer_datetime_format=True)

# Ordinal directions to angles in radians
directions = ["E", "ENE", "NE", "NNE", 
              "N", "NNW", "NW", "WNW", 
              "W", "WSW", "SW", "SSW",
              "S", "SSE", "SE", "ESE"]
dir_to_rad = {d: i*np.pi / 8 for i, d 
              in enumerate(directions)}

# Drop bad data
aus = (aus.drop(columns=["RISK_MM"])
          .dropna(subset=["Location", "RainToday", "RainTomorrow"])
          .dropna(thresh=int(0.75 * len(aus.columns)), axis="index")
          .dropna(thresh=int(0.75 * len(aus)), axis="columns")
      )

### Fill missing data 
def impute_sameday(row) -> pd.Series:
    """
    For data columns that have both a 9am and a 3pm reading, if one is null,
    fill it with the other. 
    """
    dual_day_cols = [c.replace("3pm", "") for c in row.index if "3pm" in c]
    for v in dual_day_cols:
        if pd.isna(row[f"{v}3pm"]) and pd.notnull(row[f"{v}9am"]):
            row[f"{v}3pm"] = row[f"{v}9am"]
        elif pd.notnull(row[f"{v}3pm"]) and pd.isna(row[f"{v}9am"]):
            row[f"{v}9am"] = row[f"{v}3pm"]
    return row

# If a column is null for an enire location, drop that column 
drop_since_na = set()
for nm, gp in aus.groupby("Location"):
    nas = gp.isna().mean()
    for c in nas[nas == 1].index:
        drop_since_na.add(c)

# Fill missing values by interpolating between closest dates with values 
aus = (aus.drop(columns=list(drop_since_na))
          .fillna({"Rainfall": 0.0})
          .apply(impute_sameday, axis="columns")
          .set_index("Date").sort_index()     
          .groupby("Location")
          .apply(lambda group: group.interpolate(limit_direction="both"))
          .replace({**dir_to_rad, **{"Yes": True, "No": False}})
      )
aus = (aus.reset_index().sort_values(["Location", "Date"])
          .reset_index(drop=True).dropna())

# Instead of using date, use day of year (ignore that it's a time series)
aus["DayOfYear"] = aus["Date"].dt.dayofyear
aus = aus.drop(columns=["Date"])

# There are a lot of cities, which would expand our data a lot if one-hot 
# encoding. So instead, we'll use lat/long 
cities = pd.read_json("australian_cities.json", orient="index")
aus = (aus.merge(cities, left_on="Location", right_index=True, how="left")
          .drop(columns="Location"))

### Save the data 
aus.to_csv("cleaned_weather.csv", index=False)
print("DONE!\n")