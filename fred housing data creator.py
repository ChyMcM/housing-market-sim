import pandas as pd
from fredapi import Fred

# Put your FRED API key here
fred = Fred(api_key="yourfredapikey")

# Series codes from FRED
series = {
    "HPI": "CSUSHPINSA",        # S&P/Case-Shiller U.S. National Home Price Index
    "RATE_30Y": "MORTGAGE30US", # 30-Year Fixed Rate Mortgage Average
    "UNEMP": "UNRATE",          # Unemployment Rate
    "INCOME": "MEHOINUSA672N",  # Real Median Household Income (annual -> will forward fill monthly)
    "PERMITS": "PERMIT",        # New Private Housing Units Authorized by Building Permits
    "RENT_INDEX": "CUSR0000SEHA", # Rent of primary residence CPI
}

# Download each series
df = pd.DataFrame()
for name, code in series.items():
    df[name] = fred.get_series(code)

# Align on monthly dates
df.index = pd.to_datetime(df.index)
df = df.resample("M").mean()

# Fix annual data (income) -> forward fill across months
df["INCOME"] = df["INCOME"].ffill()

# Reset index to column
df = df.reset_index().rename(columns={"index": "date"})

# Save for your model script
df.to_csv("real_housing_data.csv", index=False)

print("Saved real_housing_data.csv with shape:", df.shape)

print(df.head())
