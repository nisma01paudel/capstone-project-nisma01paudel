import pandas as pd

def load_climate_data():
    return pd.read_csv('data/climate_data/climate_data_nepal_district_wise_monthly.csv')

def load_air_quality_data():
    return pd.read_csv('data/climate_data/openaq_location_3460_measurments.csv')

def load_daily_climate_data():
    return pd.read_csv('data/climate_data/dailyclimate.csv')

def load_nepal_rainfall_data():
    return pd.read_csv('data/climate_data/nepal_rainfall.csv', low_memory=False)

def load_climate_normal_data():
    return pd.read_csv('data/climate_data/Climate Normal for 20 Different Stations in Nepal1980-2010.csv')

def load_forestfire_data():
    return pd.read_csv('data/fire/ForestFirecsv-1615631451511.csv')

def merge_data():
    climate_data = load_climate_data()
    air_quality_data = load_air_quality_data()
    daily_climate_data = load_daily_climate_data()
    rainfall_data = load_nepal_rainfall_data()
    climate_normal_data = load_climate_normal_data()
    forestfire_data = load_forestfire_data()

    # Drop rows with missing values
    rainfall_data.dropna(inplace=True)
    daily_climate_data.dropna(inplace=True)

    # Drop 50% of daily_climate_data randomly to reduce file size
    if len(daily_climate_data) > 10000:
        daily_climate_data = daily_climate_data.sample(frac=0.5, random_state=42)

    # Normalize date columns
    for df in [climate_data, daily_climate_data, rainfall_data]:
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'DATE'}, inplace=True)
        if 'date' in df.columns:
            df.rename(columns={'date': 'DATE'}, inplace=True)

    # Convert date columns
    for df in [climate_data, daily_climate_data, rainfall_data]:
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Start merging
    merged_data = climate_data.copy()

    if 'DATE' in daily_climate_data.columns:
        merged_data = merged_data.merge(daily_climate_data, on="DATE", how="left")

    if 'LAT' in air_quality_data.columns and 'LON' in air_quality_data.columns:
        if 'LAT' in merged_data.columns and 'LON' in merged_data.columns:
            merged_data = merged_data.merge(air_quality_data, on=["LAT", "LON"], how="left")

    if 'DATE' in rainfall_data.columns:
        merged_data = merged_data.merge(rainfall_data, on="DATE", how="left")

    if 'Location' in forestfire_data.columns and 'Location' in merged_data.columns:
        merged_data = merged_data.merge(forestfire_data, on="Location", how="left")

    if 'Station' in climate_normal_data.columns and 'Station' in merged_data.columns:
        merged_data = merged_data.merge(climate_normal_data, on="Station", how="left")

    # Drop unnecessary columns
    columns_to_drop = [col for col in merged_data.columns if 'ID' in col or 'Code' in col or 'Unnamed' in col or 'Serial' in col]
    merged_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Drop remaining missing values
    merged_data.dropna(inplace=True)

    # Save optimized dataset
    merged_data.to_csv("data/merged_cleaned.csv", index=False, compression='gzip')

    return merged_data

# Test
if __name__ == "__main__":
    data = merge_data()
    print(data.shape)
    print(data.head())
