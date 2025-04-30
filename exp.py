{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5e38a946",
   "metadata": {},
   "outputs": [],
   "source": [
    "# EDA and model training experimentation notebook\n",
    "import pandas as pd\n",
    "from utils.preprocess import merge_data\n",
    "from utils.model_training import train_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1ad4278f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading and merging datasets...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\Learner\\Desktop\\AI_WITH_OMDENA\\capstone-project-nisma01paudel\\utils\\preprocess.py:42: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.\n",
      "  \n",
      "c:\\Users\\Learner\\Desktop\\AI_WITH_OMDENA\\capstone-project-nisma01paudel\\utils\\preprocess.py:64: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "# --- Load and prepare data ---\n",
    "print(\"Loading and merging datasets...\")\n",
    "data = merge_data()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "cad82371",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 🔁 Reduce size to speed up experimentation\n",
    "data = data.sample(frac=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "18dad3c3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample data:\n",
      "              DATE  YEAR  MONTH      DISTRICT   LAT    LON  PRECTOT     PS  \\\n",
      "1124654 2010-01-31  2010      1   Nawalparasi  27.6  84.00     0.43  93.10   \n",
      "557097  1988-02-29  1988      2         Gulmi  28.1  83.20     6.07  80.19   \n",
      "117017  1982-04-30  1982      4         Banke  28.1  81.70    24.16  92.50   \n",
      "1047594 1984-06-30  1984      6       Mustang  28.7  83.67   115.31  63.03   \n",
      "4873    1987-07-31  1987      7  Arghakhanchi  27.9  83.20   291.64  92.40   \n",
      "\n",
      "          QV2M   RH2M  ...                 rfh                 rfh_avg  \\\n",
      "1124654   4.26  39.65  ...  #indicator+rfh+num  #indicator+rfh_avg+num   \n",
      "557097    3.39  36.95  ...  #indicator+rfh+num  #indicator+rfh_avg+num   \n",
      "117017    7.34  34.10  ...                 NaN                     NaN   \n",
      "1047594   8.90  73.82  ...  #indicator+rfh+num  #indicator+rfh_avg+num   \n",
      "4873     17.31  67.80  ...                 NaN                     NaN   \n",
      "\n",
      "                        r1h                 r1h_avg                 r3h  \\\n",
      "1124654  #indicator+r1h+num  #indicator+r1h_avg+num  #indicator+r3h+num   \n",
      "557097   #indicator+r1h+num  #indicator+r1h_avg+num  #indicator+r3h+num   \n",
      "117017                  NaN                     NaN                 NaN   \n",
      "1047594  #indicator+r1h+num  #indicator+r1h_avg+num  #indicator+r3h+num   \n",
      "4873                    NaN                     NaN                 NaN   \n",
      "\n",
      "                        r3h_avg                 rfq                 r1q  \\\n",
      "1124654  #indicator+r3h_avg+num  #indicator+rfq+pct  #indicator+r1q+pct   \n",
      "557097   #indicator+r3h_avg+num  #indicator+rfq+pct  #indicator+r1q+pct   \n",
      "117017                      NaN                 NaN                 NaN   \n",
      "1047594  #indicator+r3h_avg+num  #indicator+rfq+pct  #indicator+r1q+pct   \n",
      "4873                        NaN                 NaN                 NaN   \n",
      "\n",
      "                        r3q  version  \n",
      "1124654  #indicator+r3q+pct  #status  \n",
      "557097   #indicator+r3q+pct  #status  \n",
      "117017                  NaN      NaN  \n",
      "1047594  #indicator+r3q+pct  #status  \n",
      "4873                    NaN      NaN  \n",
      "\n",
      "[5 rows x 59 columns]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# --- Explore data ---\n",
    "print(\"Sample data:\")\n",
    "print(data.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6a66aace",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Descriptive statistics:\n",
      "                                DATE           YEAR          MONTH  \\\n",
      "count                         354089  354089.000000  354089.000000   \n",
      "mean   2000-07-09 21:02:38.146680576    1999.978231       6.507443   \n",
      "min              1981-01-31 00:00:00    1981.000000       1.000000   \n",
      "25%              1990-09-30 00:00:00    1990.000000       4.000000   \n",
      "50%              2000-07-31 00:00:00    2000.000000       7.000000   \n",
      "75%              2010-03-31 00:00:00    2010.000000      10.000000   \n",
      "max              2019-12-31 00:00:00    2019.000000      12.000000   \n",
      "std                              NaN      11.258068       3.453864   \n",
      "\n",
      "                 LAT            LON        PRECTOT             PS  \\\n",
      "count  354089.000000  354089.000000  354089.000000  354089.000000   \n",
      "mean       27.984857      84.259274      68.693673      82.696104   \n",
      "min        26.500000      80.200000       0.000000      55.460000   \n",
      "25%        27.300000      82.350000       3.740000      77.640000   \n",
      "50%        28.000000      84.200000      22.080000      83.610000   \n",
      "75%        28.600000      85.900000      99.170000      92.660000   \n",
      "max        30.000000      88.000000     641.840000      99.780000   \n",
      "std         0.880710       2.210284      98.662231      10.696815   \n",
      "\n",
      "                QV2M           RH2M            T2M  ...   TempRange_2m  \\\n",
      "count  354089.000000  354089.000000  354089.000000  ...  354089.000000   \n",
      "mean        8.429231      55.617751      15.652646  ...      10.760467   \n",
      "min         0.750000       8.510000     -17.810000  ...       1.560000   \n",
      "25%         4.060000      38.900000       9.910000  ...       8.190000   \n",
      "50%         6.700000      52.740000      16.540000  ...      10.910000   \n",
      "75%        12.820000      74.820000      22.210000  ...      13.120000   \n",
      "max        21.700000      94.820000      35.590000  ...      23.710000   \n",
      "std         5.203558      20.679756       9.051208  ...       3.278536   \n",
      "\n",
      "       EarthSkinTemp  WindSpeed_10m  MaxWindSpeed_10m  MinWindSpeed_10m  \\\n",
      "count  354089.000000  354089.000000     354089.000000     354089.000000   \n",
      "mean       15.484096       2.348334          4.635954          0.656542   \n",
      "min       -27.100000       0.850000          1.370000          0.000000   \n",
      "25%         9.050000       1.890000          3.700000          0.310000   \n",
      "50%        16.450000       2.250000          4.480000          0.530000   \n",
      "75%        22.840000       2.690000          5.430000          0.870000   \n",
      "max        39.960000       8.780000         13.580000          6.680000   \n",
      "std         9.898818       0.658802          1.334683          0.491928   \n",
      "\n",
      "       WindSpeedRange_10m  WindSpeed_50m  MaxWindSpeed_50m  MinWindSpeed_50m  \\\n",
      "count       354089.000000  354089.000000     354089.000000     354089.000000   \n",
      "mean             3.979407       2.700168          4.599389          0.905500   \n",
      "min              0.470000       0.920000          1.490000          0.000000   \n",
      "25%              3.050000       2.050000          3.580000          0.390000   \n",
      "50%              3.900000       2.490000          4.320000          0.670000   \n",
      "75%              4.810000       3.090000          5.290000          1.180000   \n",
      "max             11.560000      11.630000         15.530000          9.520000   \n",
      "std              1.288349       0.958620          1.474567          0.768221   \n",
      "\n",
      "       WindSpeedRange_50m  \n",
      "count       354089.000000  \n",
      "mean             3.693797  \n",
      "min              0.600000  \n",
      "25%              2.800000  \n",
      "50%              3.500000  \n",
      "75%              4.370000  \n",
      "max             11.970000  \n",
      "std              1.267195  \n",
      "\n",
      "[8 rows x 44 columns]\n"
     ]
    }
   ],
   "source": [
    "print(\"\\nDescriptive statistics:\")\n",
    "print(data.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "fc54a810",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['DATE', 'YEAR', 'MONTH', 'DISTRICT', 'LAT', 'LON', 'PRECTOT', 'PS',\n",
      "       'QV2M', 'RH2M', 'T2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE',\n",
      "       'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M',\n",
      "       'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'Unnamed: 0', 'District',\n",
      "       'Latitude', 'Longitude', 'Precip', 'Pressure', 'Humidity_2m', 'RH_2m',\n",
      "       'Temp_2m', 'WetBulbTemp_2m', 'MaxTemp_2m', 'MinTemp_2m', 'TempRange_2m',\n",
      "       'EarthSkinTemp', 'WindSpeed_10m', 'MaxWindSpeed_10m',\n",
      "       'MinWindSpeed_10m', 'WindSpeedRange_10m', 'WindSpeed_50m',\n",
      "       'MaxWindSpeed_50m', 'MinWindSpeed_50m', 'WindSpeedRange_50m', 'adm2_id',\n",
      "       'ADM2_PCODE', 'n_pixels', 'rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h',\n",
      "       'r3h_avg', 'rfq', 'r1q', 'r3q', 'version'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(data.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a27696a0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "82ed31ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "target_column = 'Precip'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4499abe0",
   "metadata": {},
   "outputs": [],
   "source": [
    "if target_column not in data.columns:\n",
    "    raise ValueError(f\"❌ '{target_column}' not found in dataset columns.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "71320753",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.dropna(subset=[target_column])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "fd6b8273",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'X' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[25], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[43mX\u001b[49m\u001b[38;5;241m.\u001b[39mselect_dtypes(include\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mnumber\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      2\u001b[0m y \u001b[38;5;241m=\u001b[39m data[target_column]\n",
      "\u001b[1;31mNameError\u001b[0m: name 'X' is not defined"
     ]
    }
   ],
   "source": [
    "X = X.select_dtypes(include='number')\n",
    "y = data[target_column]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "820cd6d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n🚀 Training model...\")\n",
    "model, mse = train_model(X, y)\n",
    "print(f\"✅ Model training complete. MSE: {mse:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34b7e3c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Train and evaluate model\n",
    "model, mse, y_test, predictions = train_model(X, y)\n",
    "\n",
    "# Plot predicted vs actual\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.scatter(y_test, predictions, alpha=0.5)\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n",
    "plt.xlabel('Actual Values')\n",
    "plt.ylabel('Predicted Values')\n",
    "plt.title('📊 Predicted vs Actual Values')\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
