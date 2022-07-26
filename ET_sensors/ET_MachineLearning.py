__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Daniele Dell'Aquila"]
__version__ = "0.0.5"
__maintainer__ = "Luca Pesenti"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

import configparser
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

config = configparser.ConfigParser()
config.read('ML_config.ini')

df_list = [config['DataFrames'][df] for df in config['DataFrames']]
rms_merged = pd.read_parquet(df_list[0])
for dataframe in df_list:
    rms_merged = pd.merge(rms_merged, pd.read_parquet(dataframe), how='inner')
rms_merged.rename(columns={'time': 'dateTime'}, inplace=True)
rms_merged['dateTime'] = (rms_merged['dateTime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

weather_path = config['Paths']['weather_directory']
weather_list = glob.glob(weather_path + r'*.csv')
weather_list.sort()
weather_df = pd.DataFrame()
for csv in weather_list:
    weather_df = pd.concat([weather_df, pd.read_csv(csv)], ignore_index=True)

total_df = pd.merge(rms_merged, weather_df, on='dateTime')
cols = [x for x in total_df.columns if x not in weather_df.columns]
# total_df[cols] = total_df[cols].replace({'0': np.nan, 0: np.nan})  # remove data when RMS=0
for col in cols:
    total_df.loc[total_df[col] <= -2.5e+11, cols] = np.nan  # remove data when RMS is too low
    total_df.loc[total_df[col] >= -1.6e+11, cols] = np.nan  # remove data when RMS is too high

fig = plt.figure(figsize=(19.2, 10.8))
fig1 = plt.figure(figsize=(19.2, 10.8))
# gs = fig.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[3, 1.5])
# ax = gs.subplots(sharex=False)
ax = fig.add_subplot()
ax1 = fig1.add_subplot()

sns.heatmap(total_df.corr(), cmap='viridis', cbar=True, ax=ax1, annot=True)
sns.lineplot(data=total_df.iloc[:, 1:len(df_list) + 1], ax=ax, dashes=False)

fig.tight_layout()
fig1.tight_layout()

plt.show()
