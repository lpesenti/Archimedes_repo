__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Daniele Dell'Aquila"]
__version__ = "0.0.7"
__maintainer__ = "Luca Pesenti"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

import configparser
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read('ML_config.ini')

# Booleans
corrMatrix = config.getboolean('Bool', 'corrMatrix')
ML = config.getboolean('Bool', 'ML_analysis')


def correlation():
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

    save_df = config.getboolean('Bool', 'save_total_df')

    cols = [x for x in total_df.columns if x not in weather_df.columns]
    for col in cols:
        total_df.loc[total_df[col] <= 1e9, cols] = np.nan  # remove data when RMS is too low
        total_df.loc[total_df[col] >= 2e+11, cols] = np.nan  # remove data when RMS is too high

    total_df = total_df.dropna()
    total_df.to_parquet(r'D:\ET\2022\Total_df.brotli', compression='brotli', compression_level=9) if save_df else ''

    fig = plt.figure(figsize=(19.2, 10.8))
    fig1 = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()
    ax1 = fig1.add_subplot()

    sns.heatmap(total_df.corr(), cmap='viridis', cbar=True, ax=ax1, annot=True)
    sns.lineplot(data=total_df.iloc[:, 1:len(df_list) + 1], ax=ax, dashes=False, estimator=None)
    ax.set_xlabel("Timestamp (s)", fontsize=24)
    ax.set_ylabel(r"$\int_1^{10} ASD [ms^{-1}/Hz] d\nu$", fontsize=24)
    ax.tick_params(axis='both', which='both', labelsize=22)
    ax.grid(True, linestyle='--', which='both', axis='both')
    # ax.legend(loc='best', shadow=True, fontsize=24)
    fig.tight_layout()
    fig1.tight_layout()

    plt.show()


def ML_analysis():
    config = configparser.ConfigParser()
    config.read('ML_config.ini')
    data_path = config['Paths']['full_df']
    target = config['ML_variables']['target_variable']

    data = pd.read_parquet(data_path)
    cols = data.columns.tolist()
    cols.sort(key=target.__eq__)
    data = data[cols]
    print(data.info())
    np.savetxt(r'D:\ET\2022\Total_df.dat', data.values)
    data = np.loadtxt(r'D:\ET\2022\Total_df.dat')

    num_patterns, dim_patterns = data.shape
    num_features = dim_patterns - 1

    X = data[:, 0:num_features]
    Y = data[:, num_features]
    test_size = 0.33
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

    # To search for the best-estimator uncomment to the line "xgbr.save_model("xgbModelX_3_0.json")"
    # Parameters range to find best_estimator_
    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['reg:squarederror'],
                  'learning_rate': [0.4],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                  # 'min_child_weight': [0.3, 0.4, 1],
                  # 'silent': [1],
                  # 'gamma': [0, 1, 2, 3, 4, 5],
                  # 'subsample': [0.2, 0.3, 0.4, 0.5],
                  # 'colsample_bytree': [0.7, 0.8, 0.9, 1],
                  'n_estimators': [8, 9, 10],  # number of trees
                  'missing': [-999],
                  'seed': [1337]
                  }

    xgbtrial = xgb.XGBRegressor()

    xgb_grid = GridSearchCV(xgbtrial,
                            parameters,
                            cv=4,
                            n_jobs=4,
                            verbose=3)

    xgb_grid.fit(xtrain, ytrain)

    xgb_model = xgb_grid.best_estimator_

    # xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    # colsample_bynode=1, colsample_bytree=1, gamma=0,
    # importance_type='gain', learning_rate=0.3, max_delta_step=0,
    # max_depth=3, min_child_weight=1, missing=1, n_estimators=20,
    # n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
    # reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
    # silent=None, subsample=1, verbosity=1)

    print(xgb_model)

    # xgb_model.fit(xtrain, ytrain, sample_weight = wtrain)

    ypredTrain = xgb_model.predict(xtrain)
    ypredTest = xgb_model.predict(xtest)
    ypredAll = xgb_model.predict(X)

    scoreTrain = r2_score(ytrain, ypredTrain)
    scoreTest = r2_score(ytest, ypredTest)
    scoreAll = r2_score(Y, ypredAll)

    print("Score learning", scoreTrain)
    print("Score testing", scoreTest)
    print("Score all", scoreAll)

    xgb_model.save_model(config['Paths']['out_model'])


if __name__ == '__main__':
    correlation() if corrMatrix else ''
    ML_analysis() if ML else ''
