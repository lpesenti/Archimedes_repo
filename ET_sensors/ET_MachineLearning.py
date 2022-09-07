__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Daniele Dell'Aquila"]
__version__ = "0.1.0"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

r"""
This script contains two methods (August 29, 2022) that are used to develop a ML model to predict the RMS value of a
seismometer.
For further information see 'A Machine Learning approach for seismic analysis' at 
https://docs.google.com/presentation/d/1dGBIvaTinZ9yRlQbHvqnxKXBgPIFWWQ1/edit?usp=sharing&ouid=102833711495125222214&rtpof=true&sd=true
"""

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
    r"""
    This function is used to merge all the seismometer DataFrame given in the ML_config.ini file with the data from the
    weather station. In addition, it plots the correlation matrix of these quantities.

    Note
    -----
    To use the ML_analysis() method it is necessary to have the DataFrame created with this method.
    """
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
    r"""
    In this function is performed the ML analysis on a target variable specified in the ML_config.ini file.
    This function automatically organize the DataFrame parsed to have the following format:

    | Feature 1 | Feature 2 | ... | Feature N | Target variable |
    |-----------+-----------+-----+-----------+-----------------|
    | ......... | ......... | ... | ......... | ............... |
    | ......... | ......... | ... | ......... | ............... |
    | ......... | ......... | ... | ......... | ............... |

    Note
    ------
    The DataFrame format used for the analysis is the same produced by the correlation() method above
    """
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

    fig0 = plt.figure(figsize=(19.2, 10.8))
    fig1 = plt.figure(figsize=(19.2, 10.8))
    fig2 = plt.figure(figsize=(19.2, 10.8))
    ax0 = fig0.add_subplot()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()

    ax0.set_title(r'$\frac{\mathrm{Train} - \mathrm{Pred Train}}{\mathrm{Train} + \mathrm{Pred Train}}$', fontsize=30)
    ax1.set_title(r'$\frac{\mathrm{Test} - \mathrm{Pred Test}}{\mathrm{Test} + \mathrm{Pred Test}}$', fontsize=30)
    ax2.set_title(r'$\frac{\mathrm{All} - \mathrm{Pred All}}{\mathrm{All} + \mathrm{Pred All}}$', fontsize=30)

    ax0.scatter(np.arange(len(ytrain)), (ytrain - ypredTrain) / (ytrain + ypredTrain) * 100)
    ax1.scatter(np.arange(len(ytest)), (ytest - ypredTest) / (ytest + ypredTest) * 100)
    ax2.scatter(np.arange(len(Y)), (Y - ypredAll) / (Y + ypredAll) * 100)

    ax0.set_xlabel("Instance number", fontsize=24)
    ax1.set_xlabel("Instance number", fontsize=24)
    ax2.set_xlabel("Instance number", fontsize=24)

    ax0.set_ylabel(r"Percentage %", fontsize=24)
    ax1.set_ylabel(r"Percentage %", fontsize=24)
    ax2.set_ylabel(r"Percentage %", fontsize=24)

    ax0.tick_params(axis='both', which='both', labelsize=22)
    ax1.tick_params(axis='both', which='both', labelsize=22)
    ax2.tick_params(axis='both', which='both', labelsize=22)

    ax0.set_ylim([-100, 100])
    ax1.set_ylim([-100, 100])
    ax2.set_ylim([-100, 100])

    ax0.axhline(y=10, color='tab:red', linestyle='--')
    ax0.axhline(y=-10, color='tab:red', linestyle='--')
    ax1.axhline(y=10, color='tab:red', linestyle='--')
    ax1.axhline(y=-10, color='tab:red', linestyle='--')
    ax2.axhline(y=10, color='tab:red', linestyle='--')
    ax2.axhline(y=-10, color='tab:red', linestyle='--')

    ax0.grid(True, linestyle='--', which='both', axis='both')
    ax1.grid(True, linestyle='--', which='both', axis='both')
    ax2.grid(True, linestyle='--', which='both', axis='both')

    fig0.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == '__main__':
    correlation() if corrMatrix else ''
    ML_analysis() if ML else ''
