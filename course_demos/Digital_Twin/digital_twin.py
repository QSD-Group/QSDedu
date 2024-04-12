#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDedu: Education Modules for Quantitative Sustainable Design

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

'''

import os
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor

# Set up directory
folder = os.path.dirname(__file__)
path = os.path.join(folder, 'dyninfluent_bsm2.csv')

# %%

# Import data
columns = [
    't', # time, [d]
    'S_I',      # inert soluble material, g COD/m3
    'S_S',      # readily biodegradable substrate, g COD/m3
    'X_I',      # inert particulate material, g COD/m3
    'X_S',      # slowly biodegradable substrate, g COD/m3
    'X_BH',     # heterotrophic biomass, g COD/m3
    'X_BA',     # autotrophic biomass, g COD/m3
    'X_P',      # inert particulate material from biomass decay, g COD/m3
    'S_O',      # dissolved oxygen, g COD/m3
    'S_NO',     # nitrate and nitrite, g N/m3
    'S_NH',     # ammonia and ammonium, g N/m3
    'S_ND',     # soluble organic nitrogen associated with SS, g N/m3
    'X_ND',     # particulate organic nitrogen associated with XS, g N/m3
    'S_ALK',    # alkalinity
    'TSS',      # total suspended solids, g SS/m3
    'Q',        # flow rate, m3/d
    'T',        # temperature, °C
    'S_D1',     # dummy variable (soluble) no 1
    'S_D2',     # dummy variable (soluble) no 2
    'S_D3',     # dummy variable (soluble) no 3
    'X_D4',     # dummy variable (particulate) no 1
    'X_D5',     # dummy variable (particulate) no 2
    ]
df = pd.read_csv(path, header=None, names=columns)

# Throw out the last five columns (all dummies)
data = df.iloc[:, :-5].copy()

# Original data was taken at a 15-min interval, resample to a 4-hour for faster simulation
# data = data[:].iloc[::16, :]


# %%




# %%

# Plot and calculate error
def evaluate(y_train, y_test, y_train_pred, y_test_pred, figsize=figsize):
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    axs = y_train.plot(color='0.25', subplots=True, sharex=True, figsize=figsize)
    axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
    axs = y_train_pred.plot(color='C0', subplots=True, sharex=True, ax=axs)
    axs = y_test_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
    for ax in axs: ax.legend([])
    return rmse_train, rmse_test, axs

# Trend prediction only
def trend_prediction(
        features=['T', 'Q'],
        target='S_I',
        test_size=0.25,
        figsize=(50, 2),
        ):
    # Create trend features
    y = data[[target]].copy()
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        constant=True,  # the intercept
        order=2,        # quadratic trend
        drop=True,      # drop terms to avoid collinearity
    )
    X = dp.in_sample()  # features for the training data
    
    X = data[features].copy()

    # Split training and test data
    idx_train, idx_test = train_test_split(
        y.index, test_size=test_size, shuffle=False,
    )
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    # Fit trend model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_fit = pd.DataFrame(
        model.predict(X_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    y_pred = pd.DataFrame(
        model.predict(X_test),
        index=y_test.index,
        columns=y_test.columns,
    )
    
    # Create residuals (the collection of detrended series) from the training set
    y_resid = y_train - y_fit
    # Train XGBoost on the residuals
    xgb = XGBRegressor()
    xgb.fit(X_train, y_resid)
    
    # Add the predicted residuals onto the predicted trends
    y_fit_resid = xgb.predict(X_train)
    y_fit_resid = y_fit_resid.reshape(y_fit.shape)
    y_fit_boosted = y_fit_resid + y_fit

    y_pred_resid = xgb.predict(X_test)
    y_pred_resid = y_pred_resid.reshape(y_pred.shape)
    y_pred_boosted = y_pred_resid + y_pred


# %%



# %%

def hybrid_prediction(
        features=['T', 'Q'],
        target='S_I',
        test_size=0.25,

        ):
    # Create trend features
    y = data[[target]].copy()
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        constant=True,  # the intercept
        order=2,        # quadratic trend
        drop=True,      # drop terms to avoid collinearity
    )
    X = dp.in_sample()  # features for the training data
    
    X = data[features].copy()

    # Split training and test data
    idx_train, idx_test = train_test_split(
        y.index, test_size=test_size, shuffle=False,
    )
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    # Fit trend model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_fit = pd.DataFrame(
        model.predict(X_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    y_pred = pd.DataFrame(
        model.predict(X_test),
        index=y_test.index,
        columns=y_test.columns,
    )
    
    # Create residuals (the collection of detrended series) from the training set
    y_resid = y_train - y_fit
    # Train XGBoost on the residuals
    xgb = XGBRegressor()
    xgb.fit(X_train, y_resid)
    
    # Add the predicted residuals onto the predicted trends
    y_fit_resid = xgb.predict(X_train)
    y_fit_resid = y_fit_resid.reshape(y_fit.shape)
    y_fit_boosted = y_fit_resid + y_fit

    y_pred_resid = xgb.predict(X_test)
    y_pred_resid = y_pred_resid.reshape(y_pred.shape)
    y_pred_boosted = y_pred_resid + y_pred

    return evaluate(y_train, y_test, y_train_pred, y_test_pred, figsize=figsize)


# %%

outputs = hybrid_prediction(
    features=['S_I'],
    target='S_S',
    test_size=0.25,
    )


# for column in columns[:-5]:
#     hybrid_prediction(
#         target=column, test_size=0.25)