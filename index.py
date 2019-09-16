
from __future__ import print_function
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from flask import Flask, redirect, url_for, request, render_template
from gevent.pywsgi import WSGIServer
import json

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import  style
from matplotlib.pyplot import figure
import matplotlib as mpl
import mpld3

# In python 2.7
import sys


app = Flask(__name__)

def model_predict_quad(f):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2019, 1, 10)

    df = web.DataReader(f,'yahoo', start, end)
    df.tail()

    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()

    mpl.rc('figure',figsize=(8,7))
    mpl.__version__

    style.use('ggplot')
    close_px.plot(label='f')
    mavg.plot(label='mavg')
    plt.legend()


    rets = close_px / close_px.shift(1) - 1

    rets.plot(label='return')

    dfcomp = web.DataReader(['AAPL', 'GE' , 'GOOG' , 'IBM', 'MSFT'], 'yahoo',
         start=start,end=end) ['Adj Close']

    retscomp = dfcomp.pct_change()

    corr = retscomp.corr()

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # # Linear regression
    # clfreg = LinearRegression(n_jobs=-1)
    # clfreg.fit(X_train, y_train)
    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # # Quadratic Regression 3
    # clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    # clfpoly3.fit(X_train, y_train)
    #
    #
    # # KNN Regression
    # clfknn = KNeighborsRegressor(n_neighbors=2)
    # clfknn.fit(X_train, y_train)

    # confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    # confidencepoly3 = clfpoly3.score(X_test,y_test)
    # confidenceknn = clfknn.score(X_test, y_test)

    forecast_set = clfpoly2.predict(X_lately)
    dfreg['Forecast'] = np.nan

    return str(forecast_set)


def model_predict_knn(f):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2019, 1, 10)

    df = web.DataReader(f,'yahoo', start, end)
    df.tail()

    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()

    mpl.rc('figure',figsize=(8,7))
    mpl.__version__

    style.use('ggplot')
    close_px.plot(label='f')
    mavg.plot(label='mavg')
    plt.legend()


    rets = close_px / close_px.shift(1) - 1

    rets.plot(label='return')

    dfcomp = web.DataReader(['AAPL', 'GE' , 'GOOG' , 'IBM', 'MSFT'], 'yahoo',
         start=start,end=end) ['Adj Close']

    retscomp = dfcomp.pct_change()

    corr = retscomp.corr()

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # # Linear regression
    # clfreg = LinearRegression(n_jobs=-1)
    # clfreg.fit(X_train, y_train)
    # # Quadratic Regression 2
    # clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    # clfpoly2.fit(X_train, y_train)
    #
    # # Quadratic Regression 3
    # clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    # clfpoly3.fit(X_train, y_train)


    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    # confidencereg = clfreg.score(X_test, y_test)
    # confidencepoly2 = clfpoly2.score(X_test,y_test)
    # confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)

    forecast_set = clfknn.predict(X_lately)
    dfreg['Forecast'] = np.nan

    return str(forecast_set)


def model_predict(f):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2019, 1, 10)
    print(f, file=sys.stderr)

    df = web.DataReader(f,'yahoo', start, end)
    df.tail()

    close_px = df['Adj Close']
    print(close_px, file=sys.stderr)
    mavg = close_px.rolling(window=100).mean()

    mpl.rc('figure',figsize=(8,7))
    mpl.__version__

    style.use('ggplot')
    close_px.plot(label='f')
    mavg.plot(label='mavg')
    plt.legend()


    rets = close_px / close_px.shift(1) - 1

    rets.plot(label='return')

    dfcomp = web.DataReader(['AAPL', 'GE' , 'GOOG' , 'IBM', 'MSFT'], 'yahoo',
         start=start,end=end) ['Adj Close']

    retscomp = dfcomp.pct_change()

    corr = retscomp.corr()

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)
    # Quadratic Regression 2
    # clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    # clfpoly2.fit(X_train, y_train)
    #
    # # Quadratic Regression 3
    # clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    # clfpoly3.fit(X_train, y_train)


    # KNN Regression
    # clfknn = KNeighborsRegressor(n_neighbors=2)
    # clfknn.fit(X_train, y_train)

    confidencereg = clfreg.score(X_test, y_test)
    # confidencepoly2 = clfpoly2.score(X_test,y_test)
    # confidencepoly3 = clfpoly3.score(X_test,y_test)
    # confidenceknn = clfknn.score(X_test, y_test)

    forecast_set = clfreg.predict(X_lately)
    dfreg['Forecast'] = np.nan

    return str(forecast_set)



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method =='POST':
        f = request.get_data().decode('utf8')
        print(f, file=sys.stderr)
        preds = model_predict(f)
        return preds

@app.route('/predictquad', methods=['GET','POST'])
def predictquad():
    if request.method =='POST':
        f = request.get_data().decode('utf8')
        preds = model_predict_quad(f)
        return preds

@app.route('/predictknn', methods=['GET','POST'])
def predictknn():
    if request.method =='POST':
        f = request.get_data().decode('utf8')
        preds = model_predict_knn(f)
        return preds


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    app.run(debug = True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
