import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class SplitLagg():
    def __init__(self,df):
        self.df = df
    
    def transactions_X(self):
        self.y=self.df[["transactions"]].copy()
        self.X=self.df.drop("transactions",axis=1)
    
    def lagg_X(self, lags, dropnan=True):
        cols, names = list(), list()
        n_vars = 1 if type(self.X) is list else self.X.shape[1] #df.shape[1] is the number of columns
        if type(self.X) is list:
            self.X = pd.DataFrame(self.X)
	    # input sequence (t-n, ... t-1)
        for i in range(lags, 0, -1):
            cols.append(self.X.shift(i))
            names += [('var%d X(t-%d)' % (j+1, i)) for j in range(n_vars)]
	    # forecast sequence (t, t+1, ... t+n)
        for i in range(0, 1):
            cols.append(self.X.shift(-i))
            if i == 0:
                names += [('var%d X(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d X(t+%d)' % (j+1, i)) for j in range(n_vars)]
	    # put it all together
        self.agg_X = pd.concat(cols, axis=1)
        self.agg_X.columns = names
	    # drop rows with NaN values
        if dropnan:
            self.agg_X.dropna(inplace=True)
        return self.agg_X
    
    def lagg_y(self, lags, dropnan=True):
        cols, names = list(), list()
        n_vars = 1 if type(self.y) is list else self.y.shape[1] #df.shape[1] is the number of columns
        if type(self.y) is list:
            self.y = pd.DataFrame(self.y)
	# input sequence (t-n, ... t-1)
        for i in range(lags, 0, -1):
            cols.append(self.y.shift(i))
            names += [('var%d y(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
        for i in range(0, 1):
            cols.append(self.y.shift(-i))
            if i == 0:
                names += [('var%d y(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d y(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
        self.agg_y = pd.concat(cols, axis=1)
        self.agg_y.columns = names
	# drop rows with NaN values
        if dropnan:
            self.agg_y.dropna(inplace=True)
        return self.agg_y

def split_train_test(X,y,date):
    date=pd.to_datetime(date)
    X_train = X.loc[:date].copy()
    X_test = X.loc[date+pd.Timedelta(days=1):date+pd.Timedelta(days=15)].copy()
    train_y = y.loc[:date].copy()
    test_y = y.loc[date+pd.Timedelta(days=1):date+pd.Timedelta(days=15)].copy()
    y_train = train_y.loc[:,"var1 y(t)"].copy()
    train_y.drop("var1 y(t)", axis=1, inplace=True)
    X_train = pd.concat([X_train,train_y], axis=1)
    y_test = test_y.iloc[:,-1].copy()
    test_y.drop("var1 y(t)", axis=1, inplace=True)
    X_test = pd.concat([X_test,test_y], axis=1)
    return X_train,y_train,X_test,y_test,test_y

def recurs_Lin_regr(X_train,y_train,X_test,test_y):
    LinReg = LinearRegression()
    LinReg.fit(X_train,y_train)
    predictions = []
    test_pred = test_y.copy()
    index = len(X_train.columns)-4
    for i in range(len(X_test)):
        pred_i = LinReg.predict(X_test.iloc[i].to_frame().T)
        if i<= len(X_test)-5:
            test_pred.iat[i+1,3] = pred_i[0]
            test_pred.iat[i+2,2] = pred_i[0]
            test_pred.iat[i+3,1] = pred_i[0]
            test_pred.iat[i+4,0] = pred_i[0]
            X_test.iloc[i+1:i+5,index:] = test_pred.iloc[i+1:i+5,:].values
        elif i==len(X_test)-4:
            test_pred.iat[i+1, 3] = pred_i[0]
            test_pred.iat[i+2, 2] = pred_i[0]
            test_pred.iat[i+3,1] = pred_i[0]
            X_test.iloc[i+1:i+4,index:] = test_pred.iloc[i+1:i+4,:].values
        elif i==len(X_test)-3:
            test_pred.iat[i+1, 3] = pred_i[0]
            test_pred.iat[i+2, 2] = pred_i[0]
            X_test.iloc[i+1:i+3,index:] = test_pred.iloc[i+1:i+3,:].values
        elif i == len(X_test)-2:
            test_pred.iat[i+1, 3] = pred_i[0]
            X_test.iloc[i+1,index:] = test_pred.iloc[i+1,:].values
        predictions.append(pred_i[0])
    return predictions,test_pred

def transactions_pred(y_test,predictions):
    df = y_test.to_frame(name="transactions")
    df["predictions"] = predictions
    df["predictions"] = df["predictions"].astype("int")
    return df
