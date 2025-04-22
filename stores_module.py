import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, root_mean_squared_log_error as rmsle



def store_data(store_nb):
    """ Slice the data to get the data for one store and get rid off the useless columns.
    """
    with open("AngeliqueFile.pkl", "rb") as f:
        data = pickle.load(f)
    data.drop(["sales","id","dcoilwtico"], axis=1, inplace=True)
    data["National Workday"] = data["National Workday"].cat.add_categories(["Recupero"])
    data.loc[~(data["National Workday"] == "nope"),"National Workday"] = "Recupero"
    data["National Workday"] = data["National Workday"].cat.remove_unused_categories()
    data["Local Holiday"] = data["Local Holiday"].cat.reorder_categories(new_categories =\
                                                                         ["nope","Local Holiday"],\
                                                                            ordered=True)
    data["weekday"] = data["date"].dt.weekday
    store = data[data["store_nbr"] == store_nb].copy()
    store.drop(["store_nbr","city","state","type","cluster"], axis=1, inplace=True)
    return store

class binom_cat_bool:
    """ Class to store the binomial categories.
    """
    def __init__(self, df):
        self.df = df
        self.list_binom_cols = []
        self.list_binom_bool_dict = []
        self.binom_to_bool = self.binom_to_bool
    def binom_to_bool(self,col_name, cat_1):
        """ Convert a column of binomial categories to bool.
        """
        new = np.where(self.df[col_name] == cat_1,0,np.where(pd.isna(self.df[col_name]),np.nan,1))
        return new
    def binom_bool(self):
        """ Save the list of binom. columns and the corresponding dict as a list.
        """
        for col in self.df.select_dtypes("category").columns:
            n_cat = len(self.df[col].cat.categories)
            if n_cat == 2:
                cats = self.df[col].cat.categories
                dict_col = {cats[0]:0, cats[1]:1}
                self.list_binom_bool_dict.append(dict_col)
                self.list_binom_cols.append(col)
        return self.list_binom_cols, self.list_binom_bool_dict
    def transform(self):
        """ Transform the columns of binomial categories to bool.
        """
        for col in self.list_binom_cols:
            cat_1 = self.df[col].cat.categories[0]
            self.df[col] = self.binom_to_bool(col, cat_1)
            self.df[col] = self.df[col].astype("int32")
        return self.df

def rename_null_cat(df,col_name,cat):
    df[col_name] = df[col_name].cat.add_categories(["A"])
    df.loc[df[col_name] == cat,col_name] = "A"
    df[col_name] = df[col_name].cat.remove_categories([cat])
    return df[col_name]

class my_labelEncoder:
    def __init__(self):
        self.dict_cat = {}
    
    def fit(self,df,col):
        categories = list(df[col].cat.categories)
        for cat in df[col].cat.categories:
            self.dict_cat[cat] = categories.index(cat)
    
    def transform(self,df,col):
        nb_list = df[col].map(self.dict_cat).astype("Int64") 
        #"Int64" (with capital "I") is pandas’ nullable integer type, which allows NaN.
        return nb_list

def frame_time_of_interest(df,month):
    """ Select holidays column based on the month of interest (the one to predict).
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df_month = df[df["month"] == month].copy()
    df_month.drop(["month"], axis=1, inplace=True)
    df.drop("month", axis=1, inplace=True)
    df.set_index("date", drop=True, inplace=True)
    holidays_list = ["National Event","National holiday","National period of holiday",\
                 "National Workday","Local Holiday", "Regional Holiday", "Workday","Transfer"]
    for col in holidays_list:
        if df_month[col].sum() == 0:
            df_month.drop(col, axis=1, inplace=True)
            df.drop(col, axis=1, inplace=True)
    return df_month, df
    

def reframe_with_lags(df, lags, dropnan=True):
    """ Reframe the data to add lags.
    """
    df = df.copy()
    df.set_index("date", inplace=True, drop=True)
    cols, names = list(), list()
    n_vars = 1 if type(df) is list else df.shape[1] #df.shape[1] is the number of columns
    if type(df) is list:
        df = pd.DataFrame(df)
	# input sequence (t-n, ... t-1)
    for i in range(lags, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    #placing y at the end of the dataframe
    output_index = list(agg.columns).index("var1(t)")
    output = agg.iloc[:,output_index]
    agg.drop("var1(t)", axis=1, inplace=True)
    agg = pd.concat([agg,output], axis=1)
    return agg


def split(df, date):
    """ Split the dataset to train set and test set without randomness.
    """
    train_lim = len(df.loc[:date])
    train = df.iloc[:train_lim]
    test = df.iloc[train_lim:]
    X_train = np.delete(train, -1, axis=1)
    y_train = train.iloc[:,-1]
    X_test = np.delete(test, -1, axis=1)
    y_test = test.iloc[:,-1]
    return X_train, y_train, X_test, y_test, train, test

def predictions(X_train,y_train,X_test):
    LinReg = LinearRegression()
    LinReg.fit(X_train, y_train)
    y_pred = LinReg.predict(X_test)
    predictions = X_test.copy()
    predictions["predictions"] = y_pred.shape[0]
    return predictions


    

    