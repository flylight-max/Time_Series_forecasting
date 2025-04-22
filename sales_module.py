import pandas as pd
import numpy as np
import pickle
from stores_module import *

def transactions_cat():
    with open("AngeliqueFile.pkl","rb") as f:
        data = pickle.load(f)
    data["National Workday"] = data["National Workday"].cat.add_categories(["Recupero"])
    data.loc[~(data["National Workday"] == "nope"),"National Workday"] = "Recupero"
    data["National Workday"] = data["National Workday"].cat.remove_unused_categories()
    data["Local Holiday"] = data["Local Holiday"].cat.reorder_categories(new_categories =\
                                                                         ["nope","Local Holiday"],\
                                                                            ordered=True)
    data["weekday"] = data["date"].dt.weekday
    binom = binom_cat_bool(data)
    binom.binom_bool()
    data = binom.transform()
    no_cat_list = ["not a national event","Not a Nat holiday","Nope"]
    for col in data.columns:
        for no_cat in no_cat_list:
            if data[col].dtype == 'category' and no_cat in data[col].cat.categories:
                rename_null_cat(data,col,no_cat)
    for col in data.columns:
        if data[col].dtype == 'category':
            col_encod = my_labelEncoder()
            col_encod.fit(data,col)
            data[col] = col_encod.transform(data,col)
    return data


def family_df(family_name,df,month):
    """
    Returns a DataFrame for a specific family of items with holiday and events columns needed for the month of interest.
    Arg: 
        family_name (str): The name of the family to filter by.
        df (DataFrame): Must contain date in datetime format.
        month (int): month of interest.
    """
    family_df = df[df["family"] == family_name].copy()
    family_df["month"] = family_df["date"].dt.month
    #family_df["weekday"] = family_df.dt.weekday
    family_df.drop(["family","id"], axis=1, inplace=True)
    family_df_month = family_df[family_df["month"] == month].copy()
    family_df.drop("month", axis=1, inplace=True)
    holidays_list = ["National Event","National holiday","National period of holiday",\
                 "National Workday","Local Holiday", "Regional Holiday", "Workday","Transfer"]
    for col in holidays_list:
        if family_df_month[col].sum() == 0:
            family_df.drop(col,axis=1, inplace=True)
    family_df["weekday"] = family_df["date"].dt.weekday
    family_df.set_index("date",drop=True, inplace=True)
    return family_df

class SplitLagg_sale():
    def __init__(self,df,output_var):
        self.df = df
        self.output_var = output_var
    
    def sales_X(self):
        self.y=self.df[[self.output_var]].copy()
        self.X=self.df.drop(self.output_var,axis=1)
    
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

    
    