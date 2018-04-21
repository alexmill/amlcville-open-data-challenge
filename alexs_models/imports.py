import os
import numpy as np
import pandas as pd
import datetime
from copy import deepcopy
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def mean_absolute_percentage_error(y_true, y_pred, sample_weights=None, percentage=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)
            
    if percentage==False:
        if type(sample_weights) == type(None):
            return(np.mean(np.abs((y_true - y_pred))))
        else:
            sample_weights = np.array(sample_weights)
            assert len(sample_weights) == len(y_true)
            return( (1/sum(sample_weights)*np.dot(sample_weights, (np.abs((y_true - y_pred))))) )
        
    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return( (100/sum(sample_weights)*np.dot(sample_weights, (np.abs((y_true - y_pred) / y_true)))) )
    
    
    
    

class ModelData:
    """
    Takes dataframe (or series for Y) and column names/interactions 
    and returns a model matrix with intercept (or a properly formatted
    series for Y)
    """
    def __init__(self, train_x, Y=None, features=None, interactions=None, scaler=None, sample_weights=None):
        self.train_x = train_x
        self.features = features
        self.interactions = interactions
        self.sample_weights = sample_weights
        
        if type(Y)!=type(None):
            assert all(train_x.index == Y.index)
            if type(Y) == pd.core.frame.DataFrame:
                self.Y = Y.iloc[:,0]
            elif type(Y) == pd.core.series.Series:
                self.Y = Y
            else:
                raise Exception("WRONG TYPE: train_y")
        else:
            self.Y = None
        
        self.num_coefs = len(features) + len(interactions) + 1
        
        X = deepcopy(self.train_x[features])
        for interaction in interactions:
            X["{}_X_{}".format(*interaction)] = deepcopy(self.train_x[interaction[0]].multiply(self.train_x[interaction[1]]))
        for col in X.columns:
            X[col] = X[col].astype(float)
        
        X = X[sorted(X.columns)]
        self.X_raw = deepcopy(X)
        
        # Standardize the data by subtracting mean, dividing by std. dev. for each feature
        if type(scaler)==type(None):
            self.scaler = StandardScaler().fit(X)
        else:
            self.scaler = scaler
        X_std = self.scaler.transform(X)
        self.X = pd.DataFrame(X_std, columns=self.X_raw.columns, index=self.X_raw.index)
        self.X["intercept"] = 1
        self.X = self.X[sorted(self.X.columns)]
        
        

        
class LinearModel_MAPE:
    """
    Linear model: Y = XB, fit by minimizing the mean absolute percentage error
    with either L1 or L2 regularization
    """
    def __init__(self, regularization=0.00012, loss="L2", X=None, Y=None, sample_weights=None, log=False, beta_init=None):
        assert loss in ["L1", "L2"]
        self.regularization = regularization
        self.beta = None
        self.loss = loss
        self.sample_weights = sample_weights
        self.log = log
        self.beta_init = beta_init
        
        self.X = X
        self.Y = Y

    
    def predict(self, X):
        beta_series = pd.Series(self.beta, index=X.columns).astype(float)
        prediction = X.dot(beta_series)
        if self.log:
            prediction = np.exp(prediction)
        return(prediction)
    
    def true_error(self):
        predictions = self.predict(self.X)
        assert all(predictions.index == self.Y.index)
        return(mean_absolute_percentage_error(self.Y, predictions, sample_weights=self.sample_weights, percentage=True))
    
    def training_error(self):
        if self.log==False:
            return(self.error())
        else:
            # training on log transform
            predictions = self.predict(self.X)
            return(mean_absolute_percentage_error(np.log(self.Y), np.log(predictions), sample_weights=self.sample_weights, percentage=False))
    
    def error(self):
        if self.log==False:
            return(self.true_error())
        else:
            return(self.training_error())
    
    def l2_regularization_loss(self, beta):
        self.beta = beta
        return(self.error() + sum(self.regularization*np.array(self.beta)**2))
    
    def l1_regularization_loss(self, beta):
        self.beta = beta
        return(self.error() + sum(self.regularization*np.abs(np.array(self.beta))))
    
    def fit(self, X=None, Y=None, sample_weights=None, maxiter=250):
        
        # If inputs to fit are left undefined, then fit the
        # model using the originally given inputs
        if type(None) in [type(X), type(Y)]:
            assert type(None) not in [type(self.X), type(self.Y)]
        else:
            self.X = X
            self.Y = Y
            
        if type(sample_weights) != type(None):
            self.sample_weights = sample_weights
        
        if type(self.beta_init)==type(None):
            self.beta_init = np.array([1]*self.X.shape[1])
            idx = np.where(self.X.columns=="intercept")
            if self.log==True:
                y_mean = float(np.log(self.Y).mean())
            else:
                y_mean = float(self.Y.mean())

            self.beta_init[idx] = y_mean
        else: 
            # Use provided initial values
            pass
        
        if self.loss=="L2":
            self.loss_func = self.l2_regularization_loss
        elif self.loss=="L1":
            self.loss_func = self.l1_regularization_loss
            
        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
            
        res = minimize(self.loss_func, self.beta_init, method='BFGS', options={'maxiter': maxiter})
        self.beta = res.x
        self.beta_init = self.beta
        
        
class RF_RegressionModel:
    """
    Random Regression Forest
    regularization = num_features 
    """
    def __init__(self, 
                 n_estimators=500,
                 regularization='auto',
                 X=None, 
                 Y=None, 
                 sample_weights=None
        ):
        self.regularization = regularization
        self.sample_weights = sample_weights
        self.random_forest = RandomForestRegressor(
            n_estimators=n_estimators, criterion="mae",
            max_features=self.regularization
        )
        
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        prediction = pd.Series(self.random_forest.predict(X), index=X.index)
        return(prediction)
    
    def error(self):
        predictions = self.predict(self.X)
        assert all(self.predict(self.X).index == self.Y.index)
        return(mean_absolute_percentage_error(
            self.Y, predictions, sample_weights=self.sample_weights))
    
    def fit(self, X=None, Y=None, sample_weights=None):
        if type(None) in [type(X), type(Y)]:
            pass
        else:
            self.X = X
            self.Y = Y
            
        if type(sample_weights) != type(None):
            self.sample_weights = sample_weights
        
        self.random_forest.fit(self.X, self.Y,
            sample_weight=np.array(self.sample_weights)
        )
        

            
def get_model_df(df, metric='num_clients', col_type="X", test_data=False, include_lag=True):
    assert col_type in ["X", "Y"]
    #assert model_type in ["train", "test"]
    output_cols = ['num_clients', 'num_sessions', 'download_traffic', 'total_traffic']
    assert metric in output_cols
    lag_column = "{}_lag_7".format(metric)
    
    # Get right columns
    if col_type=="X":
        lag_cols = [c for c in df.columns if "lag" in c]
        omit_cols = ['date'] + output_cols + lag_cols
        columns = [c for c in df.columns if c not in omit_cols]
        if include_lag:
            columns.append(lag_column)
            
    elif col_type=="Y":
        columns = [metric]
    
    
    # Get right rows
    # not null output & date before test period
    if test_data==True:
        rows = (df.reset_index().date >= datetime.date(2017,12,21)) & (df.reset_index().date <= datetime.date(2017,12,27))
        rdf = deepcopy(df.reset_index().loc[rows, ['date'] + columns].set_index('date'))
        
        weather_avg_rows = (df.reset_index().date >= datetime.date(2017,12,7)) & (df.reset_index().date <= datetime.date(2017,12,20))
        weather_cols = [c for c in df.columns if c.startswith("wu_")]
        for c in weather_cols:
            rdf.loc[:,c] = df.reset_index().loc[weather_avg_rows, c].mean()
        return(rdf)
        
    else:    
        if include_lag:
            rows = ~(df.reset_index()[metric].isnull()) & ~(df.reset_index()[lag_column].isnull()) & (df.reset_index().date < datetime.date(2017,12,21))
        else:
            rows = ~(df.reset_index()[metric].isnull()) & (df.reset_index().date < datetime.date(2017,12,21))
        return(df.reset_index().loc[rows, ['date'] + columns].set_index('date'))

    
def get_sample_weights(index, lookback=20, dataframe=False):
    og_index = deepcopy(index)
    index = sorted(index)
    week_numbers = [datetime.date(i.year, i.month, i.day).isocalendar()[1] for i in index]
    wts = pd.DataFrame(week_numbers, columns=["week"], index=index)
    wts.loc[wts.week==52, "week"] = 0
    
    # Lookback = number of weeks before test period
    # which to weight upward (using exponential smoothing)
    wts["sample_weight"] = 1
    
    last_week = 51
    last_week_of_ones = last_week - lookback
    
    v = np.arange(lookback)
    upweights = np.flip(1+np.exp(-5*v/len(v)),0)
    
    weeks_to_upweight = wts.loc[wts.week > last_week_of_ones, "week"]
    for i, w in enumerate(sorted(weeks_to_upweight.unique())):
        wts.loc[wts.week==w, "sample_weight"] = upweights[i]       
        
    # Add emphasis for days around Thanksgiving
    wts.loc[[(d.month,d.day) in [(11,22), (11,23), (11,24)] for d in wts.reset_index()["index"]], "sample_weight"] = 2
    
    if dataframe==True:
        return(wts)
    return(wts.loc[og_index, "sample_weight"])


class UsagePredictor:
    def __init__(self, daily_model, hdf):
        self.daily_model = daily_model
        self.hdf = hdf
        self.X = daily_model.X
        self.Y = daily_model.Y
        
    def predict(self, X):
        hdf = self.hdf
        daily_preds = pd.DataFrame({"date": pd.to_datetime(pd.Series(X.index)).values, 
                            "prediction": self.daily_model.predict(X)})

        daily_preds["day_of_week"] = daily_preds.date.apply(
            lambda x : ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'][x.weekday()]
        )

        df = pd.DataFrame({
            "date": np.array([[d]*6 for d in daily_preds["date"]]).ravel(),
            "day_of_week": np.array([[d]*6 for d in daily_preds["day_of_week"]]).ravel(),
            "hour": [2,6,10,14,18,22]*X.shape[0]
        })
        df["prediction"] = np.nan

        for i, day in enumerate(X.index):
            day_of_week = str(daily_preds.loc[daily_preds.date==day, "day_of_week"].values[0])
            day_traffic = float(daily_preds.loc[daily_preds.date==day, "prediction"].values[0])

            for hour in [2,6,10,14,18,22]:
                hourly_prop = float(hdf.loc[(hdf.day_of_week==day_of_week) & (hdf.hour==hour),"hourly_proportion"].values[0])
                df.loc[(df.date==day) & (df.hour==hour), "prediction"] = hourly_prop*day_traffic

        df["datetime"] = pd.to_datetime(df.apply(lambda row: datetime.datetime(row.date.year, row.date.month, row.date.day, row.hour), axis=1))
        df = df.set_index("datetime")
        hourly_predictions = df.prediction
        return(hourly_predictions)
    
    

def get_test_X(df, training_object):
    metric = training_object.Y.name
    mdf = get_model_df(df, metric=metric, col_type="X", test_data=True)
    data_object = ModelData(mdf, features=training_object.features, interactions=training_object.interactions, scaler=training_object.scaler)
    
    return(data_object.X)
