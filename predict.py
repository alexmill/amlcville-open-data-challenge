import argparse
import pickle
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)

parser = argparse.ArgumentParser(description='This script generates predictions for each of the three variables')
parser.add_argument('-v','--variable', help='Which variable to predict: must be one of clients, sessions, or usage', required=True)
parser.add_argument('-x','--predictors', help='Input data to generate predictions. Data must be formatted properly (see Notebook Walkthrough file). Must either be a range of row numbers from the training data (e.g., "10,20") or the word "test" to generate predictions on test data.', required=True)
args = vars(parser.parse_args())

if __name__ == "__main__":
    variable = args["variable"]
    pred = args["predictors"]
    
    assert variable in ["clients", "sessions", "usage"]

    with open("alexs_models/{}.pickle".format(variable), "rb") as input_file:
        model = pickle.load(input_file)

    if variable=="usage":
        from alexs_models.imports import UsagePredictor
        hourly_proportions = pd.read_csv("my_data/training_data/hourly_proportions.csv", index_col=0)
        model = UsagePredictor(model, hourly_proportions)
        
    if pred=="test":
        X = pd.read_csv("my_data/test_data/{}.csv".format(variable), index_col=0)
    elif "," in pred:
        rows = [int(r) for r in pred.split(",")]
        X = model.X.iloc[rows[0]:rows[1],:]
    else:
        raise Exception('-x/--predictors input must either be a range of row numbers from the training data (e.g., "10,20") or the word "test" to generate predictions on test data.')
    
    predictions = model.predict(X)
    predictions.index = pd.to_datetime(predictions.index) + pd.DateOffset(hours=0)
    #print(predictions)
    
    for r in predictions.index:
        print("{}, {}".format(r, predictions.loc[r]))
