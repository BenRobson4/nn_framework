from Predictor import Predictor
from data_preparers.DataPreparerTemplate import DataPreparerTemplate
from loss_functions.CustomLossTemplate import CustomLossTemplate
from nn_utils import *
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from config.logging_config import setup_logging
import datetime

class Main:
    def __init__(self, debug=False):
        setup_logging(debug)

    def train_new_model(model_name, data_preparer, loss_function, tickers, data, epochs=25, debug=False):
        """
        Train a new model using the provided data.
        Args:
            model_name (str): Name of the model to train
            data_preparer (DataPreparer): DataPreparer object to prepare data
            loss_function (LossFunction): LossFunction object to calculate loss
            tickers (list): List of tickers to train on
            data (dict): Dictionary of dataframes for each ticker
            epochs (int): Number of epochs to train the model
            debug (bool): Enable debug mode
        Returns:
            None
        """

        # Create a predictor object
        predictor = Predictor(
            model=None,
            model_name=model_name, 
            loss_function=loss_function,
            data_preparer=data_preparer,
            tickers=tickers,
            debug=debug
            )

        # Prepare data
        X, y, ticker_indices, ticker_map = predictor.data_preparer.prepare_data(data)

        # Train model
        predictor.train(X, y, ticker_indices, epochs)

    def train_old_model(filepath, data, debug=False):
        """
        Train an existing model using the provided data.
        Args:
            filepath (str): Path to the model file
            data (dict): Dictionary of dataframes for each ticker
            debug (bool): Enable debug mode
        Returns:
            None
        """

        # Define parameters for loading a model
        model = [filepath, True]
        debug=debug

        # Create a predictor object
        predictor = Predictor(
            model = model,
            debug=debug
        )

        # Train model with new data
        X, y, ticker_indices, ticker_map = predictor.data_preparer.prepare_data(data)
        predictor.train(X, y, ticker_indices, epochs=25)

    def predict(model_path, data, debug=False):
        """
        Predict using a trained model.
        Args:
            model_path (str): Path to the model file
            data (dict): Dictionary of dataframes for each ticker
            debug (bool): Enable debug mode
        Returns:
            None
        """
        # Define parameters for loading a model
        model = [model_path, False]
        debug=debug

        # Create a predictor object
        predictor = Predictor(
            model = model,
            debug=debug
        )

        # Prepare data
        X, y, ticker_indices, ticker_map = predictor.data_preparer.prepare_data(data)

        # Predict
        predictions = predictor.predict(X, ticker_indices, ticker_map)
        return predictions
    
if __name__ == "__main__":
    # Define parameters
    model_name = "test_model"
    sequence_length = 10
    debug = True

    # Load data
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start="2020-01-01", end="2021-01-01", interval="1d")

    # Create a data preparer object
    scalers = {ticker: MinMaxScaler() for ticker in tickers}
    data_preparer = DataPreparerTemplate(scalers, sequence_length, debug=debug)

    # Create a loss function object
    loss_function = CustomLossTemplate(debug=debug)

    # Train a new model
    Main.train_new_model(model_name, data_preparer, loss_function, tickers, data, epochs=5, debug=debug)

    # Predict using the trained model
    model_path = f"./models/test_model/finished/{datetime.datetime.today().date()}.pth"
    predictions = Main.predict(model_path, data, debug=True)
    prepared_data, ticker_map = data_preparer.prepare_features(data)
    for ticker in tickers:
        print(f"{ticker}: {predictions[-1]}, {prepared_data[ticker_map[ticker]]}")

