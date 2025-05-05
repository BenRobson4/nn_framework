from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import torch

class DataPreparer(ABC):
    def __init__(self, scalers, sequence_length=60, debug=False):
        """
        Base class for data preparation.
        This class can be extended to implement specific data preparation strategies.
        Args:
            sequence_length (int): Number of previous time steps to consider for prediction
            scalers (dict): Dictionary of scalers for each ticker (e.g. {'AAPL': MinMaxScaler()} or {ticker: MinMaxScaler() for ticker in tickers})
        """
        self.num_features = 0
        self.features = []
        self.sequence_length = sequence_length
        self.scalers = scalers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
    @abstractmethod
    def prepare_features(self, all_data):
        """
        Prepare data for all tickers.
        This method must be implemented by subclasses.
        Args:
            all_data (dict): Dictionary of dataframes for each ticker
        Returns:
            combined_features (list): List of the features for each ticker (shape [n_samples, sequence_length, n_features])
            ticker_map (dict): Map of ticker names to their position in the combined_features list
        """
        pass

    def prepare_data(self, all_data):
        """
        Prepare data for training as torch tensors for prediction
        Args:
            X (list): List of input sequences
            y (list): List of target values
            ticker_indices (list): List of ticker indices
            ticker_map (dict): Map of ticker names to indices
        Returns:
            X (torch.Tensor): Input data (shape [n_samples, sequence_length, n_features])
            y (torch.Tensor): Target data (shape [n_samples])
            ticker_indices (torch.Tensor): Indices of tickers (shape [n_samples])
            ticker_map (dict): Map of ticker names to indices
        """

        combined_features, ticker_map = self.prepare_features(all_data)
        self.logger.debug(f"Combined Features: {combined_features[0].shape}")

        # Create sequences for each ticker
        X, y, ticker_indices = [], [], []
        self.logger.debug(f"Ticker Map: {ticker_map}")
        self.logger.debug(f"Scalers: {self.scalers}")
        for ticker, idx in ticker_map.items():
            ticker_data = combined_features[idx]
            self.logger.debug(f"Ticker: {ticker}, Data: {ticker_data.head(2)}")
            ticker_scaler = self.scalers[ticker]
            scaled_data = self.scale_data(ticker_data, ticker_scaler)
            
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i]) # Add the sequence to the input data
                y.append(scaled_data[i])  # Add the next data point as the target
                ticker_indices.append(idx)

        # Transform to torch tensors
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)
        ticker_indices = torch.LongTensor(ticker_indices).to(self.device)

        self.logger.debug(f"Debug shapes after preparation:")
        self.logger.debug(f"X shape: {X.shape}")
        self.logger.debug(f"y shape: {y.shape}")
        self.logger.debug(f"ticker_indices shape: {ticker_indices.shape}")
        
        return X, y, ticker_indices, ticker_map

    def get_features(self):
        """
        Returns the number of features in the prepare_data output.
        This method must be implemented by subclasses.
        Args:
            None
        Returns:
            integer: Number of features
            list: List of feature names
        """
        if self.num_features > 0:
            # Return the number of features if already calculated
            return self.num_features, self.features
        else:
            # Create mock data to calculate the number of features otherwise
            mock_data = self.create_mock_data(['AAPL'])
            self.logger.debug(f"Mock data: {mock_data['AAPL'].head(2), mock_data['AAPL'].tail(2)}")

            combined_features, _ = self.prepare_features(mock_data)

            self.num_features = combined_features[0].shape[1] if combined_features[0].shape else 0 # Evaluate and set the number of features as an attribute
            self.features = combined_features[0].columns  # Set the features as an attribute

            self.logger.debug(f"Number of features: {self.num_features}")
            self.logger.debug(f"Features: {self.features}")

            assert self.num_features == len(self.features), "Number of features does not match the number of columns in the DataFrame"
            return self.num_features, self.features

    def create_mock_data(self, tickers, num_days=100):
        """
        Sub-method to generate mock data.
        Primarily used in num_features method.
        Args:
            tickers (list): List of tickers
            num_days (int): Number of days to generate
        Returns:
            dict: Dictionary of mock data
            format: The same as prepare_data input,
              dictionary of Dataframes with the same columns as yf.download output)
        """
        data = {}
        # Generate a date range
        dates = pd.date_range(start='2022-01-01', periods=num_days, freq='B')  # Business days
        # Create random data for the DataFrame
        for ticker in tickers:
            mock_entry = {
                'Open': np.random.uniform(low=100, high=200, size=num_days),
                'High': np.random.uniform(low=200, high=300, size=num_days),
                'Low': np.random.uniform(low=50, high=100, size=num_days),
                'Close': np.random.uniform(low=100, high=200, size=num_days),
                'Adj Close': np.random.uniform(low=100, high=200, size=num_days),
                'Volume': np.random.randint(low=1000, high=10000, size=num_days)
            }
            # Create the DataFrame
            df = pd.DataFrame(mock_entry, index=dates)
            # Add the DataFrame to the dictionary
            data[ticker] = df
        return data

    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to process
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        return df.ffill().bfill()  # Forward and backward fill

    def scale_data(self, data, scaler):
        """
        Scale the data using the provided scaler.
        Args:
            data (pd.DataFrame): Data to scale
            scaler: Scaler object (e.g., MinMaxScaler)
        Returns:
            np.ndarray: Scaled data
        """
        self.logger.debug(f"Scaling data with shape: {data.shape}")

        return scaler.fit_transform(data)

    def calculate_sma(self, prices, window):
        """Calculate simple moving average (SMA) for a price series.
        Args:
            prices (pd.Series): Price series
            window (int): Window size for moving average 
        Returns:
            pd.Series: Moving average values
        """
        self.logger.debug(f"Calculating SMA for prices: {prices.head(2), prices.tail(2)} with window size: {window}")

        sma_values = prices.rolling(window=window, min_periods=1).mean()

        self.logger.debug(f"Calculated SMA values: {sma_values.head(2), sma_values.tail(2)}")

        return sma_values
    
    def calculate_rsi(self, prices, window=14):
        """
        Calculate the Relative Strength Index (RSI) for a price series.
        Args:
            prices (pd.Series): Price series
            window (int): Window size for RSI calculation
        Returns:
            pd.Series: RSI values
        """
        self.logger.debug(f"Calculating RSI for prices: {prices.head(2), prices.tail(2)} with window size: {window}")

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss

        self.logger.debug(f"Calculated RSI values: {rs.head(2), rs.tail(2)}")

        return 100 - (100 / (1 + rs))
    
    def calculate_ema(self, prices, span):
        """
        Calculate the Exponential Moving Average (EMA) for a price series.
        Args:
            prices (pd.Series): Price series (e.g., closing prices).
            span (int): The span for the EMA.
        Returns:
            pd.Series: EMA values.
        """
        self.logger.debug(f"Calculating EMA for prices: {prices.head(2), prices.tail(2)} with span: {span}")

        ema_values = prices.ewm(span=span, min_periods=1, adjust=False).mean()

        self.logger.debug(f"Calculated EMA values: {ema_values.head(2), ema_values.tail(2)}")
        
        return ema_values
    
    def calculate_macd(self, prices, short_window=12, long_window=26, signal_window=9):
        """
        Calculate the MACD for a price series using the EMA function.
        Args:
            prices (pd.Series): Price series (e.g., closing prices).
            short_window (int): The short period for the EMA.
            long_window (int): The long period for the EMA.
            signal_window (int): The period for the signal line EMA.
        Returns:
            pd.DataFrame: DataFrame containing the MACD line and the signal line.
            Columns are labelled: "MACD Line", "Signal Line".
        """
        self.logger.debug(f"Calculating MACD for prices: {prices.head(2), prices.tail(2)} with short window: {short_window}, long window: {long_window}, signal window: {signal_window}")

        short_ema = self.calculate_ema(prices, short_window)
        long_ema = self.calculate_ema(prices, long_window)
        
        macd_line = short_ema - long_ema
        signal_line = self.calculate_ema(macd_line, signal_window)

        macd_dict = {
            'MACD Line': macd_line,
            'Signal Line': signal_line
        }

        self.logger.debug(f"Calculated MACD values: {macd_dict['MACD Line'].head(2), macd_dict['MACD Line'].tail(2)}")
        
        return macd_dict
    
    def calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        """
        Calculate Bollinger Bands for a price series.
        
        Args:
            prices (pd.Series): Price series (e.g., closing prices).
            window (int): The number of periods for the moving average.
            num_std_dev (int): The number of standard deviations for the bands.
        
        Returns:
            pd.DataFrame: DataFrame containing the middle band, upper band, and lower band.
            Columns are labelled: "Middle Band", "Upper Band", "Lower Band".
        """
        self.logger.debug(f"Calculating Bollinger Bands for prices: {prices.head(2), prices.tail(2)} with window: {window}, num_std_dev: {num_std_dev}")

        middle_band = prices.rolling(window=window, min_periods=1).mean()
        std_dev = prices.rolling(window=window, min_periods=1).std()
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)

        self.logger.debug(f"Calculated Bollinger Bands: {middle_band.head(2), middle_band.tail(2)}")
        self.logger.debug(f"Structure of Bollinger Bands: {middle_band.shape, upper_band.shape, lower_band.shape}")

        bb_dict = {
            'Middle Band': middle_band,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        }

        self.logger.debug(f"Calculated Bollinger Bands: {bb_dict['Middle Band'].head(2), bb_dict['Middle Band'].tail(2)}")
                          
        return bb_dict