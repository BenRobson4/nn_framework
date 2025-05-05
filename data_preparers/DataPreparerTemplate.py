import numpy as np
import pandas as pd
import torch
from data_preparers.DataPreparer import DataPreparer

class DataPreparerTemplate(DataPreparer):
    def __init__(self, scalers, sequence_length, debug=False):
        """
        Template class for data preparation. Modify the prepare_data method to suit your needs.
        Inherits metrics such as RSI, SMA, EMAs, MACD, and Bollinger Bands from the DataPreparer class.
        """
        super().__init__(scalers, sequence_length, debug)
        self.logger.debug("Debug mode enabled for DataPreparerTemplate")

    def prepare_features(self, all_data):
        """
        Prepare data for all tickers
        Args:
            all_data (dict): Dictionary of dataframes for each ticker
            Columns: ['Open', 'High', 'Low', 'Close', 'Volume'] if using yfinance
        Returns:
            X (torch.Tensor): Input data (shape [n_samples, sequence_length, n_features])
            y (torch.Tensor): Target data (shape [n_samples])
            ticker_indices (torch.Tensor): Indices of tickers (shape [n_samples])
            ticker_map (dict): Map of ticker names to indices
        """
        self.logger.debug(f"Debug shapes before preparation:")
        self.logger.debug(f"all_data shape: {all_data.keys()}")
        combined_features = []
        ticker_map = {}  # Map to keep track of which sequences belong to which ticker
        
        for idx, (ticker, data) in enumerate(all_data.items()):
            self.logger.debug(f"Processing data for {ticker}")
            self.logger.debug(f"Data: {data.head(2)}")
            self.logger.debug(f"Data shape: {data.shape}")
            # Create features for this ticker
            features = pd.DataFrame(index=data.index)
            # Define features directly from data
            # Below are examples pulled directly from the data or calculated using the DataPreparer class
            features['Close'] = data['Close']
            features['Volume'] = data['Volume']
            features['SMA20'] = super().calculate_sma(data['Close'], 20)
            features['RSI'] = super().calculate_rsi(data['Close'])
            BB_dict = super().calculate_bollinger_bands(data['Close'])
            features['BB_Upper'] = BB_dict['Upper Band']
            MACD_dict = super().calculate_macd(data['Close'])
            features['MACD'] = MACD_dict['MACD Line']

            # Handle NaN values **IMPORTANT** (annoying to debug)
            features = super().handle_missing_values(features)
            features = features.ffill().bfill()
            
            # Add features to combined_features and update ticker_map
            combined_features.append(features)
            ticker_map[ticker] = idx
            self.logger.debug(f"Ticker map: {ticker_map}, idx: {idx}")

        return combined_features, ticker_map
