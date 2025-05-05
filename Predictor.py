import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import torch
import datetime
from sklearn.preprocessing import MinMaxScaler
from LSTM import LSTM
from typing import final
from nn_utils import *

class Predictor:
    def __init__(self, model=None, model_name=None, loss_function=None, data_preparer=None, tickers=None, debug=False):
        """
        Base predictor class for training and evaluating models
        Args:
            model: in the form [model_filepath (str), for_training (bool)], None if creating a new model
            model_name: name of the model, None if loading a model
            loss_function: the loss function to use, must be a subclass of LossFunction, None if loading a model
            data_preparer: the data preparer to use, must be a subclass of DataPreparer, None if loading a model
            optimizer: the optimizer to use, e.g. torch.optim.Adam, None if loading a model
            tickers: list of tickers to predict, given a list e.g. ['AAPL', 'GOOGL'], None if loading a model
            sequence_length: number of time steps to consider, integer 60 is default
            debug: if True, print debug information, False by default
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.debug("Debug mode enabled for Predictor")

        if model:
            model_filepath, for_training = model
            self.load_model(model_filepath, for_training)
            self.for_training = for_training
        else:
            self.name = model_name
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sequence_length = data_preparer.sequence_length
            self.loss_function = loss_function
            self.data_preparer = data_preparer
            self.model = self.build_model(input_shape=(None, self.sequence_length, self.data_preparer.get_features()[0], self.loss_function.get_targets()[0]))
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.tickers = tickers
            self.scalers = data_preparer.scalers
            self.for_training = True
            self.current_epoch = 0  # Track current epoch
        
        assert all(target in self.data_preparer.get_features()[1] for target in self.loss_function.targets), "Loss function target outputs do not match data preparer features"
        assert all(param in self.data_preparer.get_features()[1] for param in self.loss_function.required_params), "Loss function required parameters do not match data preparer features"

    @final
    def build_model(self, input_shape):
        """
        Create LSTM model
        Args:
            input_shape: shape of input data (batch_size, sequence_length, num_features, num_outputs), e.g. (None, 60, 7, 7) where None means we can use a variable batch size
        Returns:
            LSTM model
        """
        if isinstance(input_shape, int):
            raise ValueError(f"Expected tuple or list for input_shape, got int: {input_shape}")
        
        if not os.path.exists(f'./models/{self.name}'):
            os.makedirs(f'./models/{self.name}')
            os.makedirs(f'./models/{self.name}/checkpoints')
            os.makedirs(f'./models/{self.name}/finished')
        
        try:
            input_size = input_shape[2]  # Number of features
            output_size = input_shape[3]  # Number of outputs
            
            self.model = LSTM(
                input_size=input_size,
                hidden_size=50,
                num_layers=2,
                output_size=output_size
            ).to(self.device)
            return self.model
        
        except Exception as e:
            print(f"Error in build_model: {e}")
            print(f"Input shape type: {type(input_shape)}")
            raise
    
    @final
    def save_model(self, filepath):
        """
        Save the model, optimizer state, and training data
        Args:
            filepath: path to save the model
        Returns:
            None
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),  # Save optimizer state
            'name': self.name,
            'loss_function': self.loss_function,    # Standard nn.Module object so hopefully no serialization issues
            'data_preparer': self.data_preparer,    # A little more complex, especially with the scalers dict so stored scalers and sequence length separetely
            'tickers': self.tickers,
            'sequence_length': self.sequence_length,
            'scalers': self.scalers,
            'epoch': self.current_epoch  # Add epoch tracking
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @final
    def load_model(self, filepath, for_training=True):
        """
        Load a saved model and its data
        Args:
            filepath: path to saved model
            for_training: if True, prepare model for further training
        Returns:
            None but updates self.model and self.optimizer as well as the associated data
        """
        checkpoint = torch.load(filepath, weights_only=False)
        
        # Restore model parameters
        self.loss_function = checkpoint['loss_function']
        self.data_preparer = checkpoint['data_preparer']
        self.name = checkpoint['name']
        self.tickers = checkpoint['tickers']
        self.sequence_length = checkpoint['sequence_length']
        self.scalers = checkpoint['scalers']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = checkpoint.get('epoch', 0)  # Get last epoch
        self.for_training = for_training

        assert self.data_preparer.sequence_length == self.sequence_length, "Data preparer sequence length does not match model sequence length"
        assert self.data_preparer.scalers == self.scalers, "Data preparer scalers do not match model scalers"
        self.logger.debug(f"\nModel {self.name} loaded from {filepath}")
        self.logger.debug(f"loss_function: {self.loss_function}")
        self.logger.debug(f"data_preparer: {self.data_preparer}")
        self.logger.debug(f"tickers: {self.tickers}")
        self.logger.debug(f"sequence_length: {self.sequence_length}")
        self.logger.debug(f"scalers: {self.scalers}")
        self.logger.debug(f"epoch: {self.current_epoch}")
        
        # Rebuild model with correct input size
        num_features, _ = self.data_preparer.get_features()
        num_outputs, _ = self.loss_function.get_targets()
        self.model = self.build_model(input_shape=(None, self.sequence_length, num_features, num_outputs))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.for_training:
            self.model.train()  # Set to training mode
        else:
            self.model.eval()   # Set to evaluation mode
        
        print(f"Model loaded from {filepath} in {'training' if for_training else 'evaluation'} mode")

    def train(self, X, y, ticker_indices, epochs=50, batch_size=32):
        """
        Method to train the model
        Args:
            X (torch.Tensor): training data in the form [X_train, y_train, ticker_indices_train]
            y (torch.Tensor): validation data in the form [X_val, y_val, ticker_indices_val]
            epochs: number of epochs to train, integer 50 by default
            batch_size: size of batches, integer 32 by default
        Returns:
            None
        """
        assert self.for_training, "Model is not set for training"
        training_start = datetime.datetime.now()
        self.logger.info(f"Training started at {training_start}")
        criterion = self.loss_function
        optimizer = self.optimizer
        num_features, feature_list = self.data_preparer.get_features()

        self.logger.debug(f"Training data shape: {X.shape}, {y.shape}, {ticker_indices.shape}")

        train_dataset = torch.utils.data.TensorDataset(X, y, ticker_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        assert self.model.training, "The model is not in training mode."
        for epoch in range(epochs):
            self.current_epoch += 1
            total_loss = 0

            for batch_X, batch_y, batch_ticker_indices in train_loader:
                optimizer.zero_grad()

                # Construct output data and parameters for loss function
                outputs = self.model(batch_X)
                predictions = outputs[:, -1, :]
                previous_features = batch_X[:, -1, :]
                previous_features_np = previous_features.numpy()
                parameters = {
                    feature_list[i]: previous_features_np[:, i] for i in range(num_features)
                }
                """
                self.logger.debug(f"Inputs: {batch_X}, Shape: {batch_X.shape}")
                self.logger.debug(f"Outputs: {outputs}, Shape: {outputs.shape}")
                self.logger.debug(f"Targets: {batch_y}, Shape: {batch_y.shape}")
                self.logger.debug(f"Parameters: {parameters}")
                """
                # Compute loss
                loss = criterion(
                    predictions=predictions,
                    targets=batch_y,
                    parameters=parameters
                )
                # 
                assert not torch.isnan(loss), "NaN loss encountered"

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if (epoch + 1) % 1 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {to_x_sig_figs(total_loss, 5)}, Timestamp: {timestamp}')
            
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
               self.save_model(f'./models/{self.name}/checkpoints/{self.name}_checkpoint_epoch_{epoch+1}_{timestamp}.pth')
        
        self.save_model(f'./models/{self.name}/finished/{datetime.datetime.today().date()}.pth')
        self.logger.info(f"Training finished and model saved to /models/{self.name}/finished/{datetime.datetime.today().date()}")

    def predict(self, X, ticker_indices, ticker_map):
        """
        Method to predict the next value
        Args:
            X (torch.Tensor): input data in the form [X, y, ticker_indices]
        Returns:
            torch.Tensor: predicted values
        """
        assert not self.model.training, "The model is in training mode."
        with torch.no_grad():
            scaled_predictions = self.model(X)
            self.logger.debug(f"Scaled Predictions: {scaled_predictions}, Shape: {scaled_predictions.shape}")
            unscaled_predictions = self.unscale_predictions(scaled_predictions, ticker_indices, ticker_map)
            self.logger.debug(f"Unscaled Predictions: {unscaled_predictions}, Shape: {unscaled_predictions.shape}")
            return unscaled_predictions
    
    def unscale_predictions(self, scaled_predictions, ticker_indices, ticker_map):
        """
        Unscale predictions for each ticker
        Args:
            scaled_predictions (torch.Tensor): Scaled predictions 
                Shape: [batch_size, sequence_length, num_features]
            ticker_indices (torch.Tensor): Indices of tickers for each prediction
        Returns:
            torch.Tensor: Unscaled predictions
        """
        # Convert to numpy for inverse transform
        scaled_np = scaled_predictions.cpu().numpy()
        unscaled_predictions = np.zeros_like(scaled_np)
        
        # Unscale each prediction based on its ticker
        for i, ticker_index in enumerate(ticker_indices):
            # Find the corresponding ticker
            ticker = [key for key, value in ticker_map.items() if value == ticker_index][0]
            scaler = self.scalers[ticker]
            
            # Unscale this prediction
            unscaled_predictions[i] = scaler.inverse_transform(scaled_np[i])
        
        return torch.tensor(unscaled_predictions)