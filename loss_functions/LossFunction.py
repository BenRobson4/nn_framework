import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

class LossFunction(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.num_targets = 0
        self.num_targets, self.targets = self.get_targets()
        self.required_params = self.required_parameters()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def generate_log(self, message):
        self.logger.debug(message)

    def validate_inputs(self, predictions, targets, parameters):
        if not isinstance(predictions, torch.Tensor):
            raise ValueError("Predictions must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise ValueError("Targets must be a torch.Tensor")
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape")

    @abstractmethod
    def forward(self, predictions, targets, parameters):
        """
        Compute the loss given predictions and targets.
        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.
            parameters (Dict): Additional context or data (if needed), e.g. {'previous_prices':}
        Returns:
            torch.Tensor: The computed loss value.
        """
        self.validate_inputs(predictions, targets, parameters)
        #self.logger.debug(f"Debug Mode: Predictions: {predictions}, Targets: {targets}, Additional Parameters: {parameters}")
        # Placeholder for derived class implementation
        pass

    @abstractmethod
    def get_targets(self):
        """
        Placeholder for a method to get the number of target values.
        """
        pass

    @abstractmethod
    def required_parameters(self):
        """
        Return a list of required parameters for the loss function.
        Args:
            None
        Returns:
            List: List of required parameters
        """
        pass

if __name__ == "__main__":
    # Test the LossFunction class
    loss_function = LossFunction(debug=True)
    loss_function.generate_log("Test log message")
