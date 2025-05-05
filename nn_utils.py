import numpy as np
from math import log10, floor

def to_x_sig_figs(num, sig_figs):
    """
    Round a number to a specified number of significant figures
    Args:
        num (float): Number to round
        sig_figs (int): Number of significant figures
    Returns:
        float: Rounded number
    """
    highest_power = floor(log10(abs(num)))
    return round(num, sig_figs - 1 - highest_power)
