import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataPreparerTemplate import DataPreparerTemplate
from sklearn.preprocessing import MinMaxScaler
from nn_utils import *
from math import floor, log10

dpt = DataPreparerTemplate(scalers={'AAPL': MinMaxScaler()}, sequence_length=5, debug=True)
print(dpt.get_features())
print(dpt.get_features())

#print(to_x_sig_figs(0.123456789, 5))
