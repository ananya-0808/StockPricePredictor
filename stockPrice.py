import numpy as np #math for arrays
import pandas as pd #tables
import matplotlib.pyplot as plt #charts
import yfinance as yf #uses yahoo finance data
from sklearn.preprocessing import StandardScaler #rescales numbers
from tensorflow import keras #builds and trains the neural network

TICKER = "AAPL" #specific stock to use
START = "2015-01-01" #how far back to download
WINDOW = 60
N_SPLITS = 4
TEST_SIZE = 100
MIN_TRAIN = 400
EPOCHS = 25

df = yf.download(TICKER, start = START, auto_adjust = True, progress = False)