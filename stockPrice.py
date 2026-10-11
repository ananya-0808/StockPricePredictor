import numpy as np                         # fast math on arrays
import pandas as pd                        # tables (DataFrames), good for time series
import matplotlib.pyplot as plt            # charts
import yfinance as yf                      # downloads free stock data from Yahoo Finance
from sklearn.preprocessing import StandardScaler   # rescales numbers (explained in Step 5)
from tensorflow import keras               # builds and trains the neural network

TICKER = "AAPL"        # which stock. Try "MSFT", "NVDA", "SPY" (an S&P 500 fund), etc.
START = "2015-01-01"   # how far back to download. More history = more training data.
WINDOW = 60            # the model looks at the last 60 trading days to make each prediction.
N_SPLITS = 4           # how many times we test the model (walk-forward folds, see Step 7).
TEST_SIZE = 100        # each test block is 100 trading days (~5 months).
MIN_TRAIN = 400        # require at least this many training samples before the first test.
EPOCHS = 25            # max passes over the training data per fold (early stopping may end sooner).

df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
 
# Newer versions of yfinance return two-level column names like ("Close", "AAPL").
# This flattens them to just "Close", so the rest of the code works either way.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
 
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
print(f"Downloaded {len(df)} trading days for {TICKER}")
df.tail()   # in VS Code's interactive window, the last line of a cell displays as a table

 # %% STEP 4: FEATURE ENGINEERING
# A "feature" is one input number the model can learn from. Instead of feeding raw prices
# (which just trend upward over the years), we use SCALE-FREE features like percentages and
# ratios. That way "a 2% move" means the same thing in 2016 and in 2025.
close = df["Close"]
features = pd.DataFrame(index=df.index)
 
# --- Returns: how much the price changed. We use LOG returns: log(today / yesterday).
# They behave like percentage changes, but add up nicely across days.
features["ret_1"] = np.log(close).diff()           # 1-day return
features["ret_5"] = np.log(close).diff(5)          # 5-day return (short-term momentum)
 
# --- Candle shape: what happened inside the day
features["hl_range"] = (df["High"] - df["Low"]) / close     # how wide the day's swing was
features["oc_change"] = (close - df["Open"]) / df["Open"]   # open-to-close change
 
# --- Volume: how much trading happened (log1p squashes huge numbers; diff = change vs yesterday)
features["vol_chg"] = np.log1p(df["Volume"]).diff()
 
# --- Moving averages: is the price above or below its recent average?
# (close / average) - 1 = how far above (+) or below (-) the average we are, as a fraction.
features["sma10_ratio"] = close / close.rolling(10).mean() - 1
features["sma30_ratio"] = close / close.rolling(30).mean() - 1
 