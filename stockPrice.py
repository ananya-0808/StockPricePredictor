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

# --- Volatility: how bumpy the last 10 days were (standard deviation of daily returns)
features["volatility_10"] = features["ret_1"].rolling(10).std()
 
# --- RSI (Relative Strength Index): 0-100 scale of recent gains vs losses.
# Above ~70 is traditionally "overbought", below ~30 "oversold".
delta = close.diff()
avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
features["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
 
# --- MACD: difference between a fast (12-day) and slow (26-day) exponential moving average.
# It's a momentum signal. The "signal line" is a smoothed version of it.
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
features["macd"] = macd / close                                     # divided by price = scale-free
features["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / close

# --- Bollinger Bands: a band of +/- 2 standard deviations around the 20-day average.
sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
features["bb_width"] = (4 * std20) / sma20                          # how wide the band is
features["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)    # where price sits (0=bottom, 1=top)
 
print(f"{features.shape[1]} features built:", list(features.columns))
 
 
# %% STEP 5: THE TARGET, AND CLEANING
# What are we predicting? TOMORROW's log return. `.shift(-1)` pulls tomorrow's value up to
# today's row, so each row reads: "given everything known today, what happens tomorrow?"
target = np.log(close).diff().shift(-1)

# Baseline helper: the average of the last 5 closing prices (used as a comparison later).
ma5 = close.rolling(5).mean()
 
# The first ~30 rows have missing values (a 30-day average needs 30 days of history).
# Keep only rows where every feature exists.
valid = features.notna().all(axis=1)
features, target, close, ma5 = features[valid], target[valid], close[valid], ma5[valid]

# %% STEP 6: BUILD "WINDOWS" (what the LSTM actually sees)
# An LSTM reads SEQUENCES. Each training example is a block of the last 60 days of features
# (shape: 60 days x 13 features), paired with ONE answer: the return on the following day.
F = features.values.astype("float32")
end_rows = np.arange(WINDOW - 1, len(F))   # the last day of each window
 
X_all = np.stack([F[i - WINDOW + 1 : i + 1] for i in end_rows])   # (samples, 60, 13)
y_all = target.values[end_rows]
dates_all = features.index[end_rows]
close_all = close.values[end_rows]
ma5_all = ma5.values[end_rows]
 
# The newest window has no "tomorrow" yet, so it has no answer (NaN). We can't train or test on
# it, but we save it: it's the window we'll use to forecast the real next day at the very end.
X_latest = X_all[-1:]
has_answer = ~np.isnan(y_all)
X, y = X_all[has_answer], y_all[has_answer]
dates, close_arr, ma5_arr = dates_all[has_answer], close_all[has_answer], ma5_all[has_answer]
 
print("X shape:", X.shape, "-> (samples, days in window, features)")
print("y shape:", y.shape, "-> one next-day return per sample")
 