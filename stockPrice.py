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
 
 # %% STEP 7: THE MODEL + TRAINING FUNCTION
def build_model(window, n_features):
    """A 2-layer LSTM. An LSTM is a neural network with 'memory' for sequences."""
    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(64, return_sequences=True),  # return_sequences=True passes the whole
                                                       # sequence on so a second LSTM can stack on top
        keras.layers.Dropout(0.2),                     # randomly ignores 20% of units while training;
                                                       # this fights overfitting (memorizing the past)
        keras.layers.LSTM(32),                         # second LSTM returns just its final summary
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),     # a small regular layer to combine the signals
        keras.layers.Dense(1),                         # final output: ONE number, tomorrow's return
    ])
    # Adam = the optimizer that adjusts the weights. Huber loss = like mean squared error,
    # but less thrown off by rare giant market moves.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
    return model
 
 def scale(X_part, scaler):
    """Standardize features (mean 0, spread 1) so no feature dominates just because its numbers
    are bigger (RSI is ~50 while returns are ~0.01). Clip extremes to +/-5 for stability."""
    n, w, f = X_part.shape
    return np.clip(scaler.transform(X_part.reshape(-1, f)).reshape(n, w, f), -5, 5)
 
 
def fit_model(X_train, y_train):
    """Train a fresh model. Returns (model, scaler, y_std) so we can predict later."""
    keras.utils.set_random_seed(42)   # makes results repeatable run to run
 
    # IMPORTANT: the scaler learns its mean/spread from TRAINING data only. If it peeked at the
    # test period, information from the "future" would leak into training and inflate results.
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[2]))
    y_std = float(np.std(y_train))            # returns are tiny (~0.01), so we rescale them too
    Xs, ys = scale(X_train, scaler), y_train / y_std

   # Hold out the LAST 10% of training data (in time order!) to monitor progress.
    n_val = max(int(len(Xs) * 0.1), 1)
    model = build_model(X_train.shape[1], X_train.shape[2])
    model.fit(
        Xs[:-n_val], ys[:-n_val],
        validation_data=(Xs[-n_val:], ys[-n_val:]),
        epochs=EPOCHS, batch_size=32, verbose=0,
        # Early stopping: quit when validation error stops improving for 5 epochs,
        # and roll back to the best version. Another guard against overfitting.
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    return model, scaler, y_std
 
 
def predict_returns(model, scaler, y_std, X_part):
    """Predict next-day log returns (undoing the rescaling of y)."""
    return model.predict(scale(X_part, scaler), verbose=0).ravel() * y_std

def predict_returns(model, scaler, y_std, X_part):
    """Predict next-day log returns (undoing the rescaling of y)."""
    return model.predict(scale(X_part, scaler), verbose=0).ravel() * y_std
 
 
# %% STEP 8: WALK-FORWARD VALIDATION (the honest test)
# The tutorial approach trains once and tests on the last 5%. That's one lucky-or-unlucky sample.
# Walk-forward repeats it:  train on the past -> predict the next block -> slide forward.
#
#   Fold 1: [ train ................ ][test]
#   Fold 2: [ train ....................][test]
#   Fold 3: [ train ........................][test]
#
# The model never sees the future during training, just like in real life.
n = len(y)
assert n >= MIN_TRAIN + N_SPLITS * TEST_SIZE, (
    f"Only {n} samples. Use an earlier START, or lower N_SPLITS / TEST_SIZE."
)
 
predicted = np.full(n, np.nan)   # will hold predictions for test blocks only
for k in range(N_SPLITS):
    test_end = n - (N_SPLITS - 1 - k) * TEST_SIZE
    test_start = test_end - TEST_SIZE
    print(f"Fold {k + 1}/{N_SPLITS}: training on {test_start} samples, testing on {TEST_SIZE}...")
    model, scaler, y_std = fit_model(X[:test_start], y[:test_start])
    predicted[test_start:test_end] = predict_returns(model, scaler, y_std, X[test_start:test_end])

# Keep only the rows that were actually predicted (the test blocks).
m = ~np.isnan(predicted)
pred_ret, true_ret = predicted[m], y[m]
t_dates, t_close, t_ma5 = dates[m], close_arr[m], ma5_arr[m]
print("Done. Out-of-sample predictions:", len(pred_ret))
 
 
# %% STEP 9: COMPARE AGAINST BASELINES
# A model is only impressive if it beats simple rules. We convert returns back to prices:
# price_tomorrow = price_today * e^(return).
true_price = t_close * np.exp(true_ret)
true_dir = np.sign(true_price - t_close)    # +1 went up, -1 went down