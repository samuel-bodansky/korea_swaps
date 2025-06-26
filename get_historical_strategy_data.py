import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.interpolate import interp1d
import os
market_data_folder = "market_data"
os.makedirs(market_data_folder, exist_ok=True)
from market_data.df_risk import risk
# --- Configuration ---
history_years = 2
window_size_days = 60  # Approximately 3 months of trading days for rolling calculations
trading_days_per_year = 252  # For annualizing volatility
window_size_days_3m = 60 # Approximately 3 months of trading days for rolling calculations
window_size_days_1m = 20 # A
# Define the tenors as numerical values directly as per your request
tenors_num = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0],
                      dtype=float)
tenors_str_cols = [str(t) for t in tenors_num]  # For DataFrame column names

# --- 2. Load Market Data ---
df_spot = pd.read_csv(os.path.join(market_data_folder, "df_spot.csv"), index_col='Dates',
                                  parse_dates=True).sort_index()
df_cr = pd.read_csv(os.path.join(market_data_folder, "df_cr.csv"), index_col='Dates',
                                   parse_dates=True).sort_index()

# Keys are strategy names, values are dictionaries with 'tenors' (numerical) and 'weights'
strategies = {
    # --- Existing Curve Strategies ---
    "2s10s Steepener": {'tenors': ['2y', '10y'], 'weights': [-1, 1]},
    "5s30s Steepener": {'tenors': ['5y', '30y'], 'weights': [-1, 1]},
    "1y5y Steepener": {'tenors': ['1y', '5y'], 'weights': [-1, 1]},
    "2y5y Flattener": {'tenors': ['2y', '5y'], 'weights': [1, -1]},
    "7y10y Flattener": {'tenors': ['7y', '10y'], 'weights': [1, -1]},
    "10y30y Steepener": {'tenors': ['10y', '30y'], 'weights': [-1, 1]},
    "10y30y Flattener": {'tenors': ['10y', '30y'], 'weights': [1, -1]},

    # --- Existing Butterfly Strategies ---
    "1y2y5y Butterfly": {'tenors': ['1y', '2y', '5y'], 'weights': [1, -2, 1]},
    "3y5y7y Butterfly": {'tenors': ['3y', '5y', '7y'], 'weights': [1, -2, 1]},
    "7y10y20y Butterfly": {'tenors': ['7y', '10y', '20y'], 'weights': [1, -2, 1]},
    "5s10s30s Butterfly": {'tenors': ['5y', '10y', '30y'], 'weights': [1, -2, 1]},

    # --- 10 NEW Butterfly Strategies ---
    # Very Short End
    "3m6m1y Butterfly": {'tenors': ['3m', '6m', '1y'], 'weights': [1, -2, 1]},
    "6m1y2y Butterfly": {'tenors': ['6m', '1y', '2y'], 'weights': [1, -2, 1]},

    # Short-Mid Curve
    "2y3y5y Butterfly": {'tenors': ['2y', '3y', '5y'], 'weights': [1, -2, 1]},
    "3y4y6y Butterfly": {'tenors': ['3y', '4y', '6y'], 'weights': [1, -2, 1]},
    "4y5y8y Butterfly": {'tenors': ['4y', '5y', '8y'], 'weights': [1, -2, 1]},

    # Mid-Long Curve
    "6y8y10y Butterfly": {'tenors': ['6y', '8y', '10y'], 'weights': [1, -2, 1]},
    "8y10y12y Butterfly": {'tenors': ['8y', '10y', '12y'], 'weights': [1, -2, 1]},
    "10y12y15y Butterfly": {'tenors': ['10y', '12y', '15y'], 'weights': [1, -2, 1]},

    # Long End
    "15y20y30y Butterfly": {'tenors': ['15y', '20y', '30y'], 'weights': [1, -2, 1]},
    "12y20y30y Butterfly": {'tenors': ['12y', '20y', '30y'], 'weights': [1, -2, 1]}, # Another long end variant
}

# --- 4. Function to Calculate Strategy Metrics ---
def calculate_strategy_metrics(strategy_def: dict,
                               df_spot_hist: pd.DataFrame,
                               df_carry_hist: pd.DataFrame,
                               window_1m: int,
                               window_3m: int,
                               annualization_factor: float,
                               benchmark_tenor: str = '5y') -> pd.DataFrame:
    """
    Calculates weighted yield, Z-score, volatility, and carry-adjusted volatility for a strategy.

    Args:
        strategy_def (dict): Dictionary with 'tenors' (list of floats) and 'weights' (list of floats).
        df_spot_hist (pd.DataFrame): Historical spot rates DataFrame.
        df_carry_hist (pd.DataFrame): Historical carry rates DataFrame (in decimal).
        df_rolldown_hist (pd.DataFrame): Historical roll-down rates DataFrame (in decimal).
        window_size (int): Rolling window size for Z-score and volatility calculations.
        annualization_factor (float): Factor to annualize volatility (e.g., sqrt(252)).

    Returns:
        pd.DataFrame: DataFrame with 'Weighted Yield', '3m Z-score', 'Vol', 'Carry Adjusted Vol' columns.
    """
    strategy_tenors = [str(t) for t in strategy_def['tenors']]  # Convert to string for column indexing
    strategy_weights = np.array(strategy_def['weights'])
    # Ensure all required tenors exist in the historical data
    missing_tenors = [t for t in strategy_tenors if t not in df_spot_hist.columns]
    if missing_tenors:
        print(f"Warning: Missing tenors {missing_tenors} for strategy. Skipping calculation.")
        return pd.DataFrame()  # Return empty DataFrame if tenors are missing

    # --- Weighted Yield ---
    # Dot product of spot rates for relevant tenors with strategy weights
    weighted_yield = df_spot_hist[strategy_tenors].dot(strategy_weights)
    weighted_yield.name = 'Weighted Yield'  # Assign a name to the Series

    # --- 3m Z-score ---
    rolling_mean = weighted_yield.rolling(window=window_3m).mean()
    rolling_std = weighted_yield.rolling(window=window_3m).std()
    z_score_3m = (weighted_yield - rolling_mean) / rolling_std
    z_score_3m.name = '3m Z-score'

    # --- Volatility (Annualized) ---
    # Calculate daily changes for volatility
    daily_yield_changes = weighted_yield.diff()
    vol = daily_yield_changes.rolling(window=window_3m).std() * annualization_factor
    vol.name = 'Vol'

    # --- Carry Adjusted Vol (Annualized) ---
    # Weighted historical carry for the strategy
    # Note: df_carry_hist should be in decimal form for this calculation
    weighted_carry = df_carry_hist[strategy_tenors].dot(strategy_weights)
    weighted_carry.name = 'Weighted Carry'
    weighted_dv01 =np.array([float(risk.get(t)) for t in strategy_tenors]).dot(strategy_weights)
    # Total return = daily yield change + weighted carry
    # This is a simplified total return for volatility calculation
    total_return = daily_yield_changes + weighted_carry
    vol_adjusted_carry = weighted_carry/(weighted_dv01*vol)
    vol_adjusted_carry.name = 'Vol Adjusted Carry'

    # Benchmark daily changes
    benchmark_spot_series = df_spot_hist[benchmark_tenor]  # benchmark_tenor is '5y' by default
    benchmark_daily_changes = benchmark_spot_series.diff()
    benchmark_spot_level = df_spot_hist[benchmark_tenor]
    benchmark_spot_level.name = f'{benchmark_tenor} Outright Level' # Name for clarity in concat

    # ... (Z-score, Volatility, Carry Adjusted Vol calculations)

    # --- Rolling Betas to Benchmark ---
    # Rolling 1-month Beta
    # Ensure non-overlapping windows for cov/var calculation by aligning their indices
    aligned_levels = pd.concat([weighted_yield, benchmark_spot_level], axis=1).dropna()

    corr_1m = aligned_levels[weighted_yield.name].rolling(window=window_1m).corr(aligned_levels[benchmark_spot_level.name])
    corr_1m.name = '1m Correlation to 5y'

    corr_3m = aligned_levels[weighted_yield.name].rolling(window=window_3m).corr(aligned_levels[benchmark_spot_level.name])
    corr_3m.name = '3m Correlation to 5y'

    # --- Absolute Difference of Betas ---
    abs_beta_diff = (corr_3m - corr_1m).abs()
    abs_beta_diff.name = 'Abs Beta Diff (1m-3m)'

    # --- Combine all results into a DataFrame ---
    strategy_df = pd.concat([weighted_yield, z_score_3m, vol, vol_adjusted_carry, corr_1m, corr_3m,abs_beta_diff,weighted_carry], axis=1)

    # --- Combine results into a DataFrame ---
    return strategy_df


# --- 5. Main Loop and Dictionary Creation ---
strategy_results_dict = {}

print("--- Calculating Strategy Metrics (including Correlation to 5y Outright Levels) ---")
for strategy_name, strategy_def in strategies.items():
    print(f"Calculating for: {strategy_name}")

    strategy_df = calculate_strategy_metrics(
        strategy_def,
        df_spot,df_cr,
        window_size_days_1m,
        window_size_days_3m,
        annualization_factor=np.sqrt(trading_days_per_year)
# Explicitly set 5y as benchmark
    )

    if not strategy_df.empty:
        strategy_results_dict[strategy_name] = strategy_df
    else:
        print(f"Could not calculate metrics for {strategy_name}. Check missing tenors or data issues.")

print("\n--- Results Dictionary (First 5 rows of each strategy's DataFrame) ---")
for strategy_name, df_result in strategy_results_dict.items():
    print(f"\nStrategy: {strategy_name}")
    print(df_result.head().to_string())

print("\nCalculation Complete!")
import pickle
pickle_file_path = os.path.join(market_data_folder, 'strategy_results1.pkl')

try:
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(strategy_results_dict, f)
    print(f"\nSuccessfully pickled strategy_results_dict to: {pickle_file_path}")
except Exception as e:
    print(f"\nError pickling strategy_results_dict: {e}")
