import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.interpolate import interp1d
import os
from market_data.strategies_list import curve_and_butterfly_strategies

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
df_spot = pd.read_csv(os.path.join(market_data_folder, "spot_new.csv"), index_col='Dates',
                                  parse_dates=True).sort_index()
df_cr = pd.read_csv(os.path.join(market_data_folder, "df_cr_bps.csv"), index_col='Date',
                                   parse_dates=True).sort_index()
df_spot.columns = df_cr.columns
df_usd_krw= pd.read_csv(os.path.join(market_data_folder, "dollar_won_rate.csv"), index_col='Dates',
                                   parse_dates=True).sort_index()
# Keys are strategy names, values are dictionaries with 'tenors' (numerical) and 'weights'
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
    weighted_yield = df_spot_hist[strategy_tenors].dot(strategy_weights)*100
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
    # vol_adjusted_carry = weighted_carry/(weighted_dv01*vol)
    vol_adjusted_carry = weighted_carry/(vol)
    if max(abs(vol_adjusted_carry.dropna()))>2:
        print(1)
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

    '''
    let's calculate dollar won rolling residual 3m
    '''
    print(1)

    combined_data_for_residual = pd.DataFrame({
        'Yield_3m_Spot': weighted_yield,
        'FX_Rate': df_usd_krw['usdkrw']
    }).dropna()
    common_dates = df_usd_krw.index.intersection(weighted_yield.index)

    residual_weighted_yield_df = pd.DataFrame(index=common_dates, columns=['3m Rolling Residual Weighted Yield'])

    if not combined_data_for_residual.empty and len(combined_data_for_residual) >= window_size_days_3m:
        # Calculate rolling means
        rolling_mean_yield = combined_data_for_residual['Yield_3m_Spot'].rolling(window=window_size_days_3m).mean()
        rolling_mean_fx = combined_data_for_residual['FX_Rate'].rolling(window=window_size_days_3m).mean()

        # Calculate rolling covariance and variance
        rolling_cov = combined_data_for_residual['Yield_3m_Spot'].rolling(window=window_size_days_3m).cov(
            combined_data_for_residual['FX_Rate'])
        rolling_var_fx = combined_data_for_residual['FX_Rate'].rolling(window=window_size_days_3m).var()

        # Calculate rolling beta (slope)
        # Handle division by zero for rolling_var_fx
        rolling_beta = rolling_cov / rolling_var_fx
        rolling_beta.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN

        # Calculate rolling alpha (intercept)
        rolling_alpha = rolling_mean_yield - rolling_beta * rolling_mean_fx

        # Calculate rolling predicted yield
        # Ensure alignment of indices for multiplication
        rolling_predicted_yield = rolling_alpha.reindex(combined_data_for_residual.index) + \
                                  rolling_beta.reindex(combined_data_for_residual.index) * combined_data_for_residual[
                                      'FX_Rate']

        # Calculate rolling residuals
        rolling_residuals = combined_data_for_residual['Yield_3m_Spot'] - rolling_predicted_yield

        # Calculate 3m rolling mean of residuals
        rolling_residual_weighted_yield = rolling_residuals.rolling(window=window_size_days_3m).mean()

        # Assign the calculated series to the DataFrame, ensuring index alignment
        residual_weighted_yield_df['3m Rolling Residual Weighted Yield'] = rolling_residual_weighted_yield.reindex(
            common_dates)
    else:
        print("Warning: Not enough data for 3m Rolling Residual Weighted Yield calculation. Result will be NaN.")
        residual_weighted_yield_df['3m Rolling Residual Weighted Yield'] = np.nan
    res_series = residual_weighted_yield_df['3m Rolling Residual Weighted Yield']

    # Save the 3m rolling residual weighted yield results to a new CSV file
    # --- Combine all results into a DataFrame ---
    strategy_df = pd.concat([weighted_yield, z_score_3m, vol, vol_adjusted_carry, corr_1m, corr_3m,abs_beta_diff,weighted_carry,res_series], axis=1)

    # --- Combine results into a DataFrame ---
    return strategy_df


# --- 5. Main Loop and Dictionary Creation ---
strategy_results_dict = {}
strategies = curve_and_butterfly_strategies
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
    print(df_result.tail().to_string())

print("\nCalculation Complete!")
import pickle
pickle_file_path = os.path.join(market_data_folder, 'strategy_results6.pkl')

try:
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(strategy_results_dict, f)
    print(f"\nSuccessfully pickled strategy_results_dict to: {pickle_file_path}")
except Exception as e:
    print(f"\nError pickling strategy_results_dict: {e}")
