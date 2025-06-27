import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# Define the directory where the market data files are located
MARKET_DATA_DIR = 'market_data'

# Define the expected tenor labels for the columns (excluding the Date column)
EXPECTED_TENOR_LABELS = ['3m', '6m', '9m', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y', '12y', '15y',
                         '20y', '30y']

# Define the mapping from tenor labels to their numerical values in years
TENOR_LABEL_TO_YEARS = {
    '3m': 0.25, '6m': 0.5, '9m': 0.75,
    '1y': 1.0, '2y': 2.0, '3y': 3.0, '4y': 4.0, '5y': 5.0,
    '6y': 6.0, '7y': 7.0, '8y': 8.0, '9y': 9.0, '10y': 10.0,
    '12y': 12.0, '15y': 15.0, '20y': 20.0, '30y': 30.0
}

# Sort the target tenors by their numerical value for consistent interpolation
sorted_target_tenors = sorted(TENOR_LABEL_TO_YEARS.items(), key=lambda item: item[1])
target_tenor_labels_sorted = [label for label, _ in sorted_target_tenors]
target_tenor_values_sorted = [value for _, value in sorted_target_tenors]


# Function to safely interpolate (cubic with linear fallback if not enough points)
def safe_interp1d(x_data, y_data, x_target):
    """
    Performs cubic interpolation if enough data points are available (>= 4),
    otherwise falls back to linear interpolation (if >= 2 points),
    or returns NaN if insufficient data.
    """
    if len(x_data) >= 4:
        kind = 'cubic'
    elif len(x_data) >= 2:
        kind = 'linear'
    else:
        return np.full_like(x_target, np.nan)  # Not enough points for any interpolation

    f_interp = interp1d(x_data, y_data, kind=kind, fill_value='extrapolate')
    return f_interp(x_target)


# Load the data
try:
    # Load spot data, assuming 'Dates' and then the tenor labels as columns
    spot_df = pd.read_csv(os.path.join(MARKET_DATA_DIR, 'spot_new.csv'))
    # Load forward data, assuming 'Dates' and then the tenor labels as columns
    forward_df = pd.read_csv(os.path.join(MARKET_DATA_DIR, 'forward_new.csv'))
except FileNotFoundError as e:
    print(
        f"Error loading file: {e}. Please ensure 'spot_new.csv' and 'forward_new.csv' are in the '{MARKET_DATA_DIR}' directory.")
    exit()

# Rename 'Dates' column to 'Date' for consistency
spot_df.rename(columns={'Dates': 'Date'}, inplace=True)
forward_df.rename(columns={'Dates': 'Date'}, inplace=True)

# Convert 'Date' columns to datetime objects
spot_df['Date'] = pd.to_datetime(spot_df['Date'])
forward_df['Date'] = pd.to_datetime(forward_df['Date'])

# Set 'Date' as index for both dataframes
spot_df.set_index('Date', inplace=True)
forward_df.set_index('Date', inplace=True)

# Align dataframes on common dates
common_dates = spot_df.index.intersection(forward_df.index)
spot_df = spot_df.loc[common_dates]
forward_df = forward_df.loc[common_dates]

# Verify and set column names based on EXPECTED_TENOR_LABELS
# This assumes the order in the CSV matches EXPECTED_TENOR_LABELS after the Date column
if len(spot_df.columns) != len(EXPECTED_TENOR_LABELS) or \
        len(forward_df.columns) != len(EXPECTED_TENOR_LABELS):
    print("Error: Number of columns in CSVs (excluding Date) does not match the expected tenor labels.")
    print(f"Expected {len(EXPECTED_TENOR_LABELS)} tenor columns.")
    print(f"Spot DF has {len(spot_df.columns)} columns (excluding Date).")
    print(f"Forward DF has {len(forward_df.columns)} columns (excluding Date).")
    exit()

# Rename columns to the standardized tenor labels
spot_df.columns = EXPECTED_TENOR_LABELS
forward_df.columns = EXPECTED_TENOR_LABELS

# Prepare rates for interpolation: convert tenor labels to numerical years for internal use
forward_rates_numeric_cols = forward_df.rename(columns=TENOR_LABEL_TO_YEARS)
spot_rates_numeric_cols = spot_df.rename(columns=TENOR_LABEL_TO_YEARS)

# Initialize DataFrames to store interpolated curves
interpolated_forward_curve_df = pd.DataFrame(index=common_dates, columns=target_tenor_labels_sorted)
interpolated_spot_curve_df = pd.DataFrame(index=common_dates, columns=target_tenor_labels_sorted)

# Perform interpolation for each date for both forward and spot curves
for date in common_dates:
    # Forward Curve Interpolation
    available_fwd_tenors = forward_rates_numeric_cols.loc[date].dropna().index.values
    available_fwd_rates = forward_rates_numeric_cols.loc[date].dropna().values
    interpolated_fwd_rates = safe_interp1d(available_fwd_tenors, available_fwd_rates, target_tenor_values_sorted)
    interpolated_forward_curve_df.loc[date] = interpolated_fwd_rates

    # Spot Curve Interpolation
    available_spot_tenors = spot_rates_numeric_cols.loc[date].dropna().index.values
    available_spot_rates = spot_rates_numeric_cols.loc[date].dropna().values
    interpolated_spot_rates = safe_interp1d(available_spot_tenors, available_spot_rates, target_tenor_values_sorted)
    interpolated_spot_curve_df.loc[date] = interpolated_spot_rates

# --- Historical Carry Calculation (DataFrame) ---
# Carry for tenor T: S(T) - F(3m, T-3m)
# For '3m' tenor, carry is F(0, 3m) - S(0, 3m)
carry_df = pd.DataFrame(index=common_dates, columns=target_tenor_labels_sorted)

for date in common_dates:
    for tenor_label in target_tenor_labels_sorted:
        t_years = TENOR_LABEL_TO_YEARS[tenor_label]
        t_3m_years = TENOR_LABEL_TO_YEARS['3m']

        if tenor_label == '3m':
            # Standard carry for 3m: F(0, 3m) - S(0, 3m)
            if tenor_label in interpolated_forward_curve_df.columns and tenor_label in spot_df.columns:
                carry_df.loc[date, tenor_label] = (interpolated_forward_curve_df.loc[date, tenor_label] - spot_df.loc[
                    date, tenor_label])*-1
            else:
                carry_df.loc[date, tenor_label] = np.nan
        else:
            # Carry = S(T) - F(3m, T-3m)
            if tenor_label in spot_df.columns and '3m' in interpolated_forward_curve_df.columns and tenor_label in interpolated_forward_curve_df.columns:
                t_diff_years = t_years - t_3m_years
                if t_diff_years > 0:
                    F_0_T = interpolated_forward_curve_df.loc[date, tenor_label]
                    F_0_3m = interpolated_forward_curve_df.loc[date, '3m']

                    # Calculate F(3m, T-3m)
                    # Handle potential division by zero if t_diff_years is too small (shouldn't happen for T > 3m)
                    if t_diff_years != 0:
                        F_3m_T_minus_3m = (F_0_T * t_years - F_0_3m * t_3m_years) / t_diff_years
                        carry_df.loc[date, tenor_label] = spot_df.loc[date, tenor_label] - F_3m_T_minus_3m
                    else:
                        carry_df.loc[date, tenor_label] = np.nan
                else:
                    carry_df.loc[date, tenor_label] = np.nan  # Should not happen for T > 3m
            else:
                carry_df.loc[date, tenor_label] = np.nan

# --- Historical Rolldown Calculation (DataFrame) ---
# Rolldown for tenor T: S(T) - S(T - 3m)
rolldown_df = pd.DataFrame(index=common_dates, columns=target_tenor_labels_sorted)

for date in common_dates:
    for tenor_label in target_tenor_labels_sorted:
        t_years = TENOR_LABEL_TO_YEARS[tenor_label]
        t_3m_years = TENOR_LABEL_TO_YEARS['3m']

        if tenor_label == '3m':
            # For 3m, S(T-3m) would be S(0), which is not available. Set to NaN.
            rolldown_df.loc[date, tenor_label] = 0
        else:
            if tenor_label in spot_df.columns:
                target_spot_tenor_years = t_years - t_3m_years

                # Interpolate S(T-3m) from the interpolated_spot_curve_df
                # We need to use the numerical tenors for interpolation
                spot_curve_for_interp_x = interpolated_spot_curve_df.loc[date].index.map(TENOR_LABEL_TO_YEARS).values
                spot_curve_for_interp_y = interpolated_spot_curve_df.loc[date].values.astype(float)  # Ensure float type

                # Filter out NaN values for interpolation
                valid_indices = ~np.isnan(spot_curve_for_interp_y)
                spot_curve_for_interp_x = spot_curve_for_interp_x[valid_indices]
                spot_curve_for_interp_y = spot_curve_for_interp_y[valid_indices]

                if len(spot_curve_for_interp_x) >= 2:  # Need at least 2 points for linear, 4 for cubic
                    S_T_minus_3m = safe_interp1d(
                        spot_curve_for_interp_x,
                        spot_curve_for_interp_y,
                        np.array([target_spot_tenor_years])
                    )[0]  # Get the single interpolated value
                    rolldown_df.loc[date, tenor_label] = spot_df.loc[date, tenor_label] - S_T_minus_3m
                else:
                    rolldown_df.loc[date, tenor_label] = np.nan
            else:
                rolldown_df.loc[date, tenor_label] = np.nan


# --- Functions for single-date calculations ---

def calculate_3m_carry_on_date(date: str) -> float:
    """
    Calculates the 3-month carry for a given date from the pre-computed carry_df.
    """
    try:
        date_dt = pd.to_datetime(date)
        if date_dt not in carry_df.index:
            print(f"Warning: Date {date} not found in carry data. Cannot calculate 3m carry.")
            return np.nan
        return carry_df.loc[date_dt, '3m']
    except Exception as e:
        print(f"Error calculating 3m carry for date {date}: {e}")
        return np.nan


def calculate_3m_roll_on_date(date: str) -> float:
    """
    Calculates the 3-month rolldown for a given date from the pre-computed rolldown_df.
    """
    try:
        date_dt = pd.to_datetime(date)
        if date_dt not in rolldown_df.index:
            print(f"Warning: Date {date} not found in rolldown data. Cannot calculate 3m rolldown.")
            return np.nan
        return rolldown_df.loc[date_dt, '3m']
    except Exception as e:
        print(f"Error calculating 3m rolldown for date {date}: {e}")
        return np.nan


# Save the interpolated forward curve to a new CSV file
interpolated_forward_curve_df.to_csv('interpolated_forward_curve.csv')
print("Interpolated forward curve calculated and saved to 'interpolated_forward_curve.csv'.")

# Save the carry results to a new CSV file
carry_df.to_csv('historical_carry.csv')
print("Historical carry calculated and saved to 'historical_carry.csv'.")

# Save the rolldown results to a new CSV file
rolldown_df.to_csv('historical_rolldown.csv')
print("Historical rolldown calculated and saved to 'historical_rolldown.csv'.")

print("\nInterpolated Forward Curve (first 5 rows):")
print(interpolated_forward_curve_df.head())
print("\nHistorical Carry (first 5 rows):")
print(carry_df.head())
print("\nHistorical Rolldown (first 5 rows):")
print(rolldown_df.head())
carry_roll = 100*(carry_df + rolldown_df)
carry_roll.to_csv(r'market_data/df_cr_bps.csv')
# Example usage of the new functions:
# You can uncomment these lines to test the functions with a specific date
# example_date = common_dates[0] # Take the first common date for example
# print(f"\nExample 3m Carry for {example_date.strftime('%Y-%m-%d')}: {calculate_3m_carry_on_date(example_date)}")
# print(f"Example 3m Rolldown for {example_date.strftime('%Y-%m-%d')}: {calculate_3m_roll_on_date(example_date)}")
