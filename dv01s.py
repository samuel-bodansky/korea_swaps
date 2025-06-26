import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

# Reload the data and re-initialize variables from the previous steps
# This ensures the script is self-contained for execution
try:
    df = pd.read_csv('Book1.csv', header=2,encoding='latin1')
    df.rename(columns={'MTY_YEARS_TDY': 'Category'}, inplace=True)

    # Extract the row containing 'SW_CNV_RISK' for spot dv01
    spot_dv01_row = df[df['Category'] == 'SW_CNV_RISK'].iloc[0]

    # Extract the tenor labels (rough maturities) from the column names, excluding the first two descriptive columns
    tenors_str = df.columns[2:].tolist()
    headers = ['6m','9m','1y','2y','3y','4y','5y','6y','7y','8y','9y','10y','12y','15y','20y','30y']
    # Convert spot_dv01 values to numeric, mapping them to the tenors
    spot_dv01 = pd.Series(pd.to_numeric(spot_dv01_row[tenors_str]), index=tenors_str)

    # Extract historical spot prices: these are the rows where 'Category' is 'Dates' or actual dates
    # Assuming 'Dates' row itself is a header for dates column
    historical_prices_start_idx = df[df['Category'] == 'Dates'].index[0] + 1
    historical_prices_df = df.iloc[historical_prices_start_idx:].copy()

    # Rename the 'Category' column to 'Dates' in the historical_prices_df
    historical_prices_df.rename(columns={'Category': 'Dates'}, inplace=True)

    # Set the 'Dates' column as the index for historical_prices_df
    historical_prices_df.set_index('Dates', inplace=True)
    # Convert all tenor columns in historical_prices_df to numeric
    for col in tenors_str:
        historical_prices_df[col] = pd.to_numeric(historical_prices_df[col], errors='coerce')

    # Drop the first column as it contains metadata and is not part of the tenor data
    # The second column was already removed by slicing df.columns[2:] for tenors_str
    historical_prices_df = historical_prices_df.iloc[:, 1:]
    historical_prices_df.columns = headers
    three_month=pd.DataFrame(df.iloc[4:,[0,1]]).set_index('Category')
    three_month.columns = ['3m']
    historical_prices_df.insert(0,'3m',three_month['3m'])

except Exception as e:
    print(f"Error loading or preparing data: {e}")
    # Exit or handle the error appropriately if data loading fails
    exit()
notional = 1e8
tenors_float = [float(x) for x in tenors_str]
# Annualized 3M forward rate starting 3M from today
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.interpolate import interp1d
df_spot = historical_prices_df
# 1. Define tenors and their numerical representation
tenors_str = ['3m', '6m', '9m', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y', '12y', '15y', '20y', '30y']


# Helper function to convert tenor string to years (returns float)
def tenor_to_years(tenor_str_val):
    if 'm' in tenor_str_val:
        return float(tenor_str_val.replace('m', '')) / 12
    elif 'y' in tenor_str_val:
        return float(tenor_str_val.replace('y', ''))
    return np.nan  # Should not be reached with our defined tenors


# Convert tenor strings to numerical years once for all calculations (all will be floats)
numerical_tenors = np.array([tenor_to_years(t) for t in tenors_str], dtype=float)


# 3. Function to calculate the 3-month forward curve for a single set of spot rates
def calculate_3m_forward_curve_for_day(spot_rates_daily_series, numerical_tenors_arr, target_forward_period_years=0.25):
    """
    Calculates the 3-month forward rate curve for a given set of daily spot rates.

    Args:
        spot_rates_daily_series (pd.Series): A Series of spot rates for the current day, indexed by tenor strings.
        numerical_tenors_arr (np.array): A NumPy array of numerical tenor years corresponding to spot_rates_daily_series.
        target_forward_period_years (float): The length of the forward period in years (e.g., 0.25 for 3 months).

    Returns:
        pd.Series: A Series of 3-month forward rates for the current day, indexed by tenor strings, all floats.
    """
    spot_rates_values = spot_rates_daily_series.values.astype(float)

    # Filter out NaN values for interpolation
    valid_mask = ~np.isnan(spot_rates_values)
    valid_numerical_tenors = numerical_tenors_arr[valid_mask]
    valid_spot_rates = spot_rates_values[valid_mask]

    if len(valid_numerical_tenors) < 2:
        return pd.Series(np.nan, index=spot_rates_daily_series.index, dtype=float)

    # Determine interpolation kind: cubic requires at least 4 points, otherwise fall back to linear
    if len(valid_numerical_tenors) >= 4:
        interp_kind = 'cubic'
    else:
        interp_kind = 'linear'

    # Create an interpolation function for this day's spot rates
    # 'fill_value='extrapolate'' allows calculation beyond the min/max numerical_tenors_arr
    # WARNING: Cubic spline extrapolation can produce very aggressive or unrealistic values
    # when extending far beyond the observed data range.
    interp_func = interp1d(valid_numerical_tenors, valid_spot_rates, kind=interp_kind, fill_value='extrapolate', bounds_error=False)

    forward_rates_daily_list = []

    # Iterate through each original tenor point (t1)
    for i in range(len(numerical_tenors_arr)):
        t1 = numerical_tenors_arr[i]
        if t1==np.float64(20.0):
            print(1)
        R_t1 = spot_rates_values[i]  # Get the exact spot rate for t1 from the input data (already float)

        # If R_t1 is NaN, skip calculation for this tenor
        if pd.isna(R_t1):
            forward_rates_daily_list.append(np.nan)
            continue

        t2 = t1 + target_forward_period_years  # The future point we are looking for (float)

        # Get R_t2 using the interpolation/extrapolation function (will be float)
        R_t2 = interp_func(t2).item()  # .item() converts 0-D array to scalar float

        # Calculate the forward rate using the discrete compounding formula:
        # F(t1, t2) = [((1 + R_t2)^t2 / (1 + R_t1)^t1)^(1/(t2-t1))] - 1
        # This formula assumes ANNUAL COMPOUNDING for the spot rates.
        # If your spot rates are compounded differently (e.g., semi-annually, simple),
        # a different formula should be used.

        # Ensure bases for exponentiation are positive and period is positive
        if (1 + R_t1) <= 0 or (1 + R_t2) <= 0 or (t2 - t1) <= 0:
            forward_rate = np.nan  # Result will be NaN (float)
        else:
            base = ((1 + R_t2) ** t2) / ((1 + R_t1) ** t1)
            if base > 0:
                forward_rate = base ** (1 / (t2 - t1)) - 1
            else:
                forward_rate = np.nan # Cannot compute real forward rate if base is non-positive

        forward_rates_daily_list.append(forward_rate)

    # Return as a pandas Series, preserving the original tenor string index, ensuring float dtype
    return pd.Series(forward_rates_daily_list, index=spot_rates_daily_series.index, dtype=float)



def calculate_3m_rolldown_curve_for_day(spot_rates_daily_series: pd.Series,
                                        numerical_tenors_arr: np.ndarray) -> pd.Series:
    """
    Calculates the 3-month roll-down for a single day's spot rate curve.

    Args:
        spot_rates_daily_series (pd.Series): A Series of spot rates for the current day,
                                            indexed by tenor strings.
        numerical_tenors_arr (np.ndarray): A NumPy array of numerical tenor years
                                            corresponding to spot_rates_daily_series.

    Returns:
        pd.Series: A Series of 3-month roll-down values (in decimal form)
                   for the current day, indexed by tenor strings.
    """
    spot_rates_values = spot_rates_daily_series.values.astype(float)

    # Create an interpolation function for the current day's spot curve
    interp_func = interp1d(numerical_tenors_arr, spot_rates_values, kind='linear', fill_value='extrapolate')

    rolldown_rates_daily_list = []
    rolldown_period_years = 0.25  # 3 months

    # Calculate roll-down for each tenor
    for i in range(len(numerical_tenors_arr)):
        current_tenor_years = numerical_tenors_arr[i]
        current_spot_rate = spot_rates_values[i]

        rolled_down_tenor_years = current_tenor_years - rolldown_period_years

        # If rolling down leads to a maturity <= 0 (e.g., 3m rolls to 0m),
        # roll-down is not typically defined on the curve.
        if rolled_down_tenor_years <= 0:
            rolled_down_spot_rate = np.nan  # Mark as NaN
        else:
            # Interpolate the spot rate at the new, shorter maturity
            rolled_down_spot_rate = interp_func(rolled_down_tenor_years).item()

        if pd.isna(rolled_down_spot_rate):
            rolldown_value = np.nan
        else:
            # Roll-down = Current Spot Rate - Spot Rate at the shorter (rolled-down) maturity
            rolldown_value = current_spot_rate - rolled_down_spot_rate
            rolldown_value = float(rolldown_value)

        rolldown_rates_daily_list.append(rolldown_value)

    # Return as a pandas Series, preserving the original tenor string index, ensuring float dtype
    return pd.Series(rolldown_rates_daily_list, index=spot_rates_daily_series.index, dtype=float)


# 4. Initialize an empty DataFrame for forward rates with the same structure as df_spot
# Ensures the DataFrame is pre-allocated with float NaN values
df_forward = pd.DataFrame(index=df_spot.index, columns=df_spot.columns, dtype=float)
df_rolldown_bps = pd.DataFrame(index=df_spot.index, columns=df_spot.columns, dtype=float)
# Apply the calculation function to each row (date) of the spot rates DataFrame
for date_index, row in df_spot.iterrows():
    # .loc ensures assignment to the correct row, and the returned Series directly fits the columns
    df_forward.loc[date_index] = calculate_3m_forward_curve_for_day(row, numerical_tenors)
    df_rolldown_bps.loc[date_index] = calculate_3m_rolldown_curve_for_day(row, numerical_tenors)*100
df_rolldown_bps['3m']=0
print("--- Output 3-Month Forward Rates DataFrame ---")
print(df_forward.to_string())
print(df_rolldown_bps.to_string())

df_carry_bps = (df_spot.astype('float') - df_forward) * 100# .to_string() for full, un-truncated view

df_carry_rolldown = df_carry_bps.add(df_rolldown_bps)
print(1)