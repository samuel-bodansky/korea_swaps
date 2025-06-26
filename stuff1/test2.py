import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

# Define the notional for carry calculation
NOTIONAL = 100_000_000  # 100 Million

# Reload the data and re-initialize variables from the previous steps
# This ensures the script is self-contained for execution
try:
    # Use latin1 encoding for reading the CSV file
    df = pd.read_csv('../Book1.csv', header=2, encoding='latin1')
    df.rename(columns={'MTY_YEARS_TDY': 'Category'}, inplace=True)

    # Extract the row containing 'SW_CNV_RISK' for spot dv01
    spot_dv01_row = df[df['Category'] == 'SW_CNV_RISK'].iloc[0]

    # Extract the original tenor labels (rough maturities) from the column names
    original_tenors_str = df.columns[2:].tolist()

    # Convert original tenor strings to numeric, then round to 2 significant figures, and convert back to string
    rounded_tenors_str = []
    tenor_rename_map = {}
    for t_str in original_tenors_str:
        t_float = float(t_str)
        # Round to 2 significant figures
        rounded_t_float = float(f'{t_float:.2g}')
        rounded_t_str = str(rounded_t_float)
        rounded_tenors_str.append(rounded_t_str)
        tenor_rename_map[t_str] = rounded_t_str

    # Update tenors_str to use the rounded names
    tenors_str = rounded_tenors_str

    # Convert spot_dv01 values to numeric, mapping them to the NEW (rounded) tenors
    spot_dv01 = pd.Series(pd.to_numeric(spot_dv01_row[original_tenors_str]), index=rounded_tenors_str)

    # Extract historical spot prices: these are the rows where 'Category' is 'Dates' or actual dates
    historical_prices_start_idx = df[df['Category'] == 'Dates'].index[0] + 1
    historical_prices_df = df.iloc[historical_prices_start_idx:].copy()

    # Rename the 'Category' column to 'Dates' in the historical_prices_df
    historical_prices_df.rename(columns={'Category': 'Dates'}, inplace=True)

    # Set the 'Dates' column as the index for historical_prices_df
    historical_prices_df.set_index('Dates', inplace=True)

    # Rename the columns of historical_prices_df using the rounded tenor names
    historical_prices_df.rename(columns=tenor_rename_map, inplace=True)

    # Convert all tenor columns in historical_prices_df to numeric (now using the rounded names)
    for col in tenors_str:
        historical_prices_df[col] = pd.to_numeric(historical_prices_df[col], errors='coerce')

    # Drop the first column as it contains metadata and is not part of the tenor data
    historical_prices_df = historical_prices_df.iloc[:, 1:]

except Exception as e:
    print(f"Error loading or preparing data: {e}")
    exit()

# --- Identify the 3-month tenor for spot series extraction ---
# 3 months is approximately 0.25 years
target_tenor_years_3m = 3 / 12.0  # 0.25 years

# Find the closest tenor in the available data to represent the 3-month point
numeric_tenors_available = np.array([float(t) for t in tenors_str])
closest_3m_tenor_idx = (np.abs(numeric_tenors_available - target_tenor_years_3m)).argmin()
closest_3m_tenor_str = tenors_str[closest_3m_tenor_idx]
print(f"\nIdentified tenor closest to 3 months for spot series: {closest_3m_tenor_str} years")

# --- Get the 3-month spot series ---
# This is the historical time series of spot rates for the tenor closest to 3 months.
spot_series_3m = historical_prices_df[closest_3m_tenor_str]
print(f"\n3-Month Spot Rate Series (for {closest_3m_tenor_str}yr tenor - first 5 rows):")
print(spot_series_3m.head())

# Save the 3-month spot series to a CSV
spot_series_3m_filename = f'3m_spot_series_{closest_3m_tenor_str}_tenor.csv'
spot_series_3m.to_csv(spot_series_3m_filename, header=[f'Spot Rate ({closest_3m_tenor_str}yr Tenor)'])
print(f"\n3-month spot series saved to '{spot_series_3m_filename}'")

# --- Calculate Spot Rates, 3-Month Forward Rates, Spread, and Carry for each tenor ---

# Initialize a dictionary to store data for the combined DataFrame
combined_data = {}

# Time period for the forward rate (3 months in years)
forward_period_years = 3 / 12.0  # 0.25 years

# Convert tenor column names to numeric for calculations
numeric_tenors_cols = np.array([float(t) for t in historical_prices_df.columns])

# Iterate through each date in the historical prices
for date_idx, current_date_rates_series in historical_prices_df.iterrows():
    # Filter out NaN values from current_date_rates_series and corresponding numeric_tenors
    valid_rates = current_date_rates_series.dropna()
    valid_tenors = np.array([float(col) for col in valid_rates.index])

    # Determine interpolation kind: cubic requires at least 4 points, otherwise fall back to linear
    if len(valid_tenors) >= 4:
        interp_kind = 'cubic'
    elif len(valid_tenors) >= 2:
        interp_kind = 'linear'
    else:
        # Not enough points for any interpolation on this date, fill with NaN for all outputs
        for tenor_str in tenors_str:
            prefix = f"{tenor_str}yr_"
            combined_data.setdefault(prefix + 'SpotRate', []).append(np.nan)
            combined_data.setdefault(prefix + 'ForwardRate', []).append(np.nan)
            combined_data.setdefault(prefix + 'Spread', []).append(np.nan)
            combined_data.setdefault(prefix + 'Carry', []).append(np.nan)
            combined_data.setdefault(prefix + 'Carry_bps', []).append(np.nan)  # Added for Carry in BPS
        continue

    # Create an interpolation function for the current date's curve
    # Changed fill_value to 'extrapolate' to allow calculation beyond the last tenor point.
    # Be aware that extrapolation, especially with 'cubic' kind, can produce unreliable results
    # when extending far beyond the observed data range.
    interp_func = interp1d(valid_tenors, valid_rates, kind=interp_kind, bounds_error=False, fill_value='extrapolate')

    # Iterate through each tenor to calculate spot, forward, spread, and carry
    for tenor_str in tenors_str:
        T1 = float(tenor_str)
        S1 = current_date_rates_series[tenor_str]  # Spot Rate at T1

        # Initialize values for this tenor and date
        forward_rate = np.nan
        spread = np.nan
        carry = np.nan
        carry_bps = np.nan  # Initialize carry in BPS

        # If S1 is NaN, we cannot calculate forward rate, spread, or carry for this point
        if not pd.isna(S1):
            T2 = T1 + forward_period_years
            S2 = interp_func(T2)  # Spot Rate at T2 (interpolated or extrapolated)

            # Check for valid S2 and positive time difference
            if not np.isnan(S2) and (T2 - T1) > 0:
                # Check for valid denominators to avoid division by zero or complex numbers
                if (1 + S1 * T1) > 0:
                    # Formula for 3-month forward rate starting at T1:
                    # F = ((1 + S2*T2) / (1 + S1*T1)) ** (1 / (T2 - T1)) - 1
                    # Note: If S1*T1 or S2*T2 results in a negative value such that (1 + S*T) is negative,
                    # the power calculation might yield complex numbers.
                    # We ensure the base of the power is positive.
                    base = (1 + S2 * T2) / (1 + S1 * T1)
                    if base > 0:
                        forward_rate = (base) ** (1 / (T2 - T1)) - 1
                    else:
                        forward_rate = np.nan  # Cannot compute real forward rate if base is non-positive
                else:
                    forward_rate = np.nan  # Denominator (1 + S1*T1) is zero or negative
            else:
                forward_rate = np.nan  # S2 is NaN or time difference is not positive

            # Calculate Spread: Spot Rate at T1 - Forward Rate
            spread = S1 - forward_rate

            # Calculate Carry: Spread * Notional
            carry = spread * NOTIONAL

            # Calculate Carry in Basis Points (bps) as a percentage of notional
            # 1 basis point (bp) = 0.0001
            # Spread is already a decimal rate (e.g., 0.0010 for 10 bps)
            carry_bps = spread * 10000  # Convert the decimal spread directly to bps

        # Store the calculated values for the current tenor and date
        prefix = f"{tenor_str}yr_"
        combined_data.setdefault(prefix + 'SpotRate', []).append(S1)
        combined_data.setdefault(prefix + 'ForwardRate', []).append(forward_rate)
        combined_data.setdefault(prefix + 'Spread', []).append(spread)
        combined_data.setdefault(prefix + 'Carry', []).append(carry)
        combined_data.setdefault(prefix + 'Carry_bps', []).append(carry_bps)  # Store Carry in BPS

# Create the final DataFrame from the combined data
# Set the index explicitly using historical_prices_df.index to ensure correct date alignment
carry_analysis_df = pd.DataFrame(combined_data, index=historical_prices_df.index)

print("\nCarry Analysis (Spot Rate, 3-Month Forward Rate, Spread, Carry, Carry in BPS) - First 5 Rows:")
print(carry_analysis_df.head())

# Save the combined results to a CSV file
output_analysis_filename = '../carry_analysis_per_tenor.csv'
carry_analysis_df.to_csv(output_analysis_filename)
print(f"\nDetailed carry analysis saved to '{output_analysis_filename}'")
