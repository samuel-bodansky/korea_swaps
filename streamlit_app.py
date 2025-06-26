import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta
import altair as alt  # Import Altair for more advanced plotting

# --- Configuration ---
market_data_folder = "market_data"
pickle_file_path = os.path.join(market_data_folder, 'strategy_results1.pkl')


# --- Load Data (cached for performance) ---
@st.cache_data
def load_strategy_results(path):
    """Loads the pickled strategy results dictionary."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Error: Pickle file not found at {path}. Please ensure the data generation script has been run to create it.")
        st.stop()  # Stop the app if data is not found
    except Exception as e:
        st.error(f"Error loading pickled data: {e}")
        st.stop()


loaded_strategy_results = load_strategy_results(pickle_file_path)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Strategy Threshold Monitor")

st.title("Korea Swaps Backtesting")
st.write("Adjust the thresholds and lookback period to find strategies exceeding your criteria.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Thresholds & Lookback")

# Thresholds using sliders
st.sidebar.subheader("Thresholds")
abs_beta_diff_threshold = st.sidebar.slider(
    "Absolute Beta Diff (1m-3m) Maximum Threshold",
    min_value=0.0, max_value=0.3, value=0.1, step=0.02, format="%.4f",
    help="Strategies where |1m Beta - 3m Beta| <= this value. (Lower values indicate more stable beta)."
)
vol_adjusted_carry_threshold = st.sidebar.slider(
    "Vol Adjusted Carry Minimum Level",
    min_value=0.0, max_value=0.05, value=0.005, step=0.0001, format="%.4f",
    help="Strategies where Vol Adjusted Carry >= this value. (Usually positive, in decimal format)."
)
z_score_threshold = st.sidebar.slider(
    "Z-score Minimum Threshold",
    min_value=-3.0, max_value=3.0, value=1.5, step=0.1, format="%.1f",
    help="Strategies where 3m Z-score >= this value. (Higher positive Z-scores indicate 'overbought' conditions)."
)

# Lookback Period
st.sidebar.subheader("Lookback Period")
lookback_options = {
    "1 Month": 20,  # Approx trading days
    "3 Months": 60,
    "6 Months": 120,
    "1 Year": 252,
    "All History": None  # To consider entire history
}
selected_lookback_str = st.sidebar.selectbox(
    "Select Lookback Period",
    options=list(lookback_options.keys()),
    index=1,  # Default to 3 Months
    help="Analyze strategies over the last N trading days."
)
lookback_days = lookback_options[selected_lookback_str]

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["Strategy Threshold Monitor", "Breach PnL Analysis", "Individual Strategy Yield Charts"])

# Initialize all_breaches_df outside tabs to be accessible globally for charts
all_breaches_df = pd.DataFrame()

with tab1:
    st.header("Strategies Exceeding Thresholds")

    results_list = []

    for strategy_name, df_metrics in loaded_strategy_results.items():
        if df_metrics.empty:
            continue

        # Filter by lookback period
        if lookback_days is not None:
            df_filtered = df_metrics.tail(lookback_days)
        else:
            df_filtered = df_metrics.copy()

        # Ensure required columns exist
        required_cols = ['Abs Beta Diff (1m-3m)', 'Vol Adjusted Carry', '3m Z-score']
        if not all(col in df_filtered.columns for col in required_cols):
            st.warning(
                f"Strategy '{strategy_name}' is missing one or more required columns ({required_cols}). Skipping for threshold check.")
            continue

        # Apply thresholds
        condition_beta_diff = (df_filtered['Abs Beta Diff (1m-3m)'].abs() <= abs_beta_diff_threshold)
        condition_vol_adjusted_carry = (df_filtered['Vol Adjusted Carry'] >= vol_adjusted_carry_threshold)
        condition_z_score = (df_filtered['3m Z-score'] >= z_score_threshold)

        # Combine conditions: True if ALL conditions are met
        breaching_dates = df_filtered[
            condition_beta_diff & condition_vol_adjusted_carry & condition_z_score].index.tolist()

        if breaching_dates:
            for b_date in breaching_dates:
                row_data = df_filtered.loc[b_date]
                results_list.append({
                    "Strategy": strategy_name,
                    "Date": b_date.strftime('%Y-%m-%d'),
                    "Abs Beta Diff (1m-3m)": row_data.get('Abs Beta Diff (1m-3m)'),
                    "Vol Adjusted Carry": row_data.get('Vol Adjusted Carry'),
                    "3m Z-score": row_data.get('3m Z-score'),
                    "Breaching Condition(s)": ", ".join([
                        "Abs Beta Diff" if row_data.get('Abs Beta Diff (1m-3m)', 0) <= abs_beta_diff_threshold else "",
                        "Vol Adjusted Carry" if row_data.get('Vol Adjusted Carry',
                                                             0) >= vol_adjusted_carry_threshold else "",
                        "Z-score" if row_data.get('3m Z-score', 0) >= z_score_threshold else ""
                    ]).strip(', ').replace(", , ", ", ")
                })

    if results_list:
        results_df = pd.DataFrame(results_list)

        # Sort by 3m Z-score descending and get top 10
        top_10_results = results_df.sort_values(by='3m Z-score', ascending=False).head(10)

        st.subheader("Top 10 Breaching Strategies (by Z-score, descending)")
        st.dataframe(top_10_results)

        if len(results_list) > 10:
            st.markdown(
                f"*(Note: {len(results_list) - 10} additional breaching events found, showing top 10 by Z-score)*")

        # --- Display Unique Strategies from Top 10 Table ---
        unique_strategies_found = top_10_results['Strategy'].unique().tolist()
        st.subheader("Unique Strategies Found in Top 10:")
        st.write(unique_strategies_found)

        # --- Find ALL Dates for These Strategies (1-Year Lookback) ---
        st.subheader("All Breach Dates for These Strategies (1-Year Lookback)")

        all_breaches_year_list = []
        year_lookback_days = 252  # Define 1 Year lookback

        if unique_strategies_found:
            for strategy_name in unique_strategies_found:
                df_metrics = loaded_strategy_results[strategy_name]

                # For breach calculation, consider a year.
                # Removed '[:-40]' to ensure all breaches within the lookback are captured for plotting.
                df_year_lookback_for_breach_check = df_metrics.tail(year_lookback_days)

                # Ensure required columns exist before applying filters
                if not all(col in df_year_lookback_for_breach_check.columns for col in required_cols):
                    continue

                # Re-apply the same conditions with current slider thresholds
                condition_beta_diff_year = (
                            df_year_lookback_for_breach_check['Abs Beta Diff (1m-3m)'].abs() <= abs_beta_diff_threshold)
                condition_vol_adjusted_carry_year = (
                            df_year_lookback_for_breach_check['Vol Adjusted Carry'] >= vol_adjusted_carry_threshold)
                condition_z_score_year = (df_year_lookback_for_breach_check['3m Z-score'] >= z_score_threshold)

                # Combine conditions (ALL must be met)
                breaching_dates_year = df_year_lookback_for_breach_check[
                    condition_beta_diff_year &
                    condition_vol_adjusted_carry_year &
                    condition_z_score_year
                    ].index.tolist()

                if breaching_dates_year:
                    for b_date_year in breaching_dates_year:
                        row_data_year = df_year_lookback_for_breach_check.loc[b_date_year]
                        all_breaches_year_list.append({
                            "Strategy": strategy_name,
                            "Date": b_date_year.strftime('%Y-%m-%d'),
                            "Abs Beta Diff (1m-3m)": row_data_year.get('Abs Beta Diff (1m-3m)'),
                            "Vol Adjusted Carry": row_data_year.get('Vol Adjusted Carry'),
                            "3m Z-score": row_data_year.get('3m Z-score'),
                            "Breaching Condition(s)": ", ".join([
                                "Abs Beta Diff" if row_data_year.get('Abs Beta Diff (1m-3m)',
                                                                     0) <= abs_beta_diff_threshold else "",
                                "Vol Adjusted Carry" if row_data_year.get('Vol Adjusted Carry',
                                                                          0) >= vol_adjusted_carry_threshold else "",
                                "Z-score" if row_data_year.get('3m Z-score', 0) >= z_score_threshold else ""
                            ]).strip(', ').replace(", , ", ", ")
                        })

            if all_breaches_year_list:
                all_breaches_df = pd.DataFrame(all_breaches_year_list)
                all_breaches_df['Date'] = pd.to_datetime(all_breaches_df['Date'])
                all_breaches_df = all_breaches_df.sort_values(by=['Date', 'Strategy']).reset_index(drop=True)
                all_breaches_df_display = all_breaches_df.copy()
                all_breaches_df_display['Date'] = all_breaches_df_display['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(all_breaches_df_display)
            else:
                st.info("No breaches found for these strategies over the last year with the current thresholds.")
        else:
            st.info("No unique strategies identified from the initial top 10 to check over the last year.")
    else:
        st.info("No strategies found exceeding ALL specified thresholds for the selected lookback period.")

with tab2:
    st.header("Breach PnL Analysis")

    if not all_breaches_df.empty:
        st.subheader("PnL Over Next 40 Trading Days After Breach")
        st.write(
            "Each line represents a specific breach event, showing the yield performance relative to its breach date over the subsequent 40 trading days. An increase means that yield of the fly has decreased.")

        include_cumulative_carry = st.checkbox("Include Cumulative Spot Carry (3m)", value=False,
                                               help="Adds (1/60) of 'Vol Adjusted Carry' from breach day cumulatively each day to the yield change.")

        combined_forward_yield_changes_df = pd.DataFrame(index=range(40))
        yield_column_name = 'Weighted Yield'
        carry_column_name = 'Weighted Carry'

        # List to store structured PnL and Yield results for the new table formats
        breach_analysis_results = []

        for index, breach_event in all_breaches_df.iterrows():
            strategy_name = breach_event['Strategy']
            breach_date = breach_event['Date']

            df_strategy_data = loaded_strategy_results.get(strategy_name)

            if df_strategy_data is None or df_strategy_data.empty:
                continue

            if yield_column_name not in df_strategy_data.columns:
                st.warning(
                    f"Strategy **{strategy_name}** does not have a '{yield_column_name}' column. Skipping forward yield change plot for breach on {breach_date.strftime('%Y-%m-%d')}.")
                continue

            current_line_include_carry = include_cumulative_carry
            if current_line_include_carry and carry_column_name not in df_strategy_data.columns:
                st.warning(
                    f"Strategy **{strategy_name}** does not have a '{carry_column_name}' column for cumulative carry. Skipping carry for breach on {breach_date.strftime('%Y-%m-%d')}.")
                current_line_include_carry = False

            df_strategy_data_sorted = df_strategy_data.sort_index()
            forward_slice = df_strategy_data_sorted[df_strategy_data_sorted.index >= breach_date].head(40)

            if forward_slice.empty or len(forward_slice) < 40:
                continue

            yield_at_breach = forward_slice.iloc[0][yield_column_name]
            forward_pnl_series = -1 * 100 * (forward_slice[yield_column_name] - yield_at_breach)

            if current_line_include_carry:
                spot_carry_on_breach_day = forward_slice.iloc[0][carry_column_name]
                daily_carry_addition = spot_carry_on_breach_day / 60.0
                cumulative_carry_series = pd.Series([
                    daily_carry_addition * (i + 1) for i in range(len(forward_pnl_series))
                ], index=forward_pnl_series.index)
                forward_pnl_series = forward_pnl_series + cumulative_carry_series

            line_label = f"{strategy_name} - {breach_date.strftime('%Y-%m-%d')}"
            temp_series_reindexed = pd.Series(forward_pnl_series.values, index=range(len(forward_pnl_series)))
            combined_forward_yield_changes_df[line_label] = temp_series_reindexed

            # Store results for the new table formats
            if not temp_series_reindexed.empty and len(forward_slice) == 40:  # Ensure we have 40 days for both
                breach_analysis_results.append({
                    "Strategy": strategy_name,
                    "Breach Date": breach_date,
                    "Final 40-Day PnL": temp_series_reindexed.iloc[-1],
                    "PnL_Series_40_Days": temp_series_reindexed,  # Store the full PnL series for the new table
                    "Raw_Yield_Series_40_Days": forward_slice[yield_column_name].values  # Store the raw yield values
                })

        if not combined_forward_yield_changes_df.empty:
            st.line_chart(combined_forward_yield_changes_df)
            st.markdown(
                f"*(Note: Chart shows PnL for {len(combined_forward_yield_changes_df.columns)} breach events)*")

            # --- New: Plot PnL Sum and Average (Requested) ---
            st.subheader("Aggregated PnL Progression Chart")
            if not combined_forward_yield_changes_df.empty:
                pnl_sum_series = combined_forward_yield_changes_df.sum(axis=1)
                pnl_avg_series = combined_forward_yield_changes_df.mean(axis=1)

                # Create a DataFrame for Altair plotting
                plot_data = pd.DataFrame({
                    'Day': pnl_sum_series.index,
                    'PnL Sum': pnl_sum_series.values,
                    'PnL Average': pnl_avg_series.values
                })

                # Chart for PnL Sum
                chart_sum = alt.Chart(plot_data).mark_line(color='green').encode(
                    x=alt.X('Day', title='Days After Breach (0-39)'),
                    y=alt.Y('PnL Sum', title='Aggregate PnL (Sum)'),
                    tooltip=['Day', alt.Tooltip('PnL Sum', format=".4f")]
                ).properties(
                    title='Aggregate PnL Over 40 Trading Days After Breach'
                )
                st.altair_chart(chart_sum, use_container_width=True)

                # Chart for PnL Average
                chart_avg = alt.Chart(plot_data).mark_line(color='purple').encode(
                    x=alt.X('Day', title='Days After Breach (0-39)'),
                    y=alt.Y('PnL Average', title='Average PnL'),
                    tooltip=['Day', alt.Tooltip('PnL Average', format=".4f")]
                ).properties(
                    title='Average PnL Over 40 Trading Days After Breach'
                )
                st.altair_chart(chart_avg, use_container_width=True)
            else:
                st.info("No sufficient data to plot aggregated and average PnL charts.")


            # --- Weighted Yield per Breach Event (40 Days Forward) Table ---
            st.subheader("Weighted Yield per Breach Event (40 Days Forward)")
            st.write(
                "Each row represents a specific breach event, showing the raw Weighted Yield values for the subsequent 40 days (Day 0 to Day 39).")

            if breach_analysis_results:
                yield_data_flat = []
                for item in breach_analysis_results:
                    row_dict = {
                        "Strategy": item["Strategy"],
                        "Breach Date": item["Breach Date"].strftime('%Y-%m-%d')
                    }
                    raw_yields = item["Raw_Yield_Series_40_Days"]
                    for i, yield_val in enumerate(raw_yields):
                        row_dict[f"Day {i}"] = yield_val
                    yield_data_flat.append(row_dict)

                if yield_data_flat:
                    yield_df_display = pd.DataFrame(yield_data_flat)
                    # Convert 'Day X' columns to numeric, coercing errors to NaN
                    day_columns = [col for col in yield_df_display.columns if col.startswith("Day ")]
                    for col in day_columns:
                        yield_df_display[col] = pd.to_numeric(yield_df_display[col], errors='coerce')

                    yield_df_display = yield_df_display.sort_values(by=['Strategy', 'Breach Date']).reset_index(
                        drop=True)
                    st.dataframe(yield_df_display.style.format({col: "{:.4f}" for col in day_columns}, na_rep='-'))
                else:
                    st.info("No raw weighted yield data available for table display.")
            else:
                st.info("No raw weighted yield data available for table display.")

            # --- New: Change in Yield (PnL) per Breach Event (40 Days Forward) Table ---
            st.subheader("Change in Yield (PnL) per Breach Event (40 Days Forward)")
            st.write(
                "Each row represents a specific breach event, showing the cumulative PnL (yield change, with optional carry) for the subsequent 40 days (Day 0 to Day 39).")

            if breach_analysis_results:
                pnl_progression_data_flat = []
                for item in breach_analysis_results:
                    row_dict = {
                        "Strategy": item["Strategy"],
                        "Breach Date": item["Breach Date"].strftime('%Y-%m-%d')
                    }
                    pnl_series = item["PnL_Series_40_Days"]
                    for i, pnl_val in enumerate(pnl_series.values):  # Use .values to iterate over the numpy array
                        row_dict[f"Day {i}"] = pnl_val
                    pnl_progression_data_flat.append(row_dict)

                if pnl_progression_data_flat:
                    pnl_progression_df_display = pd.DataFrame(pnl_progression_data_flat)
                    # Convert 'Day X' columns to numeric, coercing errors to NaN
                    day_columns = [col for col in pnl_progression_df_display.columns if col.startswith("Day ")]
                    for col in day_columns:
                        pnl_progression_df_display[col] = pd.to_numeric(pnl_progression_df_display[col],
                                                                        errors='coerce')

                    pnl_progression_df_display = pnl_progression_df_display.sort_values(
                        by=['Strategy', 'Breach Date']).reset_index(drop=True)
                    st.dataframe(
                        pnl_progression_df_display.style.format({col: "{:.4f}" for col in day_columns}, na_rep='-'))
                else:
                    st.info("No PnL progression data available for table display.")
            else:
                st.info("No PnL progression data available for table display.")

            # --- Aggregated and Average PnL Table (Requested) ---
            st.subheader("Aggregated and Average PnL Progression (40 Days Forward)")
            st.write("This table shows the aggregate and average PnL across all selected breach events for each of the subsequent 40 trading days.")

            if breach_analysis_results:
                # Create a DataFrame where each row is a breach event's PnL series
                all_pnl_series_df = pd.DataFrame()
                for item in breach_analysis_results:
                    # Append each PnL_Series_40_Days as a row, naming columns Day 0 to Day 39
                    temp_series = item["PnL_Series_40_Days"].rename(lambda x: f"Day {x}")
                    # Use pd.Series to create a row, then pd.DataFrame to stack them
                    all_pnl_series_df = pd.concat([all_pnl_series_df, pd.DataFrame([temp_series])], ignore_index=True)


                if not all_pnl_series_df.empty:
                    # Convert columns to numeric if not already
                    day_columns = [col for col in all_pnl_series_df.columns if col.startswith("Day ")]
                    for col in day_columns:
                        all_pnl_series_df[col] = pd.to_numeric(all_pnl_series_df[col], errors='coerce')

                    aggregate_pnl = all_pnl_series_df[day_columns].sum()
                    average_pnl = all_pnl_series_df[day_columns].mean()

                    summary_pnl_data = {
                        "Metric": ["Aggregate PnL", "Average PnL"],
                    }
                    for i in range(40):
                        col_name = f"Day {i}"
                        summary_pnl_data[col_name] = [
                            aggregate_pnl.get(col_name, np.nan), # Use .get() with default to handle missing columns gracefully
                            average_pnl.get(col_name, np.nan)
                        ]

                    summary_pnl_df = pd.DataFrame(summary_pnl_data)
                    summary_pnl_df = summary_pnl_df.set_index("Metric")
                    st.dataframe(summary_pnl_df.style.format("{:.4f}", na_rep='-'))
                else:
                    st.info("No PnL progression data available to calculate aggregate and average PnL.")
            else:
                st.info("No PnL progression data available to calculate aggregate and average PnL.")


            # --- Aggregated Metrics Table (Original) ---
            st.subheader("Performance Metrics (Aggregated PnL)")
            st.write("Metrics calculated based on the average PnL across all plotted breaching events.")

            aggregated_pnl_series = combined_forward_yield_changes_df.mean(axis=1)
            daily_pnl_aggregated = aggregated_pnl_series.diff()
            daily_pnl_aggregated.iloc[0] = aggregated_pnl_series.iloc[0]

            total_pnl_sum = sum(
                [item['Final 40-Day PnL'] for item in breach_analysis_results]) if breach_analysis_results else 0.0

            sharpe_ratio = np.nan
            sortino_ratio = np.nan

            if not daily_pnl_aggregated.empty and daily_pnl_aggregated.std() > 0:
                avg_daily_pnl = daily_pnl_aggregated.mean()
                std_dev_daily_pnl = daily_pnl_aggregated.std()
                sharpe_ratio = (avg_daily_pnl / std_dev_daily_pnl) * np.sqrt(252)

                downside_returns = daily_pnl_aggregated[daily_pnl_aggregated < 0]
                if not downside_returns.empty and downside_returns.std() > 0:
                    downside_deviation = downside_returns.std()
                    sortino_ratio = (avg_daily_pnl / downside_deviation) * np.sqrt(252)
                else:
                    sortino_ratio = np.nan
            else:
                sharpe_ratio = np.nan
                sortino_ratio = np.nan

            metrics_data = {
                "Metric": ["Total PnL (Sum of Final PnLs)", "Sharpe Ratio (Annualized)", "Sortino Ratio (Annualized)"],
                "Value": [
                    f"{total_pnl_sum:.2f}",
                    f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A",
                    f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "N/A"
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data).set_index("Metric"))

        else:
            st.info(
                "No sufficient data found to plot 40-day forward yield changes for any breaching events with the current filters.")
    else:
        st.info(
            "Please adjust thresholds and lookback in 'Strategy Threshold Monitor' tab to find breaching strategies first.")

with tab3:
    st.header("Individual Strategy Absolute Yield Charts")
    st.write(
        "Visualize the absolute yield of selected strategies over time, starting from their earliest available data point. Circles indicate breach dates.")

    all_available_strategies = list(loaded_strategy_results.keys())

    strategies_to_plot = st.multiselect(
        "Select Strategies to Plot Absolute Yield",
        all_available_strategies,
        default=all_breaches_df['Strategy'].unique().tolist() if not all_breaches_df.empty else []
    )

    if strategies_to_plot:
        for strategy_name in strategies_to_plot:
            if strategy_name in loaded_strategy_results:
                df_strategy_data = loaded_strategy_results[strategy_name]

                if df_strategy_data.empty:
                    st.warning(f"No data available for strategy: **{strategy_name}**.")
                    continue

                yield_column_name = 'Weighted Yield'
                if yield_column_name not in df_strategy_data.columns:
                    st.warning(
                        f"Strategy **{strategy_name}** does not have a '{yield_column_name}' column. Skipping individual chart.")
                    continue

                plot_start_date = df_strategy_data.index.min()
                df_for_plot = df_strategy_data[df_strategy_data.index >= plot_start_date].copy()

                if not df_for_plot.empty:
                    st.subheader(
                        f"Absolute Yield for: **{strategy_name}** (from {plot_start_date.strftime('%Y-%m-%d')})")

                    chart_data = df_for_plot.reset_index(names=['Date'])
                    chart_data['Date'] = pd.to_datetime(chart_data['Date'])

                    breaches_for_this_strategy = all_breaches_df[all_breaches_df['Strategy'] == strategy_name].copy()
                    breaches_for_this_strategy = breaches_for_this_strategy[
                        (breaches_for_this_strategy['Date'] >= plot_start_date) &
                        (breaches_for_this_strategy['Date'] <= df_for_plot.index.max())
                        ]

                    base = alt.Chart(chart_data).encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y(f'{yield_column_name}:Q', title='Weighted Yield')
                    ).properties(
                        title=f"Absolute Yield for {strategy_name}"
                    )

                    line = base.mark_line().encode(
                        color=alt.value('steelblue')
                    )

                    if not breaches_for_this_strategy.empty:
                        breach_points_data = pd.merge(
                            breaches_for_this_strategy,
                            chart_data[['Date', yield_column_name]],
                            on='Date',
                            how='left'
                        ).dropna(subset=[yield_column_name])

                        if not breach_points_data.empty:
                            circles = alt.Chart(breach_points_data).mark_circle(size=80, color='red').encode(
                                x='Date:T',
                                y=f'{yield_column_name}:Q',
                                tooltip=[alt.Tooltip('Date:T', title='Breach Date'),
                                         alt.Tooltip(f'{yield_column_name}:Q', title='Yield at Breach', format='.4f'),
                                         alt.Tooltip('Breaching Condition(s)', title='Conditions')]
                            )
                            chart = line + circles
                        else:
                            chart = line
                            st.warning(
                                f"No matching yield data for breach dates for strategy: **{strategy_name}** to plot circles.")
                    else:
                        chart = line

                    st.altair_chart(chart, use_container_width=True)

                else:
                    st.info(
                        f"No absolute yield data available for **{strategy_name}** after its determined start date ({plot_start_date.strftime('%Y-%m-%d')}).")
            else:
                st.error(f"Strategy **{strategy_name}** not found in loaded data.")
    else:
        st.info("Please select one or more strategies to display their absolute yield charts.")

st.markdown("---")
st.markdown("Metrics Definitions:")
st.markdown("- **Weighted Yield**: The weighted sum of spot rates for the strategy's tenors.")
st.markdown(
    "- **3m Z-score**: How many standard deviations the current Weighted Yield is from its 3-month rolling mean. (Positive values indicate 'overbought', negative 'oversold').")
st.markdown("- **Vol**: Annualized 3-month rolling volatility of the strategy's daily yield changes.")
st.markdown(
    "- **Vol Adjusted Carry**: Annualized 3-month rolling volatility of the strategy's daily total return (daily yield change + weighted carry/rolldown).")
st.markdown(
    "- **1m/3m Correlation to 5y**: Rolling correlation of the strategy's outright weighted yield to the outright 5-year spot rate.")
st.markdown(
    "- **1m/3m Beta to 5y**: Rolling beta of the strategy's daily yield changes to the 5-year daily spot rate changes.")
st.markdown(
    "- **Abs Beta Diff (1m-3m)**:  Absolute difference between 1-month and 3-month rolling betas to 5y. Lower values indicate more stable beta.")