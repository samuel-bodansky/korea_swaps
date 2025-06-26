import numpy as np


def discount_factors(tenors, spot_rates):
    """Convert spot rates to discount factors."""
    return {t: 1 / (1 + r) ** t for t, r in zip(tenors, spot_rates)}


def par_swap_rate(start, end, freq, dfs):
    """Compute par swap rate from discount factors."""
    times = np.arange(start + freq, end + freq / 2, freq)
    numerator = dfs[start] - dfs[end]
    denominator = sum(freq * dfs[t] for t in times)
    return numerator / denominator


def compute_swap_carry(tenors, spot_rates, forward_start=0.25, swap_length=10, freq=0.5):
    dfs = discount_factors(tenors, spot_rates)

    # Spot 10y swap: starts at 0, ends at 10
    spot_rate = par_swaap_rate(0.0, swap_length, freq, dfs)

    # Forward 10y swap: starts in 3m (0.25), ends at 10.25
    forward_rate = par_swap_rate(forward_start, forward_start + swap_length, freq, dfs)

    carry_bps = (forward_rate - spot_rate) * 10000  # in basis points
    carry_dollar = (forward_rate - spot_rate) * 1_000_000  # per $1M notional

    return {
        "spot_rate": spot_rate,
        "forward_rate": forward_rate,
        "carry_bps": carry_bps,
        "carry_dollar": carry_dollar
    }
