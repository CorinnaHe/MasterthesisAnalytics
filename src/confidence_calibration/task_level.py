import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Confidence Normalization
# --------------------------------------------------

def _normalize_confidence(df, confidence_col, method="scale_0_1"):
    """
    Normalize confidence values.

    method:
        - "scale_0_1"  -> min-max scaling to [0,1]
        - "divide_by_max" -> divide by max value (useful for 1–5 Likert)
        - None -> no normalization
    """
    data = df.copy()

    if method is None:
        data["conf_norm"] = data[confidence_col]
        return data, "conf_norm"

    if method == "scale_0_1":
        min_c = data[confidence_col].min()
        max_c = data[confidence_col].max()
        data["conf_norm"] = (data[confidence_col] - min_c) / (max_c - min_c)

    elif method == "divide_by_max":
        max_c = data[confidence_col].max()
        data["conf_norm"] = data[confidence_col] / max_c

    else:
        raise ValueError("Unknown normalization method")

    return data, "conf_norm"


# --------------------------------------------------
# 2. Binning
# --------------------------------------------------

def _create_bins(df, confidence_col, n_bins=5, discrete=True):
    """
    Create bins for reliability diagram.

    If discrete=True, treat unique confidence levels as bins.
    If discrete=False, create equal-width bins.
    """
    data = df.copy()

    if discrete:
        data["bin"] = data[confidence_col]
    else:
        data["bin"] = pd.cut(
            data[confidence_col],
            bins=n_bins,
            include_lowest=True
        )

    return data


# --------------------------------------------------
# 3. Compute Bin Statistics
# --------------------------------------------------

def _compute_bin_statistics(df, confidence_col, correct_col):
    """
    Compute mean confidence, accuracy, and count per bin.
    """
    grouped = df.groupby("bin")

    stats = grouped.agg(
        mean_confidence=(confidence_col, "mean"),
        accuracy=(correct_col, "mean"),
        count=(correct_col, "count")
    ).reset_index()

    return stats


# --------------------------------------------------
# 4. Expected Calibration Error
# --------------------------------------------------

def _compute_ece(bin_stats, total_samples):
    """
    Compute Expected Calibration Error (ECE).
    """
    ece = np.sum(
        (bin_stats["count"] / total_samples) *
        np.abs(bin_stats["accuracy"] - bin_stats["mean_confidence"])
    )
    return ece


# --------------------------------------------------
# 5. Reliability Diagram Plot
# --------------------------------------------------

def _plot_reliability_diagram(bin_stats):
    """
    Plot reliability diagram.
    """
    plt.figure(figsize=(6, 6))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--')

    # Empirical calibration
    plt.plot(
        bin_stats["mean_confidence"],
        bin_stats["accuracy"],
        marker='o'
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# 6. Wrapper Function
# --------------------------------------------------

def reliability_analysis(
    df,
    confidence_col,
    correct_col,
    n_bins=5,
    normalize_method="divide_by_max",
    discrete_bins=True,
    plot=True
):
    """
    Full reliability analysis pipeline.
    """

    # Drop missing
    data = df[[confidence_col, correct_col]].dropna()

    # Normalize confidence
    data, conf_used = _normalize_confidence(
        data,
        confidence_col,
        method=normalize_method
    )

    # Create bins
    data = _create_bins(
        data,
        confidence_col,
        n_bins=n_bins,
        discrete=discrete_bins
    )

    # Compute bin statistics
    bin_stats = _compute_bin_statistics(
        data,
        conf_used,
        correct_col
    )

    # Compute ECE
    ece = _compute_ece(
        bin_stats,
        total_samples=len(data)
    )

    # Plot if requested
    if plot:
        _plot_reliability_diagram(bin_stats)

    return {
        "bin_statistics": bin_stats,
        "ECE": ece
    }

def compute_ece_per_person(
    df,
    participant_col,
    confidence_col,
    correct_col,
    normalize_method="linear_0_1",
):
    """
    Compute ECE per participant.
    """

    data = df[[participant_col, confidence_col, correct_col]].dropna().copy()

    # --- Confidence normalization ---
    if normalize_method == "linear_0_1":
        # 1 -> 0, 5 -> 1
        data["conf_norm"] = (data[confidence_col] - 1) / 4
    elif normalize_method == "divide_by_max":
        data["conf_norm"] = data[confidence_col] / data[confidence_col].max()
    else:
        data["conf_norm"] = data[confidence_col]

    # --- Discrete bins (since 1–5 scale) ---
    data["bin"] = data[confidence_col]

    results = []

    # --- Loop per participant ---
    for pid, df_person in data.groupby(participant_col):

        bin_stats = _compute_bin_statistics(
            df_person,
            "conf_norm",
            correct_col
        )

        ece = _compute_ece(
            bin_stats,
            total_samples=len(df_person)
        )

        results.append({
            participant_col: pid,
            "ECE": ece,
            "n_trials": len(df_person)
        })

    return pd.DataFrame(results)