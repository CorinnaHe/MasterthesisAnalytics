import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_switching_vs_confidence_gap(
    model,
    df: pd.DataFrame,
    confidence_col: str = "confidence_gap",
    condition_col: str = "condition",
    title: str | None = None,
):
    """
    Plots predicted switching probability over confidence_gap
    using model-based predictions (OLS LPM).

    Assumes model: switched ~ confidence_gap * condition
    """

    # --- Create grid ---
    x_vals = np.linspace(-1, 1, 200)

    # Fix condition to a valid value
    condition_value = df[condition_col].iloc[0]

    pred_df = pd.DataFrame({
        confidence_col: x_vals,
        condition_col: condition_value
    })

    pred_res = model.get_prediction(pred_df).summary_frame(alpha=0.05)

    # --- Fully robust column detection ---
    def find_col(cols, keywords):
        for c in cols:
            if any(k in c for k in keywords):
                return c
        raise ValueError(f"Could not find column with keywords {keywords} in {cols}")

    cols = pred_res.columns

    mean_col = find_col(cols, ["mean", "predicted"])
    lower_col = find_col(cols, ["lower"])
    upper_col = find_col(cols, ["upper"])

    pred_df["mean"] = pred_res[mean_col]
    pred_df["ci_lower"] = pred_res[lower_col]
    pred_df["ci_upper"] = pred_res[upper_col]
    # Clip for LPM
    pred_df[["mean", "ci_lower", "ci_upper"]] = pred_df[
        ["mean", "ci_lower", "ci_upper"]
    ].clip(0, 1)

    # --- Style ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # --- Colors (your palette) ---
    line_color = "#593032"  # same as initial bars
    ci_color = "#6F8489"  # same as error bars before

    # --- Plot line ---
    ax.plot(
        pred_df[confidence_col],
        pred_df["mean"],
        color=line_color,
        linewidth=2.5
    )

    # --- CI as thick rounded band (your style) ---
    for i in range(len(pred_df)):
        ax.vlines(
            pred_df[confidence_col].iloc[i],
            pred_df["ci_lower"].iloc[i],
            pred_df["ci_upper"].iloc[i],
            colors=ci_color,
            linewidth=2,
            alpha=0.25,
            capstyle="round"
        )

    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=5)

    # --- Axes ---
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Confidence Gap")
    ax.set_ylabel("Predicted Probability of Switching")

    if title:
        ax.set_title(title, pad=12)

    # --- Clean ticks ---
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    # --- Subtle grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    ax.fill_between(
        pred_df[confidence_col],
        pred_df["ci_lower"],
        pred_df["ci_upper"],
        color=ci_color,
        alpha=0.25
    )

    plt.tight_layout()
    plt.show()