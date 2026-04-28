import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


def plot_binary_rate_per_condition(
    df: pd.DataFrame,
    column: str,
    condition_col: str = "condition",
    y_label: str | None = None,
):
    # --- Aggregate ---
    agg = (
        df.groupby(condition_col)[column]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'rate', 'count': 'n'})
    )

    # --- CI (normal approximation) ---
    z = 1.96
    agg["se"] = np.sqrt(agg["rate"] * (1 - agg["rate"]) / agg["n"])
    agg["ci_lower"] = (agg["rate"] - z * agg["se"]).clip(0, 1)
    agg["ci_upper"] = (agg["rate"] + z * agg["se"]).clip(0, 1)

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

    x = np.arange(len(agg))
    bar_width = 0.45

    bar_color = "#AFC3C2"
    ci_color  = "#593032"

    # --- Rounded bars ---
    for i, row in agg.iterrows():
        rect = FancyBboxPatch(
            (x[i] - bar_width / 2, 0),
            bar_width,
            row["rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=bar_color
        )
        ax.add_patch(rect)

    # --- Thick CI lines (rounded caps) ---
    for i, row in agg.iterrows():
        ax.vlines(
            x[i],
            row["ci_lower"],
            row["ci_upper"],
            colors=ci_color,
            linewidth=5,
            capstyle="round",
            zorder=3
        )

    # --- Axes ---
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(agg[condition_col])

    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(f"{column} rate")
    ax.set_xlabel("Condition")

    # --- Clean ticks ---
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    # --- Grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # --- Percentage labels (more spacing) ---
    for i, row in agg.iterrows():
        label = f"{row['rate'] * 100:.2f}".rstrip("0").rstrip(".") + "%"

        ax.text(
            x[i],
            row["ci_upper"] + 0.05,  # increased spacing
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


def plot_initial_final_per_condition(
    df: pd.DataFrame,
    initial_col: str,
    final_col: str,
    condition_col: str = "condition",
    title: str | None = None,
    y_label: str | None = None,
):
    """
    Plots initial vs final accuracy per condition with 95% CI.
    """

    def compute_stats(col):
        agg = (
            df.groupby(condition_col)[col]
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'rate', 'count': 'n'})
        )

        z = 1.96
        agg["se"] = np.sqrt(agg["rate"] * (1 - agg["rate"]) / agg["n"])
        agg["ci_lower"] = (agg["rate"] - z * agg["se"]).clip(0, 1)
        agg["ci_upper"] = (agg["rate"] + z * agg["se"]).clip(0, 1)

        return agg

    agg_init = compute_stats(initial_col)
    agg_final = compute_stats(final_col)

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

    x = np.arange(len(agg_init))
    bar_width = 0.30  # slightly slimmer for better spacing

    # --- Colors ---
    init_color = "#AFC3C2"
    final_color = "#6F8489"
    ci_color = "#593032"

    # --- Draw bars ---
    for i in range(len(x)):
        center_init = x[i] - bar_width / 2
        center_final = x[i] + bar_width / 2

        left_init = center_init - bar_width / 2
        left_final = center_final - bar_width / 2

        # Initial bar
        rect1 = FancyBboxPatch(
            (left_init, 0),
            bar_width,
            agg_init.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=init_color
        )
        ax.add_patch(rect1)

        # Final bar
        rect2 = FancyBboxPatch(
            (left_final, 0),
            bar_width,
            agg_final.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=final_color
        )
        ax.add_patch(rect2)

    # --- CI lines (centered) ---
    for i in range(len(x)):
        center_init = x[i] - bar_width / 2
        center_final = x[i] + bar_width / 2

        ax.vlines(
            center_init,
            agg_init.loc[i, "ci_lower"],
            agg_init.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=4,
            capstyle="round",
            zorder=3
        )

        ax.vlines(
            center_final,
            agg_final.loc[i, "ci_lower"],
            agg_final.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=4,
            capstyle="round",
            zorder=3
        )

    # --- Labels ---
    for i in range(len(x)):
        for center, agg_data in [
            (x[i] - bar_width / 2, agg_init),
            (x[i] + bar_width / 2, agg_final)
        ]:
            value = agg_data.loc[i, "rate"]
            ci_top = agg_data.loc[i, "ci_upper"]

            label = f"{value * 100:.2f}".rstrip("0").rstrip(".") + "%"

            ax.text(
                center,
                ci_top + 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=10
            )

    # --- Axes ---
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(agg_init[condition_col])

    ax.set_ylabel(y_label if y_label else "Proportion of 1")
    ax.set_xlabel("Condition")

    if title:
        ax.set_title(title, pad=12)

    # --- Clean ticks ---
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    # --- Grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # --- Legend ---
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=init_color),
            plt.Rectangle((0, 0), 1, 1, color=final_color),
        ],
        labels=["Initial", "Final"],
        frameon=False
    )

    plt.tight_layout()
    plt.show()


def plot_reliance_comparison(
    df: pd.DataFrame,
    over_col: str,
    appropriate_col: str,
    condition_col: str = "condition",
    y_label: str = "Reliance Rate",
):
    def aggregate(col):
        agg = (
            df.groupby(condition_col)[col]
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'rate', 'count': 'n'})
        )
        z = 1.96
        agg["se"] = np.sqrt(agg["rate"] * (1 - agg["rate"]) / agg["n"])
        agg["ci_lower"] = (agg["rate"] - z * agg["se"]).clip(0, 1)
        agg["ci_upper"] = (agg["rate"] + z * agg["se"]).clip(0, 1)
        return agg

    over = aggregate(over_col)
    app  = aggregate(appropriate_col)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(over))
    bar_width = 0.35

    bar_color = "#AFC3C2"
    bar_color2 = "#6F8489"
    ci_color  = "#593032"

    # --- Bars ---
    for i in range(len(x)):
        # Overreliance (LEFT)
        rect1 = FancyBboxPatch(
            (x[i] - bar_width, 0),   # ← FIXED
            bar_width,
            over.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=bar_color2,
        )
        ax.add_patch(rect1)

        # Appropriate reliance (RIGHT)
        rect2 = FancyBboxPatch(
            (x[i], 0),               # ← FIXED
            bar_width,
            app.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=bar_color
        )
        ax.add_patch(rect2)

    # --- CI ---
    for i in range(len(x)):
        # Over
        ax.vlines(
            x[i] - bar_width/2,
            over.loc[i, "ci_lower"],
            over.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=5,
            capstyle="round",
        )

        # Appropriate
        ax.vlines(
            x[i] + bar_width/2,
            app.loc[i, "ci_lower"],
            app.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=5,
            capstyle="round",
        )

    # --- Axes ---
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(over[condition_col])

    ax.set_ylabel(y_label)
    ax.set_xlabel("Condition")

    # --- Legend ---
    ax.legend(
        ["Overreliance", "Appropriate Reliance"],
        frameon=False
    )

    # --- Grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # --- Labels ---
    for i in range(len(x)):
        ax.text(
            x[i] - bar_width/2,
            over.loc[i, "ci_upper"] + 0.05,
            f"{over.loc[i, 'rate']*100:.1f}%",
            ha="center",
        )
        ax.text(
            x[i] + bar_width/2,
            app.loc[i, "ci_upper"] + 0.05,
            f"{app.loc[i, 'rate']*100:.1f}%",
            ha="center",
        )

    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    plt.show()

# The following function was generated using ChatGPT 5.3
def plot_switching_rate(
    df: pd.DataFrame,
    group_col: str,
    switch_col: str = "switched",
    x_label: str = "Initial Human–AI Match",
    y_label: str = "Switch (%)",
):
    label_map = {0: "Match", 1: "Mismatch"}
    df = df.copy()
    df[group_col] = df[group_col].map(label_map)

    counts = (
        df.groupby([group_col, switch_col])
        .size()
        .unstack(fill_value=0)
    )

    for col in [0, 1]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[[0, 1]]
    counts = counts.loc[["Match", "Mismatch"]]

    totals = counts.sum(axis=1)
    props = counts.div(totals, axis=0)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(counts))
    bar_width = 0.45

    not_switch_color = "#AFC3C2"
    switch_color = "#593032"

    rounding = 0.02

    for i in range(len(x)):
        p_not = props.iloc[i, 0]
        p_sw = props.iloc[i, 1]
        left = x[i] - bar_width / 2

        full_bar = FancyBboxPatch(
            (left, 0),
            bar_width,
            1.0,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0,
            facecolor=switch_color,
            zorder=1
        )
        ax.add_patch(full_bar)

        ax.add_patch(Rectangle(
            (left, 0),
            bar_width,
            p_not,
            linewidth=0,
            facecolor=not_switch_color,
            zorder=2
        ))

        n_not = counts.iloc[i, 0]
        n_sw = counts.iloc[i, 1]

        if n_not > 0:
            ax.text(
                x[i],
                p_not / 2,
                f"n={n_not}",
                ha="center",
                va="center",
                fontsize=10
            )

        if n_sw > 0:
            if p_sw < 0.12:
                label_y = 1.04

                ax.text(
                    x[i],
                    label_y,
                    f"n={n_sw}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

                ax.plot(
                    [x[i], x[i]],
                    [p_not + p_sw / 2, label_y - 0.01],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="round"
                )

            else:
                ax.text(
                    x[i],
                    p_not + p_sw / 2,
                    f"n={n_sw}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white"
                )

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1.08)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=not_switch_color),
            plt.Rectangle((0, 0), 1, 1, color=switch_color),
        ],
        labels=["Not Switch", "Switch"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_calibration_switching(
    df: pd.DataFrame,
    ai_correct_col: str,   # 1 = AI correct, 0 = AI wrong
    switch_col: str,       # 1 = switched
    condition_col: str = "condition",
    y_label: str | None = None,
):
    """
    Calibration plot:
    X-axis: AI correctness (Correct vs Incorrect)
    Bars: Switching probability
    Groups: Conditions (C1, C2, C3)
    """

    # --- Prepare data ---
    df = df.copy()
    df["ai_correct_label"] = df[ai_correct_col].map({
        1: "Correct",
        0: "Incorrect"
    })

    agg = (
        df.groupby([condition_col, "ai_correct_label"])[switch_col]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'rate', 'count': 'n'})
    )

    # --- CI ---
    z = 1.96
    agg["se"] = np.sqrt(agg["rate"] * (1 - agg["rate"]) / agg["n"])
    agg["ci_lower"] = (agg["rate"] - z * agg["se"]).clip(0, 1)
    agg["ci_upper"] = (agg["rate"] + z * agg["se"]).clip(0, 1)

    # --- Order ---
    correctness_order = ["Incorrect", "Correct"]
    conditions = sorted(agg[condition_col].unique())

    # --- Style ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(correctness_order))
    bar_width = 0.20  # smaller for grouping

    # --- Colors (consistent with your system) ---
    colors = ["#AFC3C2", "#6F8489", "#3E5C61"]  # C1, C2, C3
    ci_color = "#593032"

    # --- Draw bars ---
    for j, condition in enumerate(conditions):
        subset = agg[agg[condition_col] == condition]

        for i, correctness in enumerate(correctness_order):
            row = subset[subset["ai_correct_label"] == correctness]

            if row.empty:
                continue

            rate = row["rate"].values[0]

            center = x[i] + (j - (len(conditions)-1)/2) * bar_width
            left = center - bar_width / 2

            rect = FancyBboxPatch(
                (left, 0),
                bar_width,
                rate,
                boxstyle="round,pad=0,rounding_size=0.02",
                linewidth=0,
                facecolor=colors[j]
            )
            ax.add_patch(rect)

            # --- CI ---
            ax.vlines(
                center,
                row["ci_lower"].values[0],
                row["ci_upper"].values[0],
                colors=ci_color,
                linewidth=4,
                capstyle="round",
                zorder=3
            )

            # --- Label ---
            label = f"{rate * 100:.1f}".rstrip("0").rstrip(".") + "%"
            ax.text(
                center,
                row["ci_upper"].values[0] + 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=9
            )

    # --- Axes ---
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(correctness_order)

    ax.set_xlabel("AI Correctness (Top-1)")
    ax.set_ylabel(y_label if y_label else "Switching Probability")

    # --- Clean ticks ---
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    # --- Grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # --- Legend ---
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
        labels=conditions,
        frameon=False,
        title="Condition"
    )

    plt.tight_layout()
    plt.show()

def plot_initial_vs_final_agreement(
    df: pd.DataFrame,
    initial_col: str = "initial_agree_ai",
    final_col: str = "final_agree_ai",
    condition_col: str = "condition",
    y_label: str = "Human-AI Agreement Rate",
):

    # --- Aggregation ---
    def aggregate(col):
        agg = (
            df.groupby(condition_col)[col]
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'rate', 'count': 'n'})
        )
        z = 1.96
        agg["se"] = np.sqrt(agg["rate"] * (1 - agg["rate"]) / agg["n"])
        agg["ci_lower"] = (agg["rate"] - z * agg["se"]).clip(0, 1)
        agg["ci_upper"] = (agg["rate"] + z * agg["se"]).clip(0, 1)
        return agg

    initial = aggregate(initial_col)
    final   = aggregate(final_col)

    # --- Style (consistent with your existing plots) ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(initial))
    bar_width = 0.35

    bar_color = "#AFC3C2"
    ci_color  = "#593032"

    # --- Bars ---
    for i in range(len(x)):
        # Initial (LEFT, lighter)
        rect1 = FancyBboxPatch(
            (x[i] - bar_width, 0),
            bar_width,
            initial.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=bar_color,
            alpha=0.6
        )
        ax.add_patch(rect1)

        # Final (RIGHT, solid)
        rect2 = FancyBboxPatch(
            (x[i], 0),
            bar_width,
            final.loc[i, "rate"],
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0,
            facecolor=bar_color,
            alpha=1.0
        )
        ax.add_patch(rect2)

    # --- Confidence Intervals ---
    for i in range(len(x)):
        # Initial
        ax.vlines(
            x[i] - bar_width/2,
            initial.loc[i, "ci_lower"],
            initial.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=5,
            capstyle="round",
        )

        # Final
        ax.vlines(
            x[i] + bar_width/2,
            final.loc[i, "ci_lower"],
            final.loc[i, "ci_upper"],
            colors=ci_color,
            linewidth=5,
            capstyle="round",
        )

    # --- Axes ---
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(initial[condition_col])

    ax.set_ylabel(y_label)
    ax.set_xlabel("Condition")

    # --- Grid ---
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # --- Legend ---
    ax.legend(
        ["Initial agreement", "Final agreement"],
        frameon=False
    )

    # --- Value labels ---
    for i in range(len(x)):
        ax.text(
            x[i] - bar_width/2,
            initial.loc[i, "ci_upper"] + 0.05,
            f"{initial.loc[i, 'rate']*100:.1f}%",
            ha="center",
        )
        ax.text(
            x[i] + bar_width/2,
            final.loc[i, "ci_upper"] + 0.05,
            f"{final.loc[i, 'rate']*100:.1f}%",
            ha="center",
        )

    # --- Clean ticks ---
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    plt.show()