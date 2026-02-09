import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_binary_stacked_bar(
    counts: dict,
    *,
    category_order=None,
    outcome_order=None,
    colors=None,
    annotate=True,
    ylabel="Percentage",
    xlabel=None,
    figsize=(6, 5),
    ax=None,
):
    if category_order is None:
        category_order = list(counts.keys())

    if outcome_order is None:
        outcome_order = list(next(iter(counts.values())).keys())

    if colors is None:
        colors = {outcome: None for outcome in outcome_order}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(category_order))
    width = 0.6

    # Convert counts → percentages
    totals = np.array(
        [sum(counts[cat][out] for out in outcome_order) for cat in category_order]
    )

    bottoms = np.zeros(len(category_order))

    for outcome in outcome_order:
        values = np.array([counts[cat][outcome] for cat in category_order])
        pct = values / totals * 100

        bars = ax.bar(
            x,
            pct,
            width,
            bottom=bottoms,
            color=colors.get(outcome),
            edgecolor="black",
            label=outcome,
        )

        if annotate:
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val == 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + pct[i] / 2,
                    f"n={val}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        bottoms += pct

    # Axes formatting
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)

    if xlabel:
        ax.set_xlabel(xlabel)

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()
    return ax



def plot_box_with_jitter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list | None = None,
    show_stats: bool = True,
    figsize=(8, 6),
    jitter=0.08,
):
    if order is None:
        order = list(df[x_col].dropna().unique())

    fig, ax = plt.subplots(figsize=figsize)

    data = [df.loc[df[x_col] == cat, y_col].dropna() for cat in order]

    # --- boxplot ---
    bp = ax.boxplot(
        data,
        positions=np.arange(len(order)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black"),
        boxprops=dict(facecolor="white", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # --- jittered points ---
    for i, values in enumerate(data):
        x_jittered = np.random.normal(i, jitter, size=len(values))
        ax.scatter(
            x_jittered,
            values,
            alpha=0.7,
            s=18,
            zorder=2,
        )

        # --- mean & SD annotation ---
        if show_stats:
            mean = values.mean()
            sd = values.std()
            ax.text(
                i,
                mean,
                f"M={mean:.2f}\nSD={sd:.2f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=9,
            )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("User Choices")
    ax.set_ylabel(y_col.replace("_", " ").title())

    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6)

    plt.tight_layout()
    plt.show()
    return fig, ax
