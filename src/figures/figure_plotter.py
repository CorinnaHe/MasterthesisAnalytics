import matplotlib.pyplot as plt
import numpy as np


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
