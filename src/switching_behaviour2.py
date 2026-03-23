import numpy as np
import pandas as pd

from data_loader import load_experiment_data

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import scikit_posthocs as sp
import statsmodels.formula.api as smf

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        _,
        control_measures_df,
        participant_stats,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    #sns.histplot(participant_stats["switch_when_disagree"], bins=20)
    #plt.title("Distribution of Participant Switching Rates")
    #plt.show()

    cluster_summary = (
        participant_stats
        .groupby("cluster")
        .mean(numeric_only=True)
    )
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print(cluster_summary)
        print(participant_stats["strategy"].value_counts())

    sns.scatterplot(
        data=participant_stats,
        x="switch_when_disagree",
        y="switch_when_agree",
        hue="strategy",
        palette="Set2"
    )

    plt.xlabel("Switch when AI disagrees")
    plt.ylabel("Switch when AI agrees")
    plt.title("Participant Switching Strategies")
    #plt.show()

    strategy_by_condition = pd.crosstab(
        participant_stats["condition"],
        participant_stats["strategy"],
        normalize="index"
    )
    print(strategy_by_condition)
    table = pd.crosstab(
        participant_stats["condition"],
        participant_stats["strategy"]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    residuals = (table - expected) / np.sqrt(expected)
    print("Chi2:", chi2)
    print("p-value:", p) # < 0.05

    print("Standardized residuals")
    print(residuals)

    n = table.values.sum()
    k = min(table.shape)

    cramers_v = np.sqrt(chi2 / (n * (k - 1)))

    print("Cramer's V:", cramers_v)

    print(participant_stats.groupby("strategy")["final_accuracy"].describe())
    groups = [
        participant_stats[participant_stats["strategy"] == "AI Skeptics"]["final_accuracy"],
        participant_stats[participant_stats["strategy"] == "Selective AI reliance"]["final_accuracy"],
        participant_stats[participant_stats["strategy"] == "Exploratory"]["final_accuracy"]
    ]
    stat, p = kruskal(*groups)
    print("H statistic:", stat)
    print("p-value:", p)
    posthoc = sp.posthoc_dunn(
        participant_stats,
        val_col="final_accuracy",
        group_col="strategy",
        p_adjust="bonferroni"
    )

    print(posthoc)
    H = stat
    k = 3
    n = len(participant_stats)

    eta_sq = (H - k + 1) / (n - k)

    print("Effect size (eta²):", eta_sq)


    participant_stats["strategy_cat"] = participant_stats["strategy"].astype("category")
    participant_stats["strategy_code"] = participant_stats["strategy_cat"].cat.codes
    print(participant_stats["strategy_cat"].cat.categories)

    participant_stats = participant_stats.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    formula = """
            strategy_code ~
            C(condition) +
            initial_human_conf_mean +
            ai_literacy +
            experience +
            risk_aversion +
            cognitive_load_mental +
            ai_attitude +
            ai_trust
        """
    model = smf.mnlogit(
        formula,
        data=participant_stats
    ).fit()
    print(model.summary())
