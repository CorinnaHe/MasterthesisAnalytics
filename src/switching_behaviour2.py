import numpy as np
import pandas as pd

from data_loader import load_experiment_data

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import scikit_posthocs as sp
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf


def within_condition_strategy_analysis(participant_stats: pd.DataFrame):
    conditions = participant_stats["condition"].unique()

    results = {}

    for cond in conditions:
        print(f"\n===== CONDITION: {cond} =====")

        df_cond = participant_stats[participant_stats["condition"] == cond]

        # --- Descriptives ---
        desc = df_cond.groupby("strategy")["final_accuracy"].describe()
        print("\nDescriptives:\n", desc)

        # --- Prepare groups ---
        strategies = df_cond["strategy"].unique()
        groups = [
            df_cond[df_cond["strategy"] == strat]["final_accuracy"]
            for strat in strategies
        ]

        # Only test if at least 2 groups have data
        if len(groups) < 2:
            print("Not enough groups for statistical test.")
            continue

        # --- Kruskal-Wallis ---
        stat, p = kruskal(*groups)
        print("\nKruskal-Wallis H:", stat)
        print("p-value:", p)

        # --- Effect size (eta²) ---
        H = stat
        k = len(groups)
        n = len(df_cond)

        eta_sq = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
        print("Effect size (eta²):", eta_sq)

        # --- Posthoc Dunn ---
        try:
            posthoc = sp.posthoc_dunn(
                df_cond,
                val_col="final_accuracy",
                group_col="strategy",
                p_adjust="bonferroni"
            )
            print("\nPosthoc Dunn:\n", posthoc)
        except Exception as e:
            print("Posthoc failed:", e)
            posthoc = None

        # --- Store results ---
        results[cond] = {
            "H": stat,
            "p": p,
            "eta_sq": eta_sq,
            "posthoc": posthoc,
            "descriptives": desc
        }

    return results


def check_selective_dominance(participant_stats: pd.DataFrame):
    conditions = participant_stats["condition"].unique()

    for cond in conditions:
        df_cond = participant_stats[participant_stats["condition"] == cond]

        means = df_cond.groupby("strategy")["final_accuracy"].mean().sort_values(ascending=False)

        print(f"\n===== {cond} =====")
        print(means)

        if "Selective AI reliance" in means.index:
            rank = list(means.index).index("Selective AI reliance") + 1
            print(f"Selective rank: {rank}/{len(means)}")



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

    within_condition_strategy_analysis(participant_stats)
    check_selective_dominance(participant_stats)

    model_total = smf.ols(
        "final_accuracy ~ C(condition)",
        data=participant_stats
    ).fit()

    print(model_total.summary())

    # mediation
    model = smf.ols(
        "final_accuracy ~ C(strategy) + C(condition)",
        data=participant_stats
    ).fit()
    print(model.summary())

    # interaction model
    model = smf.ols(
        "final_accuracy ~ C(strategy) * C(condition)",
        data=participant_stats
    ).fit()
    print(model.summary())

    participant_stats["strategy_cat"] = participant_stats["strategy"].astype("category")
    participant_stats["strategy_code"] = participant_stats["strategy_cat"].cat.codes
    print(participant_stats["strategy_cat"].cat.categories)


    model_strategy = smf.mnlogit(
        "strategy_code ~ C(condition)",
        data=participant_stats
    ).fit()

    print(model_strategy.summary())

    participant_stats = participant_stats.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    formula = """
            strategy_code ~
            C(condition) +
            initial_human_conf_mean +
            initial_confidence_calibration_mean +
            final_confidence_calibration_mean +
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
