import numpy as np
import pandas as pd

from data_loader import load_experiment_data
from variable_constructer import construct_variables_df

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import scikit_posthocs as sp
import statsmodels.formula.api as smf

if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = construct_variables_df(main_trials_df)

    # variables
    main_trials_df["initial_confidence_norm"] = (main_trials_df["initial_confidence"] - 1) / 4

    df = main_trials_df.copy()

    participant_stats = (
        df.groupby("participant_code")
        .agg(
            n_trials=("switched", "count"),
            switch_rate=("switched", "mean"),
            switch_when_disagree=("switched", lambda x: x[df.loc[x.index, "initial_agree_ai"] == 0].mean()),
            switch_when_agree=("switched", lambda x: x[df.loc[x.index, "initial_agree_ai"] == 1].mean()),
            mean_human_conf=("initial_confidence", "mean"),
        )
    )
    participant_stats = participant_stats.reset_index()
    participant_stats = participant_stats.fillna(0)

    #sns.histplot(participant_stats["switch_when_disagree"], bins=20)
    #plt.title("Distribution of Participant Switching Rates")
    #plt.show()

    features = participant_stats[
        [
            "switch_rate",
            "switch_when_disagree",
            "switch_when_agree",
            "mean_human_conf"
        ]
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    participant_stats["cluster"] = kmeans.fit_predict(X)

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

    cluster_labels = {
        0: "Selective AI reliance",
        1: "AI Skeptics",
        2: "Exploratory"
    }
    participant_stats["strategy"] = participant_stats["cluster"].map(cluster_labels)

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

    participant_condition = df[["participant_code", "condition"]].drop_duplicates()

    participant_stats = participant_stats.merge(participant_condition)

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

    accuracy_by_participant = (
        main_trials_df
        .groupby("participant_code")["final_correct"]
        .mean()
        .reset_index()
    )

    participant_stats = participant_stats.merge(
        accuracy_by_participant,
        on="participant_code"
    )
    print(participant_stats.groupby("strategy")["final_correct"].describe())
    groups = [
        participant_stats[participant_stats["strategy"] == "AI Skeptics"]["final_correct"],
        participant_stats[participant_stats["strategy"] == "Selective AI reliance"]["final_correct"],
        participant_stats[participant_stats["strategy"] == "Exploratory"]["final_correct"]
    ]
    stat, p = kruskal(*groups)
    print("H statistic:", stat)
    print("p-value:", p)
    posthoc = sp.posthoc_dunn(
        participant_stats,
        val_col="final_correct",
        group_col="strategy",
        p_adjust="bonferroni"
    )

    print(posthoc)
    H = stat
    k = 3
    n = len(participant_stats)

    eta_sq = (H - k + 1) / (n - k)

    print("Effect size (eta²):", eta_sq)

    participant_stats = participant_stats.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    participant_stats["strategy_cat"] = participant_stats["strategy"].astype("category")
    participant_stats["strategy_code"] = participant_stats["strategy_cat"].cat.codes
    print(participant_stats["strategy_cat"].cat.categories)

    participant_stats["ai_literacy"] = participant_stats[
        ["ai_literacy_sk9", "ai_literacy_sk10", "ai_literacy_ail2", "ai_literacy_ue2"]
    ].mean(axis=1)
    participant_stats["experience"] = participant_stats["domain_experience"].isin(
        ["Professional experience", "Some familiarity"]
    )


    formula = """
            strategy_code ~
            C(condition) +
            mean_human_conf +
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
