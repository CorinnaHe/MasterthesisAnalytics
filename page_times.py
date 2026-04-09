from data_loader import load_experiment_data

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        _,
        control_measures_df,
        participant_stats,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    participant_stats = participant_stats.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    print("=== Page Times per Participant ===")
    print(participant_stats["mean_page_duration"].describe())

    Q1, Q3 = participant_stats["mean_page_duration"].quantile(0.25), participant_stats["mean_page_duration"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    mask = participant_stats["mean_page_duration"] > upper_bound
    outlier_count = mask.sum()
    outlier_df = participant_stats[mask]
    print(
        f"{outlier_count} did exceed the upper bound of {upper_bound}. These are {outlier_count / len(participant_stats) * 100}%")
    print("\n= Outlier Participants =")
    print(outlier_df[["participant_code", "mean_page_duration", "condition", "age", "experience"]])

    print("=== Page Times per Condition ===") # => condition had no effect on page times
    print(participant_stats.groupby("condition")["mean_page_duration"].describe())

    from scipy.stats import shapiro

    for cond in participant_stats["condition"].unique():
        stat, p = shapiro(
            participant_stats.loc[
                participant_stats["condition"] == cond,
                "mean_page_duration"
            ]
        )
        print(f"{cond}: W={stat:.3f}, p={p:.3f}")

    from scipy.stats import levene

    groups = [
        participant_stats.loc[participant_stats["condition"] == c, "mean_page_duration"]
        for c in participant_stats["condition"].unique()
    ]

    stat, p = levene(*groups)
    print(f"Levene test: stat={stat:.3f}, p={p:.3f}")

    from scipy.stats import f_oneway

    groups = [
        participant_stats.loc[participant_stats["condition"] == c, "mean_page_duration"]
        for c in ["C1", "C2", "C3"]
    ]

    f_stat, p_value = f_oneway(*groups)

    print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(
        endog=participant_stats["mean_page_duration"],
        groups=participant_stats["condition"],
        alpha=0.05
    )

    print(tukey)

    from scipy.stats import kruskal

    h_stat, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_value:.3f}")