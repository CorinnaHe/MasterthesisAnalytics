from scipy import stats
import scikit_posthocs as sp

from data_loader import load_experiment_data

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        _,
        control_measures_df,
        participant_stats,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    control_measures_df = control_measures_df.merge(participant_stats[['participant_code', 'condition']], on='participant_code', how='left')

    # Schauer et Schnurr
    groups = [
        group["trust_score"].values
        for name, group in control_measures_df.groupby("condition")
    ]
    kw_stat, kw_p = stats.kruskal(*groups)

    print("Kruskal-Wallis statistic:", kw_stat)
    print("p-value:", kw_p)

    # effect size for KW
    H = kw_stat
    k = 3 # C1, C2, C3
    n = len(control_measures_df)  # total participants

    epsilon_sq = (H - k + 1) / (n - k)

    print("Epsilon-squared effect size:", epsilon_sq) # here: small

    # only if p < 0.05
    dunn = sp.posthoc_dunn(
        control_measures_df,
        val_col="trust_score",
        group_col="condition",
        p_adjust="bonferroni"
    )

    print(dunn)
