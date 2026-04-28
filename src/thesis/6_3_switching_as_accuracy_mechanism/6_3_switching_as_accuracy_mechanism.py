import pandas as pd
from scipy.stats import chi2_contingency

from data_loader import load_experiment_data
import statsmodels.formula.api as smf

from thesis.figure_creation import plot_predicted_accuracy_lpm_ci
from thesis.figure_creation.bar_chart import plot_calibration_switching

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ].copy()

    top1_mismatch_df = main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
        ].copy()

    print("=== Switching Descriptives (Full df) ===")
    print(main_trials_df.groupby('condition')['switched'].describe())
    print("=== Switching Descriptives (Top1 mismatch df) ===")
    print(top1_mismatch_df.groupby('condition')['switched'].describe())
    print(
        top1_mismatch_df[top1_mismatch_df['switched'] == 1]
        .groupby('condition')['final_agree_ai']
        .agg(['sum', 'mean', 'count'])
    )
    print("=== Switching Descriptives (Complete mismatch df) ===")
    print(mismatch_df.groupby('condition')['switched'].describe())
    print(
        mismatch_df[mismatch_df['switched'] == 1]
        .groupby('condition')['final_agree_ai']
        .agg(['sum', 'mean', 'count'])
    )

    print("=== A14: OLS switched * top1_correct + C(condition) (Top1 mismatch df) ===")
    model = smf.ols(
        "final_correct ~ switched * top1_correct + C(condition)",
        data=top1_mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": top1_mismatch_df["participant_code"]}
    )
    print(model.summary())

    plot_predicted_accuracy_lpm_ci(
        model, )

    print("=== Additional Analysis: OLS final_correct ~ switched * C(condition) (Full df) ===")
    switch_model = smf.logit(
        "final_correct ~ switched * C(condition)",
        data=top1_mismatch_df
    ).fit(cov_type="cluster",
          cov_kwds={"groups": top1_mismatch_df["participant_code"]})
    print(switch_model.summary())

    plot_calibration_switching(top1_mismatch_df, "top1_correct", "switched")

    print("=== C3: Switching is directed to Top-1 ===")
    c3_switched = main_trials_df[
        (main_trials_df["switched"] == 1) & (main_trials_df["condition"] == "C3")
        ].copy()
    c3_switched["moved_to_top1"] = (c3_switched["final_pos_in_set"] == 1).astype(int)
    print(c3_switched["moved_to_top1"].mean())

    table_counts = pd.crosstab(
        c3_switched["initial_pos_in_set"],
        c3_switched["moved_to_top1"]
    )
    table_counts.columns = ["not_moved_to_top1", "moved_to_top1"]
    chi2, p, dof, expected = chi2_contingency(
        table_counts[["not_moved_to_top1", "moved_to_top1"]]
    )
    print("\n=== Chi-square test ===")
    print(f"Chi2 statistic: {chi2:.4f}")
    print(f"p-value: {p:.6f}")
    print(f"Degrees of freedom: {dof}")
