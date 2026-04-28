from data_loader import load_experiment_data
import statsmodels.formula.api as smf

from thesis.figure_creation import plot_initial_final_per_condition

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    print("=== Accuracy Descriptives ===")
    print(main_trials_df.groupby('condition')['initial_correct'].describe())
    print(main_trials_df.groupby('condition')['final_correct'].describe())

    print("=== Accuracy Figures ===")
    plot_initial_final_per_condition(main_trials_df, "initial_correct", "final_correct", y_label="Initial vs. Final Accuracy")

    print("=== A1a: Logistic Regression correct ~ condition ===")
    model = smf.logit(
        "final_correct ~ C(condition)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== A1b: Robustness Check: Logistic Regression correct ~ condition + case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())