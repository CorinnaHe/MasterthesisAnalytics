from data_loader import load_experiment_data
import statsmodels.formula.api as smf

from thesis.figure_creation.line_chart import plot_switching_vs_confidence_gap

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    top1_mismatch_df = main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
        ].copy()


    print("=== Switching Descriptives (Full df) ===")
    print(main_trials_df.groupby('condition')['switched'].describe())
    print("=== Switching Descriptives (Top 1 mismatch df) ===")
    print(top1_mismatch_df.groupby('condition')['switched'].describe())

    print(f"\n=== Switched by shared_ai_confidence * Condition ===")
    model = smf.logit(
        f"switched ~ shared_ai_confidence * C(condition)",
        data=top1_mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": top1_mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by initial_confidence * Condition ===")
    model = smf.logit(
        f"switched ~ initial_confidence * C(condition)",
        data=top1_mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": top1_mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by Confidence Gap * Condition ===")
    model = smf.logit(
        f"switched ~ confidence_gap * C(condition)",
        data=top1_mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": top1_mismatch_df["participant_code"]}
    )
    print(model.summary())

    plot_switching_vs_confidence_gap(model, top1_mismatch_df)