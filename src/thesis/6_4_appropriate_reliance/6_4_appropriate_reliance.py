from data_loader import load_experiment_data
import statsmodels.formula.api as smf

from thesis.figure_creation.bar_chart import plot_calibration_switching, plot_reliance_comparison

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    reliance_column = "final_agree_ai"
    main_trials_df[reliance_column] = main_trials_df[reliance_column].astype(int)
    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ].copy()

    print("=== Overreliance ===")
    ai_wrong_df = mismatch_df[mismatch_df["ai_correct"] == False]
    print(ai_wrong_df.groupby("condition")[reliance_column].mean())

    model = smf.logit(
        f"{reliance_column} ~ C(condition) + C(case_id)",
        data=ai_wrong_df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": ai_wrong_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== Underreliance ===")
    # added to Cao et al. -> Underreliance
    ai_correct_df = mismatch_df[mismatch_df["ai_correct"] == True]
    print(ai_correct_df.groupby("condition")[reliance_column].mean())

    model = smf.logit(
        f"final_agree_ai ~ C(condition) + C(case_id)",
        data=ai_correct_df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": ai_correct_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== Appropriate Reliance ===")
    # addded to Cao et al. -> Appropriate Reliance
    print(mismatch_df.groupby("condition")["appropriate_reliance"].mean())

    model = smf.logit(
        f"appropriate_reliance ~ C(condition) + C(case_id)",
        data=mismatch_df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== C2 vs. C3 comparison (worst vs. best ===")
    c2_c3_trials = mismatch_df[mismatch_df["condition"].isin(["C2", "C3"])]
    print("=== Overreliance ===")
    ai_wrong_df = c2_c3_trials[c2_c3_trials["ai_correct"] == False]
    print(ai_wrong_df.groupby("condition")[reliance_column].mean())

    model = smf.logit(
        f"{reliance_column} ~ C(condition) + C(case_id)",
        data=ai_wrong_df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": ai_wrong_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== Underreliance ===")
    # added to Cao et al. -> Underreliance
    ai_correct_df = c2_c3_trials[c2_c3_trials["ai_correct"] == True]
    print(ai_correct_df.groupby("condition")[reliance_column].mean())

    model = smf.logit(
        f"final_agree_ai ~ C(condition) + C(case_id)",
        data=ai_correct_df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": ai_correct_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== Appropriate Reliance ===")
    # addded to Cao et al. -> Appropriate Reliance
    print(c2_c3_trials.groupby("condition")["appropriate_reliance"].mean())

    model = smf.logit(
        f"appropriate_reliance ~ C(condition) + C(case_id)",
        data=c2_c3_trials
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": c2_c3_trials["participant_code"]},
        disp=False
    )
    print(result.summary())

    plot_reliance_comparison(mismatch_df, "over_reliance", "appropriate_reliance")
