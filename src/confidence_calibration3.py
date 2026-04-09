from data_loader import load_experiment_data
import statsmodels.formula.api as smf

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        control_measures_df,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = main_trials_df.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    print(main_trials_df.groupby("condition")["final_calibration_score"].describe())

    print("=== 6.1 Interaction Test ===")
    model = smf.ols(
        "final_calibration_score ~ C(shared_ai_confidence) * initial_agree_ai + C(case_id)",
        data=main_trials_df
    ).fit(cov_type="cluster", cov_kwds={"groups": main_trials_df["participant_code"]})
    print(model.summary())

    print("=== 6.2 A Agree Subset Test ===")
    agree_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 1
    ].copy()
    model = smf.ols(
        "final_calibration_score ~ C(shared_ai_confidence) + C(case_id)",
        data=agree_df
    ).fit(cov_type="cluster", cov_kwds={"groups": agree_df["participant_code"]})
    print(model.summary())

    print("=== 6.2 B Disagree Subset Test ===")
    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
    ].copy()
    model = smf.ols(
        "final_calibration_score ~ C(shared_ai_confidence) + C(case_id)",
        data=mismatch_df
    ).fit(cov_type="cluster", cov_kwds={"groups": mismatch_df["participant_code"]})
    print(model.summary())

    print("=== 6.3 Cleanest Test ===")
    model = smf.ols(
        "final_calibration_score ~ switched * initial_agree_ai + C(case_id)",
        data=main_trials_df
    ).fit(cov_type="cluster", cov_kwds={"groups": main_trials_df["participant_code"]})
    print(model.summary())

    print("=== Condition Effects ===")
    model = smf.ols(
        "final_calibration_score ~ C(shared_ai_confidence) * C(condition) + initial_agree_ai + C(case_id) ",
        data=main_trials_df
    ).fit(cov_type="cluster", cov_kwds={"groups": main_trials_df["participant_code"]})
    print(model.summary())

    model = smf.ols(
        "final_calibration_score ~ C(shared_ai_confidence) + C(condition) * initial_agree_ai + C(case_id) ",
        data=main_trials_df
    ).fit(cov_type="cluster", cov_kwds={"groups": main_trials_df["participant_code"]})
    print(model.summary())

    c3_df = main_trials_df[main_trials_df["condition"] == "C3"].copy()
    model = smf.ols(
        "final_calibration_score ~ initial_pos_in_set + C(case_id) ",
        data=c3_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c3_df["participant_code"]})
    print(model.summary())

    print("=== Decomposition Test ===")
    model = smf.logit(
        "final_correct  ~ initial_agree_ai + shared_ai_confidence + C(case_id) + C(condition)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    model = smf.ols(
        "final_confidence_norm   ~ initial_agree_ai + shared_ai_confidence + C(case_id) + C(condition)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    model = smf.ols(
        "confidence_gap  ~ C(condition) * C(shared_ai_confidence) + C(case_id) ",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())