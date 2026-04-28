import statsmodels.formula.api as smf

from data_loader import load_experiment_data
from thesis.figure_creation.bar_chart import plot_calibration_switching, plot_reliance_comparison



def run_reliance_analysis(
    df,
    title: str,
    dependent_var: str,
    subset_condition=None,
):
    print(f"=== {title} ===")

    analysis_df = df if subset_condition is None else df[subset_condition(df)]

    print(analysis_df.groupby("condition")[dependent_var].mean())

    model = smf.logit(
        f"{dependent_var} ~ C(condition) + C(case_id)",
        data=analysis_df
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis_df["participant_code"]},
        disp=False
    )

    print(result.summary())


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

    analyses = [
        {
            "title": "Overreliance",
            "dependent_var": reliance_column,
            "subset": lambda d: d["ai_correct"] == False,
        },
        {
            "title": "Underreliance",
            "dependent_var": "final_agree_ai",
            "subset": lambda d: d["ai_correct"] == True,
        },
        {
            "title": "Appropriate Reliance",
            "dependent_var": "appropriate_reliance",
            "subset": None,
        },
    ]

    print("=== All Conditions ===")
    for analysis in analyses:
        run_reliance_analysis(
            df=mismatch_df,
            title=analysis["title"],
            dependent_var=analysis["dependent_var"],
            subset_condition=analysis["subset"],
        )

    print("=== C2 vs. C3 comparison (worst vs. best) ===")
    c2_c3_df = mismatch_df[mismatch_df["condition"].isin(["C2", "C3"])]
    for analysis in analyses:
        run_reliance_analysis(
            df=c2_c3_df,
            title=analysis["title"],
            dependent_var=analysis["dependent_var"],
            subset_condition=analysis["subset"],
        )

    plot_reliance_comparison(mismatch_df, "over_reliance", "appropriate_reliance")
