from data_loader import load_experiment_data, load_page_time_data
from inspect_data import inspect_page_times, inspect_h2, inspect_accuracy, inspect_human_ai_match, \
    plot_binary_col_by_ordinal_col
from variable_constructer import construct_variables_df
from hypothesis_testing import test_h2, test_initial_ai_agree_and_switching_regulate_confidence
import statsmodels.formula.api as smf
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def _accuracy(main_trials_df: pd.DataFrame):
    inspect_accuracy(main_trials_df, "global")

    trials_C1 = main_trials_df[main_trials_df["condition"] == "C1"]
    trials_C2 = main_trials_df[main_trials_df["condition"] == "C2"]
    inspect_accuracy(trials_C1, "C1")
    inspect_accuracy(trials_C2, "C2")

    point_trials = main_trials_df[main_trials_df["condition"].isin(["C1", "C2"])]
    set_trials = main_trials_df[main_trials_df["condition"] == "C3"]
    inspect_accuracy(point_trials, "point prediction")
    inspect_accuracy(set_trials, "set prediction")

    ai_wrong_cases = main_trials_df[main_trials_df["ai_correct"] == 0]
    ai_correct_cases = main_trials_df[main_trials_df["ai_correct"] == 1]
    inspect_accuracy(ai_wrong_cases, "ai wrong")
    inspect_accuracy(ai_correct_cases, "ai correct")


if __name__ == '__main__':
    experiment_date = "2026-02-25"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    main_trials_df = construct_variables_df(main_trials_df)

    print(main_trials_df)

    page_time_df = load_page_time_data(f"PageTimes-2026-02-05.csv")
    inspect_page_times(page_time_df)

    _accuracy(main_trials_df)

    ai_inconsistency = main_trials_df[
        (main_trials_df["point_pred_correct"] != main_trials_df["set_based_correct"]) &
        (main_trials_df["initial_correct"] != 1)
        ].copy()

    result = (
        ai_inconsistency
        .groupby("is_set_based")["final_correct"]
        .mean()
        .reset_index()
        .rename(columns={"final_correct": "final_correct_rate"})
    )
    print(result)

    #inspect_human_ai_match(main_trials_df, "global")
    #test_initial_ai_agree_and_switching_regulate_confidence(main_trials_df)


    #plot_binary_col_by_confidence(main_trials_df, "initial_correct", "initial_confidence")
    #plot_binary_col_by_confidence(main_trials_df, "final_correct", "final_confidence")

    initial_ai_disagree = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ].copy()
    plot_binary_col_by_ordinal_col(initial_ai_disagree, "switched", "initial_confidence")

    switch_model = smf.logit(
        "final_correct ~ switched * C(condition)",
        data=initial_ai_disagree
    ).fit(cov_type="cluster",
          cov_kwds={"groups": initial_ai_disagree["participant_code"]})
    print(switch_model.summary())

    point_trials = main_trials_df[main_trials_df["condition"].isin(["C1", "C2"])]
    confidence_mapping = {
        "low_confidence": 3,
        "medium_confidence": 2,
        "high_confidence": 1,
    }

    point_trials["point_pred_confidence_num"] = (
        point_trials["point_predict_conf_bin"]
        .map(confidence_mapping)
    )
    plot_binary_col_by_ordinal_col(point_trials, "switched", "point_pred_confidence_num")

    set_trials = main_trials_df[main_trials_df["condition"] == "C3"]
    plot_binary_col_by_ordinal_col(set_trials, "switched", "set_size")

    inspect_h2(main_trials_df)
    test_h2(main_trials_df)