from data_loader import load_experiment_data, load_page_time_data
from inspect_data import inspect_page_times, inspect_h2, inspect_accuracy, inspect_human_ai_match
from variable_constructer import construct_variables_df
from hypothesis_testing import test_h2, test_initial_ai_agree_and_switching_regulate_confidence

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def _accuracy(main_trials_df: pd.DataFrame):
    inspect_accuracy(main_trials_df, "global")

    point_trials = main_trials_df[main_trials_df["condition"].isin(["C1", "C2"])]
    set_trials = main_trials_df[main_trials_df["condition"] == "C3"]
    inspect_accuracy(point_trials, "point prediction")
    inspect_accuracy(set_trials, "set prediction")

    ai_wrong_cases = main_trials_df[main_trials_df["ai_correct"] == 0]
    ai_correct_cases = main_trials_df[main_trials_df["ai_correct"] == 1]
    inspect_accuracy(ai_wrong_cases, "ai wrong")
    inspect_accuracy(ai_correct_cases, "ai correct")


if __name__ == '__main__':
    experiment_date = "2026-02-13"
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
    inspect_human_ai_match(main_trials_df, "global")
    test_initial_ai_agree_and_switching_regulate_confidence(main_trials_df)

    inspect_h2(main_trials_df)
    test_h2(main_trials_df)