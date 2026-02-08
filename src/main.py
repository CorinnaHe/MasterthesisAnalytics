from data_loader import load_experiment_data, load_page_time_data
from inspect_data import inspect_page_times, inspect_h2, inspect_accuracy
from variable_constructer import construct_variables_df
from hypothesis_testing import test_h2

if __name__ == '__main__':
    experiment_date = "synthetic_experiment_data"
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

    point_trials = main_trials_df[main_trials_df["condition"].isin([1, 2])]
    set_trials = main_trials_df[main_trials_df["condition"] == 3]
    inspect_accuracy(point_trials, "point prediction")
    inspect_accuracy(set_trials, "set prediction")
    inspect_accuracy(main_trials_df, "global")

    inspect_h2(main_trials_df)

    test_h2(main_trials_df)