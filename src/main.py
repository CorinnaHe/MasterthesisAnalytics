from data_loader import load_experiment_data, load_page_time_data
from inspect_data import inspect_page_times
from variable_constructer import construct_variables_df

if __name__ == '__main__':
    experiment_date = "2026-02-04"
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