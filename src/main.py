from data_loader import load_experiement_data
from variable_constructer import construct_variables_df

if __name__ == '__main__':
    experiment_date = "2026-02-02"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_experiement_data(f"all_apps_wide-{experiment_date}.csv")

    main_trials_df = construct_variables_df(main_trials_df)

    print(main_trials_df)