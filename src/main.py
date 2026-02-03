from data_loader import load_data
from variable_constructer import construct_variables_df

if __name__ == '__main__':
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_data("all_apps_wide-2026-02-02.csv")

    main_trials_df = construct_variables_df(main_trials_df)

    print(main_trials_df)