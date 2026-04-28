from data_loader import load_experiment_data
import statsmodels.formula.api as smf

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        control_measures_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    condition_df = (
        main_trials_df
        .groupby('participant_code')['condition']
        .first()
        .reset_index()
    )

    df = control_measures_df.merge(
        condition_df,
        on='participant_code',
        how='left'
    )

    print("=== Mental Load Descriptives ===")
    print(df.groupby('condition')['cognitive_load_mental'].describe())


    print("=== OLS Regression mental load ~ condition ===")
    print("\n=== Condition Logistic Regression ===")
    model = smf.ols(
        "cognitive_load_mental ~ C(condition)",
        data=df
    ).fit()
    print(model.summary())