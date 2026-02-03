import pandas as pd


def construct_variables_df(trials)->pd.DataFrame:
    trials["pa_correct"] = (
            trials["point_pred_cal"] == trials["y_true"]
    )

    trials["initial_agree_ai"] = (
            trials["initial_decision"] == trials["point_pred_cal"]
    )

    trials["final_agree_ai"] = (
            trials["final_decision"] == trials["point_pred_cal"]
    )

    trials["switched_to_ai"] = (
            (trials["initial_decision"] != trials["point_pred_cal"]) &
            (trials["final_decision"] == trials["point_pred_cal"])
    )

    trials["over_reliance_point"] = (
            (trials["condition"] <= 2) &
            (trials["switched_to_ai"] == 1) &
            (trials["pa_correct"] == 0)
    )

    trials["under_reliance_point"] = (
            (trials["condition"].isin(["C1", "C2"])) &
            (trials["final_agree_ai"] == 0) &
            (trials["pa_correct"] == 1)
    )

    return trials