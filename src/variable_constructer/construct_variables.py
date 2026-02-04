import pandas as pd

def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials["set_valid"] = (
        ((trials["y_true"] == "poor") & (trials["cp_contains_poor"] == 1)) |
        ((trials["y_true"] == "standard") & (trials["cp_contains_standard"] == 1)) |
        ((trials["y_true"] == "good") & (trials["cp_contains_good"] == 1))
    )

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
            (trials["switched_to_ai"] == 1) &
            (trials["pa_correct"] == 0)
    )

    trials["under_reliance_point"] = (
            (trials["final_agree_ai"] == 0) &
            (trials["pa_correct"] == 1)
    )

    trials["set_consistent"] = (
        ((trials["final_decision"] == "good") & (trials["cp_contains_good"] == 1)) |
        ((trials["final_decision"] == "standard") & (trials["cp_contains_standard"] == 1)) |
        ((trials["final_decision"] == "poor") & (trials["cp_contains_poor"] == 1))
    )

    trials["under_reliance_set"] = (
        (trials["set_consistent"] == 0) &
        (trials["set_valid"] == 1)
    )

    return trials


def construct_variables_df(trials)->pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    return trials