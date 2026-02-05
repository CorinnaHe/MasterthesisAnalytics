import pandas as pd

def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()

    mask_pp = trials["condition"].isin([1, 2])
    mask_set = trials["condition"] == 3

    # ai correctness
    trials["ai_correct"] = pd.NA

    trials.loc[mask_pp, "ai_correct"] = (
        trials.loc[mask_pp, "point_pred_cal"]
        .eq(trials.loc[mask_pp, "y_true"])
    )

    trials.loc[mask_set, "ai_correct"] = (
        (trials.loc[mask_set, "y_true"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "y_true"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "y_true"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    )

    # human-ai agreement
    trials["initial_agree_ai"] = pd.NA
    trials["final_agree_ai"] = pd.NA

    trials.loc[mask_pp, "initial_agree_ai"] = (
        trials.loc[mask_pp, "initial_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    )

    trials.loc[mask_pp, "final_agree_ai"] = (
        trials.loc[mask_pp, "final_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    )

    trials.loc[mask_set, "initial_agree_ai"] = (
        (trials.loc[mask_set, "initial_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "initial_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "initial_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    )

    trials.loc[mask_set, "final_agree_ai"] = (
        (trials.loc[mask_set, "final_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "final_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "final_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    )

    # switching behavior
    trials["switched_to_ai"] = (
        ~trials["initial_agree_ai"] & trials["final_agree_ai"]
    )

    # reliance
    trials["over_reliance"] = (
        trials["final_agree_ai"] & ~trials["ai_correct"]
    )

    trials["under_reliance"] = (
        ~trials["final_agree_ai"] & trials["ai_correct"]
    )

    trials["appropriate_reliance"] = (
        (trials["final_agree_ai"] & trials["ai_correct"]) |
        (~trials["final_agree_ai"] & ~trials["ai_correct"])
    )
    return trials


def _construct_confidence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials["delta_confidence"] = (
        trials["final_confidence"] - trials["initial_confidence"]
    ).astype(int)

    trials["final_correct"] = trials["final_decision"].eq(trials["y_true"])

    return trials


def construct_variables_df(trials: pd.DataFrame) -> pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    trials = _construct_confidence_metrics(trials)
    return trials
