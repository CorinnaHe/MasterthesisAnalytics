import numpy as np
import pandas as pd


def construct_trial_level_variables(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()

    mask_pp = trials["condition"].isin(["C1", "C2"])
    mask_set = trials["condition"] == "C3"

    # human correctness
    trials["initial_correct"] = (
            (trials["y_true"].eq("poor") & trials["initial_decision"].eq("poor")) |
            (trials["y_true"].eq("standard") & trials["initial_decision"].eq("standard")) |
            (trials["y_true"].eq("good") & trials["initial_decision"].eq("good"))
    ).astype(int)

    trials["final_correct"] = (
            (trials["y_true"].eq("poor") & trials["final_decision"].eq("poor")) |
            (trials["y_true"].eq("standard") & trials["final_decision"].eq("standard")) |
            (trials["y_true"].eq("good") & trials["final_decision"].eq("good"))
    ).astype(int)

    # AI correctness
    trials["point_pred_correct"] = (
        trials["point_pred_cal"]
        .eq(trials["y_true"])
    ).astype(int)

    trials["set_based_correct"] = (
            (trials["y_true"].eq("poor") & trials["cp_contains_poor"]) |
            (trials["y_true"].eq("standard") & trials["cp_contains_standard"]) |
            (trials["y_true"].eq("good") & trials["cp_contains_good"])
    ).astype(int)

    trials["ai_correct"] = np.where(
        trials["condition"].isin(["C1", "C2"]),
        trials["point_pred_correct"],
        trials["set_based_correct"]
    )

    trials["top1_correct"] = np.where(
        trials["condition"].isin(["C1", "C2"]),
        trials["ai_correct"],
        trials["cp_set_el1"].eq(trials["y_true"])
    ).astype(int)

    # human-AI agreement
    trials["initial_agree_ai"] = pd.NA
    trials["final_agree_ai"] = pd.NA

    trials.loc[mask_pp, "initial_agree_ai"] = (
        trials.loc[mask_pp, "initial_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    ).astype(int)

    trials.loc[mask_pp, "final_agree_ai"] = (
        trials.loc[mask_pp, "final_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    ).astype(int)

    trials.loc[mask_set, "initial_agree_ai"] = (
        (trials.loc[mask_set, "initial_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "initial_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "initial_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    ).astype(int)

    trials.loc[mask_set, "final_agree_ai"] = (
        (trials.loc[mask_set, "final_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "final_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "final_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    ).astype(int)

    # top-1 agreement
    trials["initial_top_1_agree"] = pd.NA

    trials.loc[mask_pp, "initial_top_1_agree"] = (
        trials.loc[mask_pp, "initial_agree_ai"]
    ).astype(int)

    trials.loc[mask_set, "initial_top_1_agree"] = (
        trials.loc[mask_set, "initial_decision"]
        .eq(trials.loc[mask_set, "cp_set_el1"])
    ).astype(int)

    trials["initial_pos_in_set"] = pd.NA
    trials["final_pos_in_set"] = pd.NA

    for col in [
        "initial_decision",
        "final_decision",
        "cp_set_el1",
        "cp_set_el2",
        "cp_set_el3"
    ]:
        trials[col] = trials[col].astype(str).str.strip().str.lower()

    trials.loc[mask_set, "initial_pos_in_set"] = np.select(
        [
            trials.loc[mask_set, "initial_decision"].eq(
                trials.loc[mask_set, "cp_set_el1"]
            ),
            trials.loc[mask_set, "initial_decision"].eq(
                trials.loc[mask_set, "cp_set_el2"]
            ),
            trials.loc[mask_set, "initial_decision"].eq(
                trials.loc[mask_set, "cp_set_el3"]
            ),
        ],
        [1, 2, 3],
        default=-1
    )

    trials.loc[mask_set, "final_pos_in_set"] = np.select(
        [
            trials.loc[mask_set, "final_decision"].eq(
                trials.loc[mask_set, "cp_set_el1"]
            ),
            trials.loc[mask_set, "final_decision"].eq(
                trials.loc[mask_set, "cp_set_el2"]
            ),
            trials.loc[mask_set, "final_decision"].eq(
                trials.loc[mask_set, "cp_set_el3"]
            ),
        ],
        [1, 2, 3],
        default=-1
    )

    # set size
    cp_cols = [
        "cp_contains_poor",
        "cp_contains_standard",
        "cp_contains_good"
    ]

    trials["set_size"] = (
        trials[cp_cols]
        .astype(int)
        .sum(axis=1)
    )

    # shared AI confidence
    trials["shared_ai_confidence"] = pd.NA

    conf = trials.loc[mask_pp, "point_pred_confidence"]
    trials.loc[mask_pp, "shared_ai_confidence"] = (
            (conf > 0.84).astype(int) * 3
            + ((conf >= 0.625) & (conf <= 0.84)).astype(int) * 2
            + (conf < 0.625).astype(int) * 1
    )

    trials.loc[mask_set, "shared_ai_confidence"] = (
        trials.loc[mask_set, "set_size"].map({1: 3, 2: 2, 3: 1})
    )

    trials["shared_ai_confidence"] = trials["shared_ai_confidence"].astype(int)

    # confidence gap
    trials["initial_confidence_norm"] = (
        trials["initial_confidence"] - 1
    ) / 4

    trials["shared_ai_norm"] = (
        trials["shared_ai_confidence"] - 1
    ) / 2

    trials["shared_ai_norm"] = pd.to_numeric(
        trials["shared_ai_norm"],
        errors="coerce"
    )

    trials["initial_confidence_norm"] = pd.to_numeric(
        trials["initial_confidence_norm"],
        errors="coerce"
    )

    trials["confidence_gap"] = (
        trials["shared_ai_norm"] - trials["initial_confidence_norm"]
    )

    # switching
    trials["switched"] = (
            trials["initial_decision"] != trials["final_decision"]
    ).astype(int)

    # reliance
    trials["over_reliance"] = (
            (trials["final_agree_ai"] == 1) &
            (trials["ai_correct"] == 0)
    ).astype(int)

    trials["under_reliance"] = (
            (trials["final_agree_ai"] == 0) &
            (trials["ai_correct"] == 1)
    ).astype(int)

    trials["final_agree_ai"] = trials["final_agree_ai"].astype(bool)
    trials["ai_correct"] = trials["ai_correct"].astype(bool)

    trials["appropriate_reliance"] = (
        trials["final_agree_ai"] == trials["ai_correct"]
    ).astype(int)

    binary_cols = [
        "initial_correct",
        "final_correct",
        "point_pred_correct",
        "set_based_correct",
        "ai_correct",
        "top1_correct",
        "initial_agree_ai",
        "final_agree_ai",
        "initial_top_1_agree",
        "switched",
        "over_reliance",
        "under_reliance",
        "appropriate_reliance",
    ]
    trials[binary_cols] = trials[binary_cols].astype(int)

    return trials


def create_participant_stats(main_trials_df: pd.DataFrame) -> pd.DataFrame:
    participant_stats = (
        main_trials_df.groupby("participant_code")
        .agg(
            mean_page_duration_stage1=("page_duration_stage1", "mean"),
            mean_page_duration_stage2=("page_duration_stage2", "mean"),
            condition=("condition", "first"),
        )
    )

    participant_stats = participant_stats.reset_index()

    return participant_stats