from statistics import mean

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()

    mask_pp = trials["condition"].isin(["C1", "C2"])
    mask_set = trials["condition"] == "C3"

    trials["is_set_based"] = (trials["condition"] == "C3").astype(int)
    # human correctness
    trials["initial_correct"] = (
            (trials["y_true"].eq("poor") & trials["initial_decision"]) |
            (trials["y_true"].eq("standard") & trials["initial_decision"]) |
            (trials["y_true"].eq("good") & trials["initial_decision"])
    ).astype(int)

    trials["final_correct"] = (
            (trials["y_true"].eq("poor") & trials["final_decision"]) |
            (trials["y_true"].eq("standard") & trials["final_decision"]) |
            (trials["y_true"].eq("good") & trials["final_decision"])
    ).astype(int)

    # ai correctness
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

    # human-ai agreement
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

    trials["initial_pos_in_set"] = pd.NA
    trials["final_pos_in_set"] = pd.NA

    for col in ["initial_decision", "final_decision", "cp_set_el1", "cp_set_el2", "cp_set_el3"]:
        trials[col] = trials[col].astype(str).str.strip().str.lower()

    trials.loc[mask_set, "initial_pos_in_set"] = np.select(
        [
            trials.loc[mask_set, "initial_decision"].eq(trials.loc[mask_set, "cp_set_el1"]),
            trials.loc[mask_set, "initial_decision"].eq(trials.loc[mask_set, "cp_set_el2"]),
            trials.loc[mask_set, "initial_decision"].eq(trials.loc[mask_set, "cp_set_el3"]),
        ],
        [1, 2, 3],
        default=-1
    )
    trials.loc[mask_set, "final_pos_in_set"] = np.select(
        [
            trials.loc[mask_set, "final_decision"].eq(trials.loc[mask_set, "cp_set_el1"]),
            trials.loc[mask_set, "final_decision"].eq(trials.loc[mask_set, "cp_set_el2"]),
            trials.loc[mask_set, "final_decision"].eq(trials.loc[mask_set, "cp_set_el3"]),
        ],
        [1, 2, 3],
        default=-1
    )

    trials["final_choice_type"] = np.select(
        [
            trials["final_pos_in_set"] == 1,
            trials["final_pos_in_set"].isin([2, 3]),
            trials["final_pos_in_set"] == -1
        ],
        ["top1", "alternative", "outside"],
        default="outside"
    )

    # set_size
    cp_cols = [
        "cp_contains_poor",
        "cp_contains_standard",
        "cp_contains_good"
    ]

    #trials["set_size"] = np.nan
    #trials.loc[mask_set, "set_size"] = (
    #    trials.loc[mask_set, cp_cols]
    #    .astype(int)
    #    .sum(axis=1)
    #)
    trials["set_size"] = (
        trials[cp_cols]
        .astype(int)
        .sum(axis=1)
    )


    # shared_ai_confidence: common 1–3 scale (high = 3) across point- and set-based conditions.
    #   For point-based (C1, included in mask_pp), use point_pred_confidence thresholds:
    #   low=1 if <0.625, medium=2 if 0.625–0.84, high=3 if >0.84.
    #   For set-based (mask_set), invert set_size so 1->3, 2->2, 3->1 to align with the same scale.
    trials["shared_ai_confidence"] = pd.NA
    conf = trials.loc[mask_pp, "point_pred_confidence"]
    trials.loc[mask_pp, "shared_ai_confidence"] = (
            (conf > 0.84).astype(int) * 3
            + ((conf >= 0.625) & (conf <= 0.84)).astype(int) * 2
            + (conf < 0.625).astype(int) * 1
    )
    trials.loc[mask_set, "shared_ai_confidence"] = trials.loc[mask_set, "set_size"].map({1: 3, 2: 2, 3: 1})

    # human-ai confidence gap: shared ai confidence norm - initial human confidence norm
    trials["initial_confidence_norm"] = (trials["initial_confidence"] - 1) / 4
    trials["final_confidence_norm"] = (trials["final_confidence"] - 1) / 4
    trials["shared_ai_norm"] = (trials["shared_ai_confidence"] - 1) / 2
    trials["shared_ai_norm"] = pd.to_numeric(trials["shared_ai_norm"], errors="coerce")
    trials["initial_confidence_norm"] = pd.to_numeric(trials["initial_confidence_norm"], errors="coerce")
    trials["confidence_gap"] = trials["shared_ai_norm"] - trials["initial_confidence_norm"]    # signed gap

    # confidence calibration score
    trials["initial_calibration_score"] = trials["initial_confidence_norm"] - trials["initial_correct"]
    trials["final_calibration_score"] = trials["final_confidence_norm"] - trials["final_correct"]

    # switching behavior
    trials["switched"] = (
            trials["initial_decision"] != trials["final_decision"]
    ).astype(int)

    trials["switched_to_ai"] = (
        ~trials["initial_agree_ai"] & trials["final_agree_ai"]
    ).astype(int)

    # reliance
    trials["over_reliance"] = (
        trials["final_agree_ai"] & ~trials["ai_correct"]
    ).astype(int)

    trials["under_reliance"] = (
        ~trials["final_agree_ai"] & trials["ai_correct"]
    ).astype(int)

    trials["final_agree_ai"] = trials["final_agree_ai"].astype(bool)
    trials["ai_correct"] = trials["ai_correct"].astype(bool)

    trials["appropriate_reliance"] = (
        trials["final_agree_ai"] == trials["ai_correct"]
    ).astype(int)

    # page duration
    trials["mean_page_duration"] = (trials["page_duration_stage1"] + trials["page_duration_stage2"]) / 2
    trials["delta_page_duration"] = (trials["page_duration_stage1"] - trials["page_duration_stage2"])

    return trials


def _construct_confidence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials["delta_confidence"] = (
        trials["final_confidence"] - trials["initial_confidence"]
    ).astype(int)

    trials["final_correct"] = trials["final_decision"].eq(trials["y_true"]).astype(int)
    trials["initial_correct"] = trials["initial_decision"].eq(trials["y_true"]).astype(int)

    return trials


def construct_trial_level_variables(trials: pd.DataFrame) -> pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    trials = _construct_confidence_metrics(trials)
    return trials


def add_consolidated_control_measures(control_measures_df: pd.DataFrame) -> pd.DataFrame:
    control_measures_df["ai_literacy"] = control_measures_df[
        ["ai_literacy_sk9", "ai_literacy_sk10", "ai_literacy_ail2", "ai_literacy_ue2"]
    ].mean(axis=1)

    control_measures_df["experience"] = control_measures_df["domain_experience"].isin(
        ["Professional experience", "Some familiarity"]
    )

    trust_items = ["ai_attitude", "ai_trust"]
    control_measures_df["trust_score"] = control_measures_df[trust_items].mean(axis=1)

    return control_measures_df

def _create_behavioral_clusters(participant_stats: pd.DataFrame) -> pd.DataFrame:
    features = participant_stats[
        [
            "switch_rate",
            "switch_when_disagree",
            "switch_when_agree",
            "initial_human_conf_mean"
        ]
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    participant_stats["cluster"] = kmeans.fit_predict(X)
    cluster_labels = {
        0: "Selective AI reliance",
        1: "AI Skeptics",
        2: "Exploratory"
    }
    participant_stats["strategy"] = participant_stats["cluster"].map(cluster_labels)

    return participant_stats


def create_participant_stats(main_trials_df: pd.DataFrame) -> pd.DataFrame:
    # aggregate measures from participants trials
    participant_stats = (
        main_trials_df.groupby("participant_code")
        .agg(
            n_trials=("switched", "count"),
            switch_rate=("switched", "mean"),
            switch_when_disagree=("switched", lambda x: x[main_trials_df.loc[x.index, "initial_agree_ai"] == 0].mean()),
            switch_when_agree=("switched", lambda x: x[main_trials_df.loc[x.index, "initial_agree_ai"] == 1].mean()),
            initial_human_conf_mean=("initial_confidence", "mean"),
            final_human_conf_mean=("final_confidence", "mean"),
            initial_accuracy=("initial_correct", "mean"),
            final_accuracy=("final_correct", "mean"),
            mean_page_duration_stage1=("page_duration_stage1", "mean"),
            mean_page_duration_stage2=("page_duration_stage2", "mean"),
            condition=("condition", "first"),
        )
    )
    participant_stats["mean_page_duration"] = (participant_stats["mean_page_duration_stage1"] + participant_stats["mean_page_duration_stage2"])/2
    participant_stats["delta_page_duration"] = (participant_stats["mean_page_duration_stage1"] - participant_stats["mean_page_duration_stage2"])
    participant_stats = participant_stats.fillna(0)
    participant_stats = participant_stats.reset_index()

    # create clusters from stage 1 switching behavior and confidence
    participant_stats = _create_behavioral_clusters(participant_stats)

    return participant_stats


def create_case_stats(main_trials_df: pd.DataFrame) -> pd.DataFrame:
    # accuracy measures
    case_df = (
        main_trials_df
        .groupby("case_id")
        .agg(
            user_only_accuracy=("initial_correct", "mean"),
            team_accuracy=("final_correct", "mean"),
            ai_accuracy=("ai_correct", "mean"),
            mean_page_duration_stage1=("page_duration_stage1", "mean"),
            mean_page_duration_stage2=("page_duration_stage2", "mean"),
            n_trials=("initial_correct", "count")
        )
        .reset_index()
    )
    case_df["team_delta"] = (
            case_df["team_accuracy"] - case_df["user_only_accuracy"]
    )
    case_df["mean_page_duration"] = (case_df["mean_page_duration_stage1"] + case_df[
        "mean_page_duration_stage2"]) / 2
    case_df["delta_page_duration"] = (
                case_df["mean_page_duration_stage1"] - case_df["mean_page_duration_stage2"])

    # add difficulty bins
    bins = [-np.inf, 0.25, 0.5, 0.75, np.inf]
    labels = ["very_hard", "hard", "medium", "easy"]
    # human difficulty
    case_df["human_difficulty"] = pd.cut(
        case_df["user_only_accuracy"],
        bins=bins,
        labels=labels
    )
    # ai difficulty
    case_df["ai_difficulty"] = pd.cut(
        case_df["ai_accuracy"],
        bins=bins,
        labels=labels
    )

    return case_df