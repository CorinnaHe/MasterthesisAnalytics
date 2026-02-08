import pandas as pd
import re
from functools import cache

from config import RAW_DATA_DIR

PLAYER_COLUMNS_TO_DROP = {
    "id_in_group",
    "role",
    "payoff",
}


CONDITION_MAP = {
    1: "poor",
    2: "standard",
    3: "good",
}

def _get_participants_df(df: pd.DataFrame) -> pd.DataFrame:
    participant_cols = [
        c for c in df.columns
        if not any(c.startswith(prefix) for prefix in [
            "example_trials.",
            "main_trials.",
            "checks.",
            "consent.",
            "instructions.",
            "cognitive_load.",
            "control_measures.",
            "closing."
        ])
    ]

    return df[participant_cols].copy()


def _extract_trials(df, prefix):
    pattern = re.compile(rf"^{prefix}\.(\d+)\.(.+)$")
    records = []

    for _, row in df.iterrows():
        participant_code = row.get("participant.code")
        condition = row.get("consent.1.player.condition")
        trial_buffer = {}

        for col, value in row.items():
            match = pattern.match(col)
            if not match:
                continue

            trial_idx = int(match.group(1))
            field = match.group(2)

            if field.startswith("player."):
                field = field.replace("player.", "", 1)

            if field in PLAYER_COLUMNS_TO_DROP:
                continue

            trial_buffer.setdefault(trial_idx, {})[field] = value

        for trial_idx, data in trial_buffer.items():
            data["participant_code"] = participant_code
            data["condition"] = condition
            data["trial_index"] = trial_idx
            records.append(data)

    df_trials = pd.DataFrame(records)
    for col in ["initial_decision", "final_decision"]:
        df_trials[col] = df_trials[col].map(CONDITION_MAP)

    if "case_id" in df_trials.columns:
        return  df_trials[df_trials["case_id"].notna()]

    return df_trials


def _extract_single_block(df, prefix, mandatory_col):
    cols = [c for c in df.columns if c.startswith(f"{prefix}.1.")]
    out = df[["participant.code"] + cols].copy()

    rename_map = {
        c: c.replace(f"{prefix}.1.player.", "", 1)
        for c in cols
    }
    out = out.rename(columns=rename_map)

    drop_cols = [
        c for c in out.columns
        if c in PLAYER_COLUMNS_TO_DROP
    ]
    out = out.drop(columns=drop_cols, errors="ignore")
    out = out.rename(columns={"participant.code": "participant_code"})
    if mandatory_col in out.columns:
        return out[out[mandatory_col].notna()]

    return out


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    # exclude bots
    df = df[df["participant._is_bot"] == 0]
    # completed participants
    df = df[
        df["participant._index_in_pages"] >=
        (df["participant._max_page_index"] - 1)
        ]
    # exclude failed checks
    df = df[df["checks.1.player.failed_checks"] <= 1]

    return df


@cache
def load_experiment_data(file_name: str) -> pd.DataFrame:
    df_raw = pd.read_csv(RAW_DATA_DIR / file_name)
    df_raw = _filter_df(df_raw)

    return (
        _get_participants_df(df_raw),
        _extract_trials(df_raw, "example_trials"),
        _extract_trials(df_raw, "main_trials"),
        _extract_single_block(df_raw, "control_measures", "age"),
    )


def load_page_time_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / file_name)

    df = df.sort_values(
        ["participant_code", "round_number", "page_index"]
    )

    df["page_time_sec"] = (
        df.groupby(["participant_code", "round_number"])["epoch_time_completed"]
        .diff()
    )

    return df