import pandas as pd

from config import INSPECT_DIR

def _mean_duration_by_page(df: pd.DataFrame):
    mean_df = df.groupby("page_index")[["page_time_sec"]].describe()
    print(mean_df)
    mean_df.to_csv(INSPECT_DIR / "page_times" / "page_time_mean.csv")


def _total_time_per_participant(df: pd.DataFrame):
    total_time_per_participant = (
        df
        .groupby("participant_code")["page_time_sec"]
        .sum()
    )
    total_time_per_participant.to_csv(INSPECT_DIR / "page_times" / "page_time_mean.csv")
    print(total_time_per_participant.describe())


def inspect_page_times(df: pd.DataFrame):
    _mean_duration_by_page(df)
    _total_time_per_participant(df)

