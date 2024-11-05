from typing import Generator

import numpy as np
import pandas as pd


def timeseries_split(df: pd.DataFrame, max_month: int, col: str, continuous: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert df[col].dtype == np.int64, "Expected np.int64 timeseries month column"
    train_index = df[col] < max_month
    test_index = df[col] == max_month

    if continuous:
        test_index = train_index | test_index

    return df[train_index], df[test_index]


def kfold_timeseries_split(
    df: pd.DataFrame, col: str, k_min: int = 1, k_max: int = -1
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    if k_max == -1:
        k_max = df[col].max()

    for i in range(k_min, k_max + 1):
        yield timeseries_split(df, i, col, continuous=False)


def build_submission_df(evaluation_df: pd.DataFrame, save_path: None | str = None) -> pd.DataFrame:
    assert all(_ in list(evaluation_df.columns) for _ in ("shop_id", "item_id", "item_cnt_month"))

    df_test_raw = pd.read_csv(".data/test.csv")
    submission_cols = ["ID", "item_cnt_month"]
    submission = df_test_raw.merge(evaluation_df, on=["shop_id", "item_id"])[submission_cols]

    if save_path is not None:
        submission.to_csv(save_path, index=False)

    return submission
