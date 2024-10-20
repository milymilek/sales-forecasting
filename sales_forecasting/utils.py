import numpy as np
import pandas as pd


def timeseries_split(df: pd.DataFrame, col: str, continuous: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert df[col].dtype == np.int64, "Expected np.int64 timeseries month column"
    max_month = df[col].max()
    train_index = df[col] <= max_month - 2
    valid_index = df[col] == max_month - 1
    test_index = df[col] == max_month

    if continuous:
        valid_index = train_index | valid_index
        test_index = train_index | test_index

    return df[train_index], df[valid_index], df[test_index]
