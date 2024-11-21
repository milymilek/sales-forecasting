import random

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries(
    df: pd.DataFrame, plt_rows: int = 10, plt_cols: int = 2, rand: bool = False, pred_col: str | None = None, save_path: str | None = None
) -> None:
    fig, ax = plt.subplots(plt_rows, plt_cols, figsize=(20, 3 * plt_rows))

    cols = ["date_block_num", "shop_id", "item_id", "item_cnt_month"]
    if pred_col is not None:
        cols.append(pred_col)

    group = df[cols].groupby(["shop_id", "item_id"])
    group_list = list(iter(group))

    if rand:
        group_list = random.sample(group_list, plt_rows * plt_cols)
    else:
        group_list = group_list[: plt_rows * plt_cols]

    for i, (_, df_group) in enumerate(group_list):
        df_group = df_group[df_group["date_block_num"] < 34]

        row = i // plt_cols
        col = i % plt_cols
        ax[row, col].plot(df_group["date_block_num"], df_group["item_cnt_month"], label="Sales")
        if pred_col is not None:
            ax[row, col].plot(df_group["date_block_num"], df_group[pred_col], label="Prediction", color="red")
        ax[row, col].scatter(
            df_group[df_group.item_cnt_month != 0]["date_block_num"], df_group[df_group.item_cnt_month != 0]["item_cnt_month"], color="blue"
        )
        ax[row, col].set_ylim(-0.1, max(df_group["item_cnt_month"]) + 1)
        ax[row, col].set_title(f"Shop ID: {df_group.shop_id.iloc[0]}, Item ID: {df_group.item_id.iloc[0]}")
        ax[row, col].set_xlabel("Month")
        ax[row, col].set_ylabel("Sales")
        ax[row, col].set_xticks(range(0, 34))
        ax[row, col].legend()
        ax[row, col].grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(model: lgb.LGBMModel) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    lgb.plot_importance(model, importance_type="gain", ax=ax[0], title="LightGBM Feature Importance (Gain)")
    lgb.plot_importance(model, importance_type="split", ax=ax[1], title="LightGBM Feature Importance (Split)")

    plt.tight_layout()
    plt.show()
