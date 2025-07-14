import pandas as pd
import numpy as np
import math
from typing import List, Union
import torch


def geometric_mean(nums: List[Union[float, torch.Tensor]]) -> float:
    log_sum = sum(math.log(x) for x in nums)
    return math.exp(log_sum / len(nums))


def preprocess_csv(df: pd.DataFrame) -> pd.DataFrame:
    # date conversion
    df["date"] = pd.to_datetime(df["date"])

    # time conversion to UTC
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)

    # compute travel_time
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()

    # create link column
    df["link"] = df["from_stop"] + ">" + df["to_stop"]

    # reorder columns
    columns_order = [
        "lau",
        "date",
        "line",
        "trip",
        "from_stop",
        "to_stop",
        "from_geometry",
        "to_geometry",
        "from_time",
        "to_time",
        "travel_time",
        "link",
        "route",
    ]
    df = df[columns_order]

    return df
