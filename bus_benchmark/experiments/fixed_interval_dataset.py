from bus_benchmark import config
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedIntervalDataset:
    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        simple_split: bool = True,
        simple_split_months: int = 6,
        simple_split_n_val_months: int = 1,
        simple_split_min_test_months: int = 5,
        n_splits: int = 5,
        freq: str = "15T",
        drop_ha_below_n_count: int = 20,
        ha_agg_func: str = "median",
        ffill_limit: int = 16,
        mad_thresh: float = 3 * 1.4826,
        interpolate_ha: bool = True,
        calculate_residuals: bool = True,
        normalize: bool = True,
        add_time_feat: bool = True,
        min_split_size: int = (60 // 15) * 24 * 7,
    ) -> None:
        assert isinstance(df, (pd.DataFrame, str)), (
            "df must be a DataFrame or a path to a parquet file"
        )

        self.simple_split = simple_split
        self.simple_split_months = simple_split_months
        self.simple_split_n_val_months = simple_split_n_val_months
        self.simple_split_min_test_months = simple_split_min_test_months
        self.n_splits = n_splits
        self.freq = freq
        self.drop_ha_below_n_count = drop_ha_below_n_count
        self.ha_agg_func = ha_agg_func
        self.ffill_limit = ffill_limit
        self.mad_thresh = mad_thresh
        self.interpolate_ha = interpolate_ha
        self.calculate_residuals = calculate_residuals
        self.normalize = normalize
        self.add_time_feat = add_time_feat
        self.min_split_size = min_split_size

        if isinstance(df, str):
            df = self._load_df(df)

        self.splits = self._split_time_series(df)

        self.data_set = self._preprocess_splits(self.splits)

    @staticmethod
    def _load_df(df) -> pd.DataFrame:
        if isinstance(df, str) and df.endswith(".parquet"):
            df = pd.read_parquet(df)
        else:
            raise ValueError("df must be a path to a parquet file")
        return df

    def _split_time_series(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        start_time = df["from_time"].min()
        end_time = df["from_time"].max()
        split_size = (end_time - start_time) / (self.n_splits + 1)

        splits = []
        for i in range(self.n_splits):
            val_size = 1 / 6

            train_end = start_time + (i + 1) * split_size * (1 - val_size)
            val_end = start_time + (i + 1) * split_size
            test_end = start_time + (i + 2) * split_size

            train = df[(df["from_time"] < train_end)]
            val = df[(df["from_time"] >= train_end) & (df["from_time"] < val_end)]
            test = df[(df["from_time"] >= val_end) & (df["from_time"] < test_end)]

            logger.info(
                f"Split {i + 1}: train={len(train)} (from {start_time} to {train_end}), val={len(val)} (from {train_end} to {val_end}), test={len(test)} (from {val_end} to {test_end})"
            )

            splits.append((train, val, test))

        return splits

    def _calculate_ha(
        self,
        df: pd.DataFrame,
        link_col: str = "link",
        time_col: str = "from_time",
        travel_time_col: str = "travel_time",
    ) -> pd.DataFrame:
        links = df[link_col].unique().tolist()
        timeslots = pd.date_range(
            "1970-01-04", periods=4 * 24 * 7, freq=self.freq
        ).strftime("%w_%H:%M")
        complete_index = pd.MultiIndex.from_product([links, timeslots.to_list()])

        df_ha = df.groupby(
            [df[link_col], df[time_col].dt.floor(self.freq).dt.strftime("%w_%H:%M")]
        ).agg({travel_time_col: "median"})
        df_ha = df_ha.reindex(complete_index)
        df_ha.index = df_ha.index.rename(["link", "timeslot"])
        df_ha = df_ha.rename(columns={travel_time_col: "median"})

        def interpolate_group(grp):
            grp = pd.concat([grp, grp, grp])
            grp["median"] = grp["median"].interpolate(method="linear")
            return grp.iloc[len(grp) // 3 : len(grp) // 3 * 2]

        if self.interpolate_ha:
            df_ha = df_ha.groupby(level="link", group_keys=False).apply(
                interpolate_group
            )

        return df_ha

    def _calculate_ha_residuals(
        self,
        df: pd.DataFrame,
        df_ha: pd.DataFrame,
        agg: str = "median",
        time_col: str = "from_time",
        travel_time_col: str = "travel_time",
        link_col: str = "link",
    ) -> pd.DataFrame:
        if agg not in {"mean", "median"}:
            raise ValueError("`agg` must be either 'mean' or 'median'")

        # copy & ensure datetime‐UTC
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

        # build your 15-min bin key
        df["timeslot"] = df[time_col].dt.floor(self.freq).dt.strftime("%w_%H:%M")

        # pull out just the desired stat from df_ha,
        # renaming its index levels to match df's columns
        stat_df = (
            df_ha[[agg]]
            .rename_axis(index=[link_col, "timeslot"])
            .reset_index()
            .rename(columns={agg: "ref_value"})
        )

        # merge & compute the deviation
        merged = df.merge(
            stat_df,
            on=[link_col, "timeslot"],
            how="left",
            validate="many_to_one",
        )
        diff_col = f"{travel_time_col}_minus_{agg}"
        merged[diff_col] = merged[travel_time_col] - merged["ref_value"]

        return merged

    def _pivot(
        self,
        df: pd.DataFrame,
        freq: str = "15T",
        ffill_limit=4,
        values_col="travel_time_minus_median",
    ) -> pd.DataFrame:
        def add_time_features(df):
            minutes = df.index.hour * 60 + df.index.minute
            df["sin_tod"] = np.sin(2 * np.pi * minutes / 1440)
            df["cos_tod"] = np.cos(2 * np.pi * minutes / 1440)
            return df

        # make sure from_time is datetime & floor it
        df_temp = df.copy()
        df_temp["time_slot"] = pd.to_datetime(df_temp["from_time"], utc=True).dt.floor(
            self.freq
        )

        # pivot to get a matrix: rows=time_slot, cols=link
        residual_matrix = (
            df_temp.pivot_table(
                index="time_slot",  # floored datetime index
                columns="link",  # one column per link
                values=values_col,  # your residuals
                aggfunc="mean",  # average if multiple days per slot
                fill_value=np.nan,  # or np.nan if you’d rather impute later
            ).sort_index()  # ensure chronological order
        )

        # build the full 15-min index
        full_idx = pd.date_range(
            start=residual_matrix.index.min(),
            end=residual_matrix.index.max(),
            freq=freq,
            tz="UTC",
        )

        # reindex to that full range
        residual_matrix = residual_matrix.reindex(full_idx)
        residual_matrix = residual_matrix.ffill(limit=ffill_limit)
        if self.add_time_feat:
            residual_matrix = add_time_features(residual_matrix)

        return residual_matrix

    def _preprocess_splits(
        self, splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
    ) -> List[
        Tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            Union[StandardScaler, None],
            pd.DataFrame,
        ]
    ]:
        new_splits = []
        for df_train, df_val, df_test in splits:
            logger.info(
                f"Processing split with {len(df_train)} train, {len(df_val)} val, {len(df_test)} test rows."
            )

            original_train_size = len(df_train)
            original_val_size = len(df_val)
            original_test_size = len(df_test)

            # filter outliers
            df_train = self._filter_outliers_mad(df_train, target_df=df_train)
            df_val = self._filter_outliers_mad(df_train, target_df=df_val)
            df_test = self._filter_outliers_mad(df_train, target_df=df_test)

            if df_train.empty or df_val.empty or df_test.empty:
                logger.warning(
                    f"One of the splits is empty after filtering outliers. Sizes: train={len(df_train)} (was {original_train_size}), val={len(df_val)} (was {original_val_size}), test={len(df_test)} (was {original_test_size}). Skipping this split."
                )
                continue

            # compute historical average travel times
            df_ha = self._calculate_ha(
                df_train,
                link_col="link",
                time_col="from_time",
                travel_time_col="travel_time",
            )

            # transform travel times to residuals
            if self.calculate_residuals:
                df_residuals_train = self._calculate_ha_residuals(
                    df_train, df_ha, agg=self.ha_agg_func
                )
                df_residuals_val = self._calculate_ha_residuals(
                    df_val, df_ha, agg=self.ha_agg_func
                )
                df_residuals_test = self._calculate_ha_residuals(
                    df_test, df_ha, agg=self.ha_agg_func
                )
            else:
                df_residuals_train = df_train
                df_residuals_val = df_val
                df_residuals_test = df_test

            # pivot the travel times
            values_col = (
                "travel_time_minus_median"
                if self.calculate_residuals
                else "travel_time"
            )
            absolut_matrix_train = self._pivot(
                df_train, values_col="travel_time", ffill_limit=self.ffill_limit
            )
            residual_matrix_train = self._pivot(
                df_residuals_train, values_col=values_col, ffill_limit=self.ffill_limit
            )
            residual_matrix_val = self._pivot(
                df_residuals_val, values_col=values_col, ffill_limit=self.ffill_limit
            )
            residual_matrix_test = self._pivot(
                df_residuals_test, values_col=values_col, ffill_limit=self.ffill_limit
            )

            if (
                len(residual_matrix_train) < self.min_split_size
                or len(residual_matrix_val) < self.min_split_size
                or len(residual_matrix_test) < self.min_split_size
            ):
                logger.warning(
                    f"One of the splits has less than {self.min_split_size} rows. Skipping this split."
                )
                continue

            # normalize the residuals
            if self.normalize:
                scaler = self._fit_scaler(residual_matrix_train)
                residual_matrix_train = self._apply_scaler(
                    residual_matrix_train, scaler
                )
                residual_matrix_val = self._apply_scaler(residual_matrix_val, scaler)
                residual_matrix_test = self._apply_scaler(residual_matrix_test, scaler)
            else:
                scaler = None

            new_splits.append(
                (
                    residual_matrix_train,
                    residual_matrix_val,
                    residual_matrix_test,
                    df_ha,
                    scaler,
                    absolut_matrix_train,
                )
            )
        return new_splits

    @staticmethod
    def _fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(
            train_df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").values
        )
        return scaler

    def _apply_scaler(self, df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        scaled = scaler.transform(
            df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").values
        )
        df_scaled = pd.DataFrame(
            scaled,
            index=df.index,
            columns=df.drop(columns=["sin_tod", "cos_tod"], errors="ignore").columns,
        )
        if "sin_tod" in df.columns and "cos_tod" in df.columns:
            # add sin/cos columns back to the scaled DataFrame
            df_scaled["sin_tod"] = df["sin_tod"]
            df_scaled["cos_tod"] = df["cos_tod"]
        return df_scaled

    def _filter_outliers_mad(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        group_col: str = "link",
        target_col: str = "travel_time",
        log_base: float = np.e,
    ) -> pd.DataFrame:
        # compute per-group (med, mad) thresholds on base_df
        thresholds: dict = {}
        for group_val, sub in reference_df.groupby(group_col):
            sub_pos = sub[sub[target_col] > 0]
            if sub_pos.empty:
                continue
            logs = np.log(sub_pos[target_col]) / np.log(log_base)
            med = logs.median()
            mad = (logs - med).abs().median()
            thresholds[group_val] = (med, mad)

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            keep_idx = []
            for group_val, sub in df.groupby(group_col):
                if group_val not in thresholds:
                    continue
                med, mad = thresholds[group_val]
                pos_mask = sub[target_col] > 0
                logs = np.log(sub.loc[pos_mask, target_col]) / np.log(log_base)

                if mad == 0:
                    valid_idx = sub.loc[pos_mask].index
                else:
                    dev = (logs - med).abs() / mad
                    valid_idx = logs[dev <= self.mad_thresh].index

                keep_idx.extend(valid_idx)

            if not keep_idx:
                return df.iloc[[]].copy()

            return df.loc[sorted(keep_idx)].reset_index(drop=True)

        return _filter(target_df)
