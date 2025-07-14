import torch
import pandas as pd
from typing import Literal, Any, Optional
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from collections.abc import MutableMapping


class RunMetrics(MutableMapping):
    """
    Class to compute and store training metrics.
    """

    def __init__(
        self,
        prefix: Literal["eval", "test"],
        absolut_matrix_train: Optional[pd.DataFrame] = None,
        x_list: Optional[list[torch.Tensor]] = None,
        y_list: Optional[list[torch.Tensor]] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.epoch: int = 1

        self.scaler = scaler
        self.prefix = prefix

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metrics: defaultdict[int, dict[str, Any]] = defaultdict(dict)

        # seasonal MASE and RMSSE
        if absolut_matrix_train is not None:
            self.mase_denominator, self.rmsse_denominator = (
                self._get_seasonal_mase_rmsse_denominator(absolut_matrix_train)
            )
            assert self.mase_denominator > 0, "MASE denominator is zero or negative"
            assert self.rmsse_denominator > 0, "RMSSE denominator is zero or negative"
        else:
            self.mase_denominator = None
            self.rmsse_denominator = None

    def __getattr__(self, name):
        current = self.metrics[self.epoch]
        return getattr(current, name)

    def __getitem__(self, key):
        return self.metrics[self.epoch][key]

    def __setitem__(self, key, value):
        self.metrics[self.epoch][key] = value

    def __delitem__(self, key):
        del self.metrics[self.epoch][key]

    def __iter__(self):
        return iter(self.metrics[self.epoch])

    def __len__(self):
        return len(self.metrics[self.epoch])

    def __repr__(self):
        all_metrics = dict(self.metrics)
        return (
            f"<{self.__class__.__name__}"
            f" current_prefix={self.prefix!r}"
            f" current_epoch={self.epoch}"
            f" metrics={all_metrics!r}>"
        )

    def update_prefix(self, prefix: Literal["eval", "test"]):
        self.prefix = prefix

    @property
    def best_val_metrics(self) -> dict[str, torch.Tensor]:
        min_val_loss_idx = min(
            [(k, v["eval/loss"]) for k, v in self.metrics.items()], key=lambda x: x[1]
        )[0]
        best_val_metrics = self.metrics[min_val_loss_idx]
        return {
            f"best_eval/{k.split('eval/')[-1]}": v
            for k, v in best_val_metrics.items()
            if "eval/" in k
        }

    @property
    def test_metrics(self) -> dict[str, torch.Tensor]:
        assert "test/loss" in self.metrics[self.epoch].keys(), (
            "Test loss not in metrics. This function can only be used if the evaluation in test mode was done in the last epoch."
        )
        return {k: v for k, v in self.metrics[self.epoch].items() if "test/" in k}

    @property
    def final_metrics(self) -> dict[str, torch.Tensor]:
        return self.best_val_metrics | self.test_metrics

    def model_mae_rmse(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: Literal["overall", "per_route", "per_horizon"] = "overall",
        scale: bool = False,
    ) -> dict[str, torch.Tensor]:
        if scale:
            assert self.scaler is not None, "scaler must be provided if scale is True"
            y_pred = self._unscale(y_pred, self.scaler)
            y_true = self._unscale(y_true, self.scaler)

        ae = (y_true - y_pred).abs()
        se = (y_true - y_pred) ** 2

        if reduction == "overall":
            mae = ae.mean()
            rmse = se.mean().sqrt()
        elif reduction == "per_route":
            mae = ae.mean(dim=(0, 2))
            rmse = se.mean(dim=(0, 2)).sqrt()
        elif reduction == "per_horizon":
            mae = ae.mean(dim=(0, 1))
            rmse = se.mean(dim=(0, 1)).sqrt()

        output = {
            f"{self.prefix}/Model_MAE_{reduction}": mae,
            f"{self.prefix}/Model_RMSE_{reduction}": rmse,
        }

        self.metrics[self.epoch].update(output)

        return output

    def ha_mae_rmse(
        self,
        res_true: torch.Tensor,
        reduction: Literal["overall", "per_route", "per_horizon"] = "overall",
        scale: bool = False,
    ) -> dict[str, torch.Tensor]:
        if scale:
            assert self.scaler is not None, "scaler must be provided if scale is True"
            res_true = self._unscale(res_true, self.scaler)

        ae_baseline = res_true.abs()
        se_baseline = res_true**2

        if reduction == "overall":
            mae_b = ae_baseline.mean()
            rmse_b = se_baseline.mean().sqrt()
        elif reduction == "per_route":
            mae_b = ae_baseline.mean(dim=(0, 2))
            rmse_b = se_baseline.mean(dim=(0, 2)).sqrt()
        elif reduction == "per_horizon":
            mae_b = ae_baseline.mean(dim=(0, 1))
            rmse_b = se_baseline.mean(dim=(0, 1)).sqrt()

        output = {
            f"{self.prefix}/HA_MAE_{reduction}": mae_b,
            f"{self.prefix}/HA_RMSE_{reduction}": rmse_b,
        }

        self.metrics[self.epoch].update(output)

        return output

    @staticmethod
    def _get_seasonal_mase_rmsse_denominator(
        absolut_matrix_train: pd.DataFrame,
    ) -> tuple[float, float]:
        df = absolut_matrix_train.copy()

        # number of 15min periods in a week
        n = 7 * 24 * 4

        # pick just the route columns (everything except sin_tod / cos_tod)
        cols = df.columns.difference(["sin_tod", "cos_tod"])

        # shift future values back to the current timestamps
        future = df[cols].shift(-n)

        # compute current minus future
        df_diff = df[cols] - future

        # drop all rows that didn’t have a value one week ahead
        df_diff = df_diff.dropna()

        mase_denominator = df_diff.abs().mean().mean()
        rmsse_denominator = (df_diff**2).mean().mean() ** 0.5

        assert mase_denominator > 0, "MASE denominator is zero or negative"
        assert rmsse_denominator > 0, "RMSSE denominator is zero or negative"

        return mase_denominator, rmsse_denominator

    def model_smase_smrsse(
        self,
        res_true: torch.Tensor,
        res_pred: torch.Tensor,
        mase_denominator: Optional[float] = None,
        rmsse_denominator: Optional[float] = None,
        reduction: Literal["overall", "per_route", "per_horizon"] = "overall",
        scale: bool = False,
    ) -> dict[str, torch.Tensor]:
        if scale:
            assert self.scaler is not None, "scaler must be provided if scale is True"
            res_pred = self._unscale(res_pred, self.scaler)
            res_true = self._unscale(res_true, self.scaler)

        mase_denominator = (
            mase_denominator if mase_denominator is not None else self.mase_denominator
        )
        rmsse_denominator = (
            rmsse_denominator
            if rmsse_denominator is not None
            else self.rmsse_denominator
        )

        if mase_denominator is None or rmsse_denominator is None:
            raise ValueError(
                "mase_denominator and rmsse_denominator must be provided or set in the constructor"
            )

        assert mase_denominator > 0, "MASE denominator is zero or negative"
        assert rmsse_denominator > 0, "RMSSE denominator is zero or negative"

        ae_model = (res_true - res_pred).abs()  # |e_model|
        se_model = (res_true - res_pred) ** 2

        if reduction == "overall":
            mae_m = ae_model.mean()
            rmse_m = se_model.mean().sqrt()
        elif reduction == "per_route":
            raise NotImplementedError
        elif reduction == "per_horizon":
            raise NotImplementedError

        mase = mae_m / mase_denominator
        rmsse = rmse_m / rmsse_denominator

        output = {
            f"{self.prefix}/Model_Seasonal_MASE_{reduction}": mase,
            f"{self.prefix}/Model_Seasonal_RMSSE_{reduction}": rmsse,
        }

        self.metrics[self.epoch].update(output)

        return output

    def ha_smase_srmsse(
        self,
        res_true: torch.Tensor,
        mase_denominator: Optional[float] = None,
        rmsse_denominator: Optional[float] = None,
        reduction: Literal["overall", "per_route", "per_horizon"] = "overall",
        scale: bool = False,
    ) -> dict[str, torch.Tensor]:
        if scale:
            assert self.scaler is not None, "scaler must be provided if scale is True"
            res_true = self._unscale(res_true, self.scaler)

        mase_denominator = (
            mase_denominator if mase_denominator is not None else self.mase_denominator
        )
        rmsse_denominator = (
            rmsse_denominator
            if rmsse_denominator is not None
            else self.rmsse_denominator
        )

        if mase_denominator is None or rmsse_denominator is None:
            raise ValueError(
                "mase_denominator and rmsse_denominator must be provided or set in the constructor"
            )

        assert mase_denominator > 0, "MASE denominator is zero or negative"
        assert rmsse_denominator > 0, "RMSSE denominator is zero or negative"

        ae_ha = res_true.abs()
        se_ha = res_true**2

        if reduction == "overall":
            mae_m = ae_ha.mean()
            rmse_m = se_ha.mean().sqrt()
        elif reduction == "per_route":
            raise NotImplementedError
        elif reduction == "per_horizon":
            raise NotImplementedError

        mase = mae_m / mase_denominator
        rmsse = rmse_m / rmsse_denominator

        output = {
            f"{self.prefix}/HA_Seasonal_MASE_{reduction}": mase,
            f"{self.prefix}/HA_Seasonal_RMSSE_{reduction}": rmsse,
        }

        self.metrics[self.epoch].update(output)

        return output

    def mase_rmsse_vs_HA(
        self,
        res_true: torch.Tensor,
        res_pred: torch.Tensor,
        reduction: Literal["overall", "per_route", "per_horizon"] = "overall",
    ) -> dict[str, torch.Tensor]:
        ae_model = (res_true - res_pred).abs()  # |e_model|
        ae_baseline = res_true.abs()  # |e_HA|
        se_model = (res_true - res_pred) ** 2
        se_baseline = res_true**2

        if reduction == "overall":
            mae_m = ae_model.mean()
            mae_b = ae_baseline.mean()
            rmse_m = se_model.mean().sqrt()
            rmse_b = se_baseline.mean().sqrt()
        elif reduction == "per_route":
            mae_m = ae_model.mean(dim=(0, 2))
            mae_b = ae_baseline.mean(dim=(0, 2))
            rmse_m = se_model.mean(dim=(0, 2)).sqrt()
            rmse_b = se_baseline.mean(dim=(0, 2)).sqrt()
        elif reduction == "per_horizon":
            mae_m = ae_model.mean(dim=(0, 1))
            mae_b = ae_baseline.mean(dim=(0, 1))
            rmse_m = se_model.mean(dim=(0, 1)).sqrt()
            rmse_b = se_baseline.mean(dim=(0, 1)).sqrt()

        mase = mae_m / mae_b.clamp(min=1e-8)
        rmsse = rmse_m / rmse_b.clamp(min=1e-8)
        output = {}

        if reduction == "per_route" or reduction == "per_horizon":
            for i in range(len(mase)):
                output[f"{self.prefix}/Model_HA_MASE_{reduction}_{i}"] = mase[i]
                output[f"{self.prefix}/Model_HA_RMSSE_{reduction}_{i}"] = rmsse[i]
        else:
            output[f"{self.prefix}/Model_HA_MASE_{reduction}"] = mase
            output[f"{self.prefix}/Model_HA_RMSSE_{reduction}"] = rmsse

        self.metrics[self.epoch].update(output)

        return output

    def _unscale(self, y, scaler):
        # get original shape
        original_shape = y.shape  # [680, 4, 25]

        # reshape to [680*4, 25] (what the scaler expects)
        flat = y.cpu().numpy().reshape(-1, original_shape[2])

        # apply inverse transform
        unflat = scaler.inverse_transform(flat)

        # reshape back and transpose to original structure
        unscaled = unflat.reshape(original_shape)

        return torch.from_numpy(unscaled).to(self.device)
