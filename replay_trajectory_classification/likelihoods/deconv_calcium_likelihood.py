"""Zero-inflated Gamma likelihood for deconvolved calcium activity."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import gammaln
import torch
from torch import nn
import xarray as xr

from replay_trajectory_classification.core import atleast_2d

TORCH_DTYPE = torch.float32


class _ZIGNet(nn.Module):
    """Small feedforward network used to fit ZIG place-field parameters."""

    def __init__(self, x_dim: int, y_dim: int, gen_nodes: int):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, gen_nodes)
        self.fc2 = nn.Linear(gen_nodes, gen_nodes)
        self.theta_head = nn.Linear(gen_nodes, y_dim)
        self.p_head = nn.Linear(gen_nodes, y_dim)
        self.logk = nn.Parameter(torch.zeros(y_dim, dtype=TORCH_DTYPE))

        range_rate1 = 1.0 / np.sqrt(float(x_dim))
        range_rate2 = 1.0 / np.sqrt(float(gen_nodes))

        self._init_linear(self.fc1, range_rate1)
        self._init_linear(self.fc2, range_rate2)
        self._init_linear(self.theta_head, range_rate2)
        self._init_linear(self.p_head, range_rate2)

    @staticmethod
    def _init_linear(layer: nn.Linear, bound: float) -> None:
        nn.init.uniform_(layer.weight, -bound, bound)
        nn.init.zeros_(layer.bias)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.fc1(inputs))
        hidden = torch.tanh(self.fc2(hidden))

        theta = torch.exp(self.theta_head(hidden)) + 1e-6
        p = torch.clamp(torch.sigmoid(self.p_head(hidden)), 1e-6, 1.0 - 1e-6)
        k = torch.exp(self.logk) + 1e-7

        return p, theta, k


def _normalize_positions(
    position: NDArray[np.float64],
    place_bin_centers: NDArray[np.float64],
    place_bin_edges: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    place_bin_edges = atleast_2d(np.asarray(place_bin_edges, dtype=np.float32))
    offset = np.nanmin(place_bin_edges, axis=0)
    scale = np.nanmax(place_bin_edges, axis=0) - offset
    scale[scale == 0.0] = 1.0

    position = atleast_2d(np.asarray(position, dtype=np.float32))
    place_bin_centers = atleast_2d(np.asarray(place_bin_centers, dtype=np.float32))

    return (position - offset) / scale, (place_bin_centers - offset) / scale


def _get_position_coords(place_bin_centers: NDArray[np.float32]) -> dict[str, object]:
    if place_bin_centers.shape[1] == 1:
        return {"position": place_bin_centers.squeeze()}

    if place_bin_centers.shape[1] == 2:
        return {
            "position": pd.MultiIndex.from_arrays(
                place_bin_centers.T.tolist(), names=["x_position", "y_position"]
            )
        }

    raise ValueError("Only 1D and 2D position inputs are supported.")


def _iterate_minibatches(
    n_samples: int, batch_size: int, rng: np.random.Generator
) -> list[NDArray[np.int64]]:
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [
        indices[start_ind : start_ind + batch_size]
        for start_ind in range(0, n_samples, batch_size)
    ]


def _set_torch_deterministic() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_training_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        probe_inputs = torch.zeros((2, 1), dtype=TORCH_DTYPE, device="cuda")
        probe_layer = nn.Linear(1, 1).to("cuda")
        probe_outputs = probe_layer(probe_inputs)
        probe_outputs.sum().backward()
        torch.cuda.synchronize()
    except RuntimeError:
        return torch.device("cpu")

    return torch.device("cuda")


def _zig_log_likelihood_torch(
    calcium_activity: torch.Tensor,
    p: torch.Tensor,
    theta: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    positive_y = torch.clamp(calcium_activity, min=1e-12)
    k_row = k.unsqueeze(0)
    positive_log_likelihood = (
        torch.log(p)
        + (k_row - 1.0) * torch.log(positive_y)
        - (positive_y / theta)
        - k_row * torch.log(theta)
        - torch.lgamma(k_row)
    )
    zero_log_likelihood = torch.log1p(-p)
    return torch.where(
        calcium_activity > 0.0, positive_log_likelihood, zero_log_likelihood
    )


def _train_zig_model(
    position_norm: NDArray[np.float32],
    calcium_activity: NDArray[np.float32],
    predict_position_norm: NDArray[np.float32],
    gen_nodes: int,
    learning_rate: float,
    n_epochs: int,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    n_samples, x_dim = position_norm.shape
    y_dim = calcium_activity.shape[1]
    batch_size = min(batch_size, n_samples)
    rng = np.random.default_rng(0)

    _set_torch_deterministic()
    device = _get_training_device()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        print(f"Using CUDA for ZIG place-field fitting: {device_name}")

    model = _ZIGNet(x_dim=x_dim, y_dim=y_dim, gen_nodes=gen_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    position_tensor = torch.as_tensor(position_norm, dtype=TORCH_DTYPE, device=device)
    calcium_tensor = torch.as_tensor(calcium_activity, dtype=TORCH_DTYPE, device=device)
    predict_tensor = torch.as_tensor(
        predict_position_norm, dtype=TORCH_DTYPE, device=device
    )

    model.train()
    for _ in range(n_epochs):
        for batch_indices in _iterate_minibatches(n_samples, batch_size, rng):
            batch_x = position_tensor[batch_indices]
            batch_y = calcium_tensor[batch_indices]

            p, theta, k = model(batch_x)
            log_likelihood = _zig_log_likelihood_torch(batch_y, p, theta, k)
            loss = -torch.sum(log_likelihood)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        p_pred, theta_pred, k_values = model(predict_tensor)

    return (
        p_pred.detach().cpu().numpy().astype(np.float32, copy=False),
        theta_pred.detach().cpu().numpy().astype(np.float32, copy=False),
        k_values.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def estimate_zig_place_fields(
    position: NDArray[np.float64],
    calcium_activity: NDArray[np.float64],
    place_bin_centers: NDArray[np.float64],
    place_bin_edges: NDArray[np.float64],
    gen_nodes: int = 64,
    learning_rate: float = 1e-3,
    n_epochs: int = 2000,
    batch_size: int = 512,
) -> tuple[xr.DataArray, NDArray[np.float32]]:
    """Fit relaxed ZIG place fields for deconvolved calcium activity."""

    position = atleast_2d(np.asarray(position, dtype=np.float32))
    calcium_activity = np.asarray(calcium_activity, dtype=np.float32)
    place_bin_centers = atleast_2d(np.asarray(place_bin_centers, dtype=np.float32))

    if calcium_activity.ndim != 2:
        raise ValueError("calcium_activity must have shape (n_time, n_neurons).")
    if position.shape[0] != calcium_activity.shape[0]:
        raise ValueError("position and calcium_activity must have the same n_time.")
    if np.any(calcium_activity < 0.0):
        raise ValueError("calcium_activity must be non-negative for the ZIG model.")

    valid_rows = np.all(np.isfinite(position), axis=1) & np.all(
        np.isfinite(calcium_activity), axis=1
    )
    if not np.any(valid_rows):
        raise ValueError("No finite training samples are available for ZIG fitting.")

    position_norm, predict_position_norm = _normalize_positions(
        position[valid_rows], place_bin_centers, place_bin_edges
    )
    calcium_activity = calcium_activity[valid_rows]

    p_fields, theta_fields, k_values = _train_zig_model(
        position_norm=position_norm,
        calcium_activity=calcium_activity,
        predict_position_norm=predict_position_norm,
        gen_nodes=gen_nodes,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    place_fields = np.stack([p_fields, theta_fields], axis=-1)
    coords = _get_position_coords(place_bin_centers)
    coords["parameter"] = ["p", "theta"]

    return (
        xr.DataArray(
            data=place_fields,
            coords=coords,
            dims=["position", "neuron", "parameter"],
        ),
        k_values,
    )


def zig_log_likelihood(
    calcium_activity: NDArray[np.float64],
    p_field: NDArray[np.float64],
    theta_field: NDArray[np.float64],
    k: float,
) -> NDArray[np.float64]:
    """Return ZIG log likelihood for one neuron across time and bins."""

    calcium_activity = np.asarray(calcium_activity, dtype=np.float32)
    if np.any(calcium_activity < 0.0):
        raise ValueError("calcium_activity must be non-negative for the ZIG model.")

    p_field = np.clip(np.asarray(p_field, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    theta_field = np.clip(np.asarray(theta_field, dtype=np.float32), 1e-6, None)
    k = float(max(k, 1e-6))

    activity = calcium_activity[:, np.newaxis]
    positive_activity = np.maximum(activity, 1e-12)

    positive_log_likelihood = (
        np.log(p_field[np.newaxis, :])
        + (k - 1.0) * np.log(positive_activity)
        - (positive_activity / theta_field[np.newaxis, :])
        - k * np.log(theta_field[np.newaxis, :])
        - gammaln(k)
    )
    zero_log_likelihood = np.log1p(-p_field[np.newaxis, :])

    return np.where(activity > 0.0, positive_log_likelihood, zero_log_likelihood)


def combined_zig_likelihood(
    calcium_activity: NDArray[np.float64],
    place_fields: xr.DataArray | NDArray[np.float64],
    k_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine ZIG log likelihood across all neurons."""

    calcium_activity = np.asarray(calcium_activity, dtype=np.float32)
    if calcium_activity.ndim != 2:
        raise ValueError("calcium_activity must have shape (n_time, n_neurons).")

    if isinstance(place_fields, xr.DataArray):
        p_fields = np.asarray(place_fields.sel(parameter="p"), dtype=np.float32)
        theta_fields = np.asarray(place_fields.sel(parameter="theta"), dtype=np.float32)
    else:
        place_fields = np.asarray(place_fields, dtype=np.float32)
        if place_fields.ndim != 3 or place_fields.shape[-1] != 2:
            raise ValueError(
                "place_fields must have shape (n_bins, n_neurons, 2) "
                "when passed as an array."
            )
        p_fields = place_fields[..., 0]
        theta_fields = place_fields[..., 1]

    if calcium_activity.shape[1] != p_fields.shape[1]:
        raise ValueError(
            "Neuron dimension must match between activity and place fields."
        )

    k_values = np.asarray(k_values, dtype=np.float32)
    if k_values.shape != (p_fields.shape[1],):
        raise ValueError("k_values must have shape (n_neurons,).")

    n_time = calcium_activity.shape[0]
    n_bins = p_fields.shape[0]
    log_likelihood = np.zeros((n_time, n_bins), dtype=np.float32)

    for activity, p_field, theta_field, k in zip(
        calcium_activity.T, p_fields.T, theta_fields.T, k_values
    ):
        log_likelihood += zig_log_likelihood(activity, p_field, theta_field, float(k))

    return log_likelihood


def estimate_zig_likelihood(
    calcium_activity: NDArray[np.float64],
    place_fields: xr.DataArray | NDArray[np.float64],
    k_values: NDArray[np.float64],
    is_track_interior: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """Estimate the ZIG log likelihood of calcium activity across position bins."""

    if is_track_interior is not None:
        is_track_interior = np.asarray(is_track_interior).ravel(order="F")
    else:
        n_bins = place_fields.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=bool)

    log_likelihood = combined_zig_likelihood(calcium_activity, place_fields, k_values)

    mask = np.ones_like(is_track_interior, dtype=float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask
