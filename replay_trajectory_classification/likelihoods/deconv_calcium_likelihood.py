"""Zero-inflated Gamma likelihood for deconvolved calcium activity."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import gammaln
import tensorflow.compat.v1 as tf
import xarray as xr

from replay_trajectory_classification.core import atleast_2d

tf.disable_v2_behavior()

DTYPE = tf.float32


class FullLayer:
    """Fully connected layer helper copied from the reference ZIG implementation."""

    def __init__(self):
        self.nl_dict = {
            "softplus": tf.nn.softplus,
            "linear": tf.identity,
            "softmax": tf.nn.softmax,
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
        }

    def __call__(
        self,
        inputs: tf.Tensor,
        nodes: int,
        nl: str = "softplus",
        scope: Optional[str] = None,
        name: str = "out",
        initializer=None,
        b_initializer=None,
        isconst: bool = False,
    ) -> tf.Tensor:
        nonlinearity = self.nl_dict[nl]
        input_dim = inputs.get_shape()[-1]

        if b_initializer is None:
            b_initializer = tf.zeros([nodes])
        if initializer is None:
            initializer = tf.orthogonal_initializer()

        with tf.variable_scope(scope):
            if isconst:
                weights = tf.get_variable(
                    "weights", dtype=DTYPE, initializer=initializer
                )
            else:
                weights = tf.get_variable(
                    "weights",
                    [input_dim, nodes],
                    dtype=DTYPE,
                    initializer=initializer,
                )

            biases = tf.get_variable("biases", dtype=DTYPE, initializer=b_initializer)
            return nonlinearity(tf.matmul(inputs, weights) + biases, name=name)


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

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(0)

        x = tf.placeholder(DTYPE, shape=[None, x_dim], name="position")
        y = tf.placeholder(DTYPE, shape=[None, y_dim], name="calcium_activity")

        fully_connected_layer = FullLayer()
        range_rate1 = 1.0 / np.sqrt(float(x_dim))
        range_rate2 = 1.0 / np.sqrt(float(gen_nodes))

        with tf.variable_scope("sngrlx_rate_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(
                x,
                gen_nodes,
                nl="tanh",
                scope="full1",
                initializer=tf.random_uniform_initializer(
                    minval=-range_rate1, maxval=range_rate1
                ),
            )
            full2 = fully_connected_layer(
                full1,
                gen_nodes,
                nl="tanh",
                scope="full2",
                initializer=tf.random_uniform_initializer(
                    minval=-range_rate2, maxval=range_rate2
                ),
            )
            full_theta = fully_connected_layer(
                full2,
                y_dim,
                nl="linear",
                scope="output_theta",
                initializer=tf.random_uniform_initializer(
                    minval=-range_rate2, maxval=range_rate2
                ),
            )
            full_p = fully_connected_layer(
                full2,
                y_dim,
                nl="linear",
                scope="output_p",
                initializer=tf.random_uniform_initializer(
                    minval=-range_rate2, maxval=range_rate2
                ),
            )

        with tf.variable_scope("sngrlx_obsmodel", reuse=tf.AUTO_REUSE):
            logk = tf.get_variable(
                "logk", initializer=tf.cast(tf.zeros(y_dim), DTYPE), dtype=DTYPE
            )
            k = tf.exp(logk) + 1e-7

        theta = tf.exp(full_theta) + 1e-6
        p = tf.clip_by_value(tf.nn.sigmoid(full_p), 1e-6, 1.0 - 1e-6)

        positive_y = tf.maximum(y, 1e-12)
        k_row = tf.reshape(k, (1, y_dim))
        positive_log_likelihood = (
            tf.math.log(p)
            + (k_row - 1.0) * tf.math.log(positive_y)
            - (positive_y / theta)
            - k_row * tf.math.log(theta)
            - tf.math.lgamma(k_row)
        )
        zero_log_likelihood = tf.math.log1p(-p)
        log_likelihood = tf.where(y > 0.0, positive_log_likelihood, zero_log_likelihood)

        loss = -tf.reduce_sum(log_likelihood)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        init_op = tf.global_variables_initializer()
        session_config = tf.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )

    with tf.Session(graph=graph, config=session_config) as session:
        session.run(init_op)

        for _ in range(n_epochs):
            for batch_indices in _iterate_minibatches(n_samples, batch_size, rng):
                session.run(
                    train_op,
                    feed_dict={
                        x: position_norm[batch_indices],
                        y: calcium_activity[batch_indices],
                    },
                )

        p_pred, theta_pred, k_values = session.run(
            [p, theta, k], feed_dict={x: predict_position_norm}
        )

    return (
        np.asarray(p_pred, dtype=np.float32),
        np.asarray(theta_pred, dtype=np.float32),
        np.asarray(k_values, dtype=np.float32),
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
