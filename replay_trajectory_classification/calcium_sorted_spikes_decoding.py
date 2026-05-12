"""Helpers for decoding deconvolved calcium spikes with SortedSpikesDecoder."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from oasis.functions import deconvolve
import xarray as xr

from replay_trajectory_classification import (
    Environment,
    RandomWalk,
    SortedSpikesDecoder,
    estimate_movement_var,
)


def deconvolve_and_binarize(
    calcium_traces: NDArray[np.float64],
) -> NDArray[np.int8]:
    """Infer binary spikes from calcium traces with OASIS.

    Parameters
    ----------
    calcium_traces : np.ndarray, shape (n_time, n_neurons) or (n_time,)
        Simulated or observed calcium traces.

    Returns
    -------
    inferred_spikes : np.ndarray, shape (n_time, n_neurons)
        Binary spike indicators derived from the deconvolved OASIS output.
    """

    calcium_traces = np.asarray(calcium_traces, dtype=float)
    if calcium_traces.ndim == 1:
        calcium_traces = calcium_traces[:, np.newaxis]

    n_time, n_neurons = calcium_traces.shape
    shift = max(1, min(100, n_time // 2))
    inferred_spikes = np.zeros((n_time, n_neurons), dtype=np.int8)

    for neuron_ind in range(n_neurons):
        _, spikes, _, _, _ = deconvolve(
            calcium_traces[:, neuron_ind],
            g=(None, None),
            penalty=1,
            shift=shift,
        )
        inferred_spikes[:, neuron_ind] = np.asarray(spikes) > 0

    return inferred_spikes


def fit_sorted_spikes_decoder(
    position: NDArray[np.float64],
    calcium_traces: NDArray[np.float64],
    sampling_frequency: float,
    *,
    position_std: float = 3.0,
    block_size: int | None = None,
    use_diffusion: bool = False,
) -> tuple[SortedSpikesDecoder, NDArray[np.int8]]:
    """Fit a SortedSpikesDecoder from OASIS-deconvolved calcium traces."""

    inferred_spikes = deconvolve_and_binarize(calcium_traces)
    decoder = make_sorted_spikes_decoder(
        position=position,
        sampling_frequency=sampling_frequency,
        position_std=position_std,
        block_size=block_size,
        use_diffusion=use_diffusion,
    )
    decoder.fit(position, inferred_spikes)
    return decoder, inferred_spikes


def make_sorted_spikes_decoder(
    position: NDArray[np.float64],
    sampling_frequency: float,
    *,
    position_std: float = 3.0,
    block_size: int | None = None,
    use_diffusion: bool = False,
) -> SortedSpikesDecoder:
    """Construct a KDE-based SortedSpikesDecoder for 1D calcium decoding."""

    movement_var = estimate_movement_var(position, sampling_frequency)
    return SortedSpikesDecoder(
        environment=Environment(place_bin_size=np.sqrt(movement_var)),
        transition_type=RandomWalk(movement_var=movement_var),
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={
            "block_size": block_size,
            "position_std": [position_std],
            "use_diffusion": use_diffusion,
        },
    )


def maximum_a_posteriori_position(
    posterior: xr.DataArray,
) -> NDArray[np.float64]:
    """Return the MAP position trace from a 1D posterior."""

    map_position = posterior.position[posterior.argmax("position")]
    return np.asarray(map_position, dtype=float)


def median_decoding_error(
    posterior: xr.DataArray,
    true_position: NDArray[np.float64],
) -> float:
    """Compute median absolute error from a 1D posterior and true position."""

    map_position = maximum_a_posteriori_position(posterior)
    true_position = np.asarray(true_position, dtype=float).squeeze()
    return float(np.nanmedian(np.abs(map_position - true_position)))
