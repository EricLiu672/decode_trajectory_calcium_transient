"""Simulate calcium traces from spatially tuned neural activity.

This module adapts the AR(2) calcium trace simulation pattern used in the
reference ``simfp`` implementation and applies it to the existing spatial
simulation pipeline in this package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_place_field_firing_rate,
    simulate_position,
    simulate_time,
)
from replay_trajectory_classification import sorted_spikes_simulation

CALCIUM_SAMPLING_FREQUENCY = 30
INTERNAL_SAMPLING_FREQUENCY = 1000
TAU_D = 400.0
TAU_R = 1.0
NOISE_SIGMA = 0.3
TRACK_HEIGHT = 180
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0**2
PLACE_FIELD_MEANS = np.arange(0, TRACK_HEIGHT + 10, 10, dtype=np.float64)
N_RUNS = 15
REPLAY_SPEEDUP = 120


def compute_ar2_coefficients(
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    dt_ms: float = 1.0,
) -> tuple[float, float]:
    """Compute AR(2) coefficients from rise and decay time constants."""
    lambda_1 = np.exp(-dt_ms / tau_d)
    lambda_2 = np.exp(-dt_ms / tau_r)
    return lambda_1 + lambda_2, -lambda_1 * lambda_2


def simulate_calcium_from_spikes(
    spikes_1ms: NDArray[np.float64],
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    subsample_factor: int = INTERNAL_SAMPLING_FREQUENCY // CALCIUM_SAMPLING_FREQUENCY,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate clean and noisy calcium traces from 1 ms spike counts."""
    spikes_1ms = np.asarray(spikes_1ms, dtype=np.float64)
    if spikes_1ms.ndim == 1:
        spikes_1ms = spikes_1ms[:, np.newaxis]

    if rng is None:
        rng = np.random.default_rng()

    n_complete_samples = spikes_1ms.shape[0] // subsample_factor
    spikes_1ms = spikes_1ms[: n_complete_samples * subsample_factor]
    n_time, n_neurons = spikes_1ms.shape

    gamma_1, gamma_2 = compute_ar2_coefficients(tau_d=tau_d, tau_r=tau_r)
    clean_calcium = np.zeros((n_time, n_neurons), dtype=np.float64)
    for time_ind in range(2, n_time):
        clean_calcium[time_ind] = (
            gamma_1 * clean_calcium[time_ind - 1]
            + gamma_2 * clean_calcium[time_ind - 2]
            + spikes_1ms[time_ind]
        )

    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n_neurons,))
    noisy_calcium = clean_calcium + rng.normal(
        loc=0.0, scale=sigma, size=clean_calcium.shape
    )

    reshaped_spikes = spikes_1ms.reshape(
        n_complete_samples, subsample_factor, n_neurons
    )
    true_spikes = reshaped_spikes.sum(axis=1)
    return (
        true_spikes,
        clean_calcium[::subsample_factor],
        noisy_calcium[::subsample_factor],
    )


def _simulate_poisson_spikes_with_rng(
    rate: NDArray[np.float64],
    sampling_frequency: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    return 1.0 * (rng.poisson(rate / sampling_frequency) > 0)


def make_simulated_run_data(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    max_rate: float = 15.0,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    make_inbound_outbound_neurons: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate calcium imaging data for a linear track run session."""
    if rng is None:
        rng = np.random.default_rng()

    subsample_factor = internal_sampling_frequency // sampling_frequency
    n_samples = int(
        n_runs * internal_sampling_frequency * 2 * track_height / running_speed
    )
    time_internal = simulate_time(n_samples, internal_sampling_frequency)
    position_internal = simulate_position(time_internal, track_height, running_speed)
    place_field_means = np.asarray(place_field_means, dtype=np.float64)

    if not make_inbound_outbound_neurons:
        place_fields_internal = np.stack(
            [
                simulate_place_field_firing_rate(
                    np.array([place_field_mean]),
                    position_internal[:, np.newaxis],
                    max_rate=max_rate,
                    variance=place_field_variance,
                )
                for place_field_mean in place_field_means
            ],
            axis=1,
        )
        spikes_internal = np.stack(
            [
                _simulate_poisson_spikes_with_rng(
                    place_field, internal_sampling_frequency, rng
                )
                for place_field in place_fields_internal.T
            ],
            axis=1,
        )
    else:
        trajectory_direction = get_trajectory_direction(position_internal)
        place_fields_list: list[NDArray[np.float64]] = []
        spikes_list: list[NDArray[np.float64]] = []
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_field_mean in place_field_means:
                place_field = simulate_place_field_firing_rate(
                    np.array([place_field_mean]),
                    position_internal[:, np.newaxis],
                    max_rate=max_rate,
                    variance=place_field_variance,
                    is_condition=is_condition,
                )
                place_fields_list.append(place_field)
                spikes_list.append(
                    _simulate_poisson_spikes_with_rng(
                        place_field, internal_sampling_frequency, rng
                    )
                )
        place_fields_internal = np.stack(place_fields_list, axis=1)
        spikes_internal = np.stack(spikes_list, axis=1)

    true_spikes, _, calcium_traces = simulate_calcium_from_spikes(
        spikes_1ms=spikes_internal,
        sigma=sigma,
        tau_d=tau_d,
        tau_r=tau_r,
        subsample_factor=subsample_factor,
        rng=rng,
    )
    n_time = true_spikes.shape[0]
    return (
        time_internal[::subsample_factor][:n_time],
        position_internal[::subsample_factor][:n_time],
        sampling_frequency,
        calcium_traces,
        true_spikes,
        place_fields_internal[::subsample_factor][:n_time],
    )


def _make_calcium_replay(
    replay_time_internal: NDArray[np.float64],
    spikes_internal: NDArray[np.float64],
    sampling_frequency: int,
    internal_sampling_frequency: int,
    sigma: float | NDArray[np.float64],
    tau_d: float,
    tau_r: float,
    rng: Optional[np.random.Generator],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    subsample_factor = internal_sampling_frequency // sampling_frequency
    true_spikes, _, calcium_traces = simulate_calcium_from_spikes(
        spikes_1ms=spikes_internal,
        sigma=sigma,
        tau_d=tau_d,
        tau_r=tau_r,
        subsample_factor=subsample_factor,
        rng=rng,
    )
    n_time = true_spikes.shape[0]
    return (
        replay_time_internal[::subsample_factor][:n_time],
        true_spikes,
        calcium_traces,
    )


def make_continuous_replay(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = REPLAY_SPEEDUP,
    is_outbound: bool = True,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a continuous replay event and its calcium traces."""
    replay_time, replay_spikes = sorted_spikes_simulation.make_continuous_replay(
        sampling_frequency=internal_sampling_frequency,
        track_height=track_height,
        running_speed=running_speed,
        place_field_means=place_field_means,
        replay_speedup=replay_speedup,
        is_outbound=is_outbound,
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_hover_replay(
    hover_neuron_ind: Optional[int] = None,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a stationary replay event and its calcium traces."""
    replay_time, replay_spikes = sorted_spikes_simulation.make_hover_replay(
        hover_neuron_ind=hover_neuron_ind,
        place_field_means=place_field_means,
        sampling_frequency=internal_sampling_frequency,
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_fragmented_replay(
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a fragmented replay event and its calcium traces."""
    replay_time, replay_spikes = sorted_spikes_simulation.make_fragmented_replay(
        place_field_means=place_field_means,
        sampling_frequency=internal_sampling_frequency,
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_hover_continuous_hover_replay(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a stationary-continuous-stationary replay and calcium traces."""
    replay_time, replay_spikes = (
        sorted_spikes_simulation.make_hover_continuous_hover_replay(
        sampling_frequency=internal_sampling_frequency
        )
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a fragmented-stationary-fragmented replay and calcium traces."""
    replay_time, replay_spikes = (
        sorted_spikes_simulation.make_fragmented_hover_fragmented_replay(
        sampling_frequency=internal_sampling_frequency
        )
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a fragmented-continuous-fragmented replay and calcium traces."""
    replay_time, replay_spikes = (
        sorted_spikes_simulation.make_fragmented_continuous_fragmented_replay(
            sampling_frequency=internal_sampling_frequency
        )
    )
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )


def make_theta_sweep(
    sampling_frequency: int = CALCIUM_SAMPLING_FREQUENCY,
    internal_sampling_frequency: int = INTERNAL_SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = 145,
    sigma: float | NDArray[np.float64] = NOISE_SIGMA,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    rng: Optional[np.random.Generator] = None,
    n_sweeps: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a theta sweep event and its calcium traces."""
    _, replay_spikes_outbound = sorted_spikes_simulation.make_continuous_replay(
        sampling_frequency=internal_sampling_frequency,
        track_height=track_height,
        running_speed=running_speed,
        place_field_means=place_field_means,
        replay_speedup=replay_speedup,
        is_outbound=True,
    )
    _, replay_spikes_inbound = sorted_spikes_simulation.make_continuous_replay(
        sampling_frequency=internal_sampling_frequency,
        track_height=track_height,
        running_speed=running_speed,
        place_field_means=place_field_means,
        replay_speedup=replay_speedup,
        is_outbound=False,
    )
    single_sweep = [
        replay_spikes_inbound[replay_spikes_inbound.shape[0] // 2 :],
        replay_spikes_outbound,
    ]
    replay_spikes = np.concatenate(
        single_sweep * n_sweeps
    )
    replay_time = np.arange(replay_spikes.shape[0]) / internal_sampling_frequency
    return _make_calcium_replay(
        replay_time,
        replay_spikes,
        sampling_frequency,
        internal_sampling_frequency,
        sigma,
        tau_d,
        tau_r,
        rng,
    )
