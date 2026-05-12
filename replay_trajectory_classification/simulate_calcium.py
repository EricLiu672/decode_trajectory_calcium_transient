"""Functions for generating calcium imaging simulation data.

This module provides simulation functions for calcium imaging data,
modelling the AR(2) fluorescence dynamics observed in two-photon imaging
experiments. Spikes are simulated at 1 ms resolution then convolved with
the AR(2) kernel and downsampled to the imaging frame rate.

Outputs per neuron:
- ground-truth spike train (binned to imaging frames)
- noiseless calcium trace (downsampled)
- noisy calcium trace (downsampled, Gaussian noise added)

References
----------
.. [1] Vogelstein, J.T. et al. (2010). Fast nonnegative deconvolution for
   spike train inference from population calcium imaging. J. Neurophysiology.

"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_place_field_firing_rate,
    simulate_position,
    simulate_time,
)
from replay_trajectory_classification.sorted_spikes_simulation import (
    PLACE_FIELD_MEANS,
    PLACE_FIELD_VARIANCE,
    REPLAY_SPEEDUP,
    RUNNING_SPEED,
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
    make_continuous_replay,
    make_fragmented_replay,
    make_hover_replay,
)

# ── Module-level defaults ────────────────────────────────────────────────────

IMAGING_FREQUENCY: float = 30.0  # Hz — typical two-photon frame rate
INTERNAL_DT_MS: float = 1.0      # ms — internal simulation timestep
TAU_D: float = 400.0             # ms — calcium decay time constant
TAU_R: float = 1.0               # ms — calcium rise time constant
CALCIUM_NOISE_STD: float = 0.1   # Gaussian noise std (signal units)
MAX_RATE: float = 15.0           # Hz — default peak place-field firing rate
N_RUNS: int = 15                 # number of laps for run simulation


# ── Low-level helpers ────────────────────────────────────────────────────────


def simulate_ar2_calcium(
    spike_train: NDArray[np.float64],
    noise_std: Union[float, NDArray[np.float64]] = CALCIUM_NOISE_STD,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    imaging_frequency: float = IMAGING_FREQUENCY,
    random_seed: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply AR(2) calcium dynamics to a spike train at 1 ms resolution.

    Simulates fluorescence by convolving a binary spike train with an AR(2)
    kernel parameterised by a decay time constant `tau_d` and rise time
    constant `tau_r`, then adds Gaussian noise and downsamples to the
    imaging frame rate.

    Parameters
    ----------
    spike_train : NDArray[np.float64], shape (n_time_1ms,) or (n_time_1ms, n_neurons)
        Binary (or integer) spike train at 1 ms resolution.
    noise_std : float or NDArray[np.float64], shape (n_neurons,), optional
        Per-neuron Gaussian noise standard deviation. A scalar is broadcast
        to all neurons.
    tau_d : float, optional
        Calcium decay time constant in milliseconds.
    tau_r : float, optional
        Calcium rise time constant in milliseconds.
    imaging_frequency : float, optional
        Target frame rate in Hz. Determines the downsampling factor
        ``k = round(1000 / imaging_frequency)``.
    random_seed : int or None, optional
        Seed for the random number generator. Pass an integer for
        reproducible results.

    Returns
    -------
    spikes_binned : NDArray[np.float64], shape (n_frames,) or (n_frames, n_neurons)
        Spike counts summed within each imaging frame.
    calcium_clean : NDArray[np.float64], same shape as spikes_binned
        Noiseless AR(2) calcium trace sampled at each frame boundary.
    calcium_noisy : NDArray[np.float64], same shape as spikes_binned
        Noisy calcium trace (clean + Gaussian noise).

    Notes
    -----
    AR(2) recursion::

        γ₁ = λ₁ + λ₂,   γ₂ = −λ₁ λ₂
        c[t] = γ₁ c[t−1] + γ₂ c[t−2] + s[t]

    where ``λ₁ = exp(−dt / τ_d)``, ``λ₂ = exp(−dt / τ_r)``, dt = 1 ms.

    """
    rng = np.random.default_rng(random_seed)

    spike_train = np.asarray(spike_train, dtype=np.float64)
    is_1d = spike_train.ndim == 1
    if is_1d:
        spike_train = spike_train[:, np.newaxis]

    n_time, n_neurons = spike_train.shape

    # AR(2) coefficients
    lambda_1 = np.exp(-INTERNAL_DT_MS / tau_d)
    lambda_2 = np.exp(-INTERNAL_DT_MS / tau_r)
    gamma_1 = lambda_1 + lambda_2
    gamma_2 = -lambda_1 * lambda_2

    # Forward AR(2) pass
    calcium = np.zeros_like(spike_train)
    for t in range(2, n_time):
        calcium[t] = gamma_1 * calcium[t - 1] + gamma_2 * calcium[t - 2] + spike_train[t]

    # Noise — broadcast scalar to per-neuron array
    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=np.float64), (n_neurons,))
    noise = rng.normal(0.0, noise_std[np.newaxis, :], size=(n_time, n_neurons))
    calcium_noisy = calcium + noise

    # Downsample to imaging frequency
    k = round(1000.0 / imaging_frequency)
    n_frames = n_time // k

    if n_frames == 0:
        # Spike train shorter than one imaging frame: collapse to a single frame.
        spikes_binned = spike_train.sum(axis=0, keepdims=True)
        calcium_clean_ds = calcium[[-1]]
        calcium_noisy_ds = calcium_noisy[[-1]]
    else:
        frame_indices = np.arange(n_frames) * k
        # Spike counts per frame (sum)
        spikes_binned = np.stack(
            [spike_train[i * k : (i + 1) * k].sum(axis=0) for i in range(n_frames)],
            axis=0,
        )
        calcium_clean_ds = calcium[frame_indices]
        calcium_noisy_ds = calcium_noisy[frame_indices]

    if is_1d:
        return spikes_binned[:, 0], calcium_clean_ds[:, 0], calcium_noisy_ds[:, 0]

    return spikes_binned, calcium_clean_ds, calcium_noisy_ds


def simulate_calcium_neuron(
    firing_rate: NDArray[np.float64],
    spike_model: str = "poisson",
    random_seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate a binary spike train at 1 ms resolution from a firing rate.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_time_1ms,)
        Instantaneous firing rate in Hz at each 1 ms time step.
    spike_model : {"poisson", "bernoulli"}, optional
        Spike generation process.  ``"poisson"`` draws from a Poisson
        distribution; ``"bernoulli"`` draws a Bernoulli trial capped at
        probability 1.
    random_seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    spikes : NDArray[np.float64], shape (n_time_1ms,)
        Binary spike train (0 or 1 at each millisecond).

    """
    rng = np.random.default_rng(random_seed)
    rate_per_ms = np.asarray(firing_rate, dtype=np.float64) / 1000.0

    if spike_model == "poisson":
        return (rng.poisson(rate_per_ms) > 0).astype(np.float64)
    elif spike_model == "bernoulli":
        prob = np.minimum(rate_per_ms, 1.0)
        return rng.binomial(1, prob).astype(np.float64)
    else:
        raise ValueError(
            f"spike_model must be 'poisson' or 'bernoulli', got '{spike_model}'."
        )


# ── High-level run simulation ────────────────────────────────────────────────


def make_calcium_run_data(
    imaging_frequency: float = IMAGING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    max_rate: float = MAX_RATE,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    noise_std: Union[float, NDArray[np.float64]] = CALCIUM_NOISE_STD,
    spike_model: str = "poisson",
    make_inbound_outbound_neurons: bool = False,
    random_seed: Optional[int] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate an animal running on a linear track with calcium imaging.

    Generates position at 1 ms resolution, derives place field firing rates
    for each neuron, produces spike trains via the requested spike model, then
    applies AR(2) calcium dynamics and Gaussian noise before downsampling to
    the imaging frame rate.

    Parameters
    ----------
    imaging_frequency : float, optional
        Calcium imaging frame rate in Hz.
    track_height : float, optional
        Length of the linear track (same units as position).
    running_speed : float, optional
        Animal running speed in position units per second.
    n_runs : int, optional
        Number of full back-and-forth laps to simulate.
    place_field_variance : float, optional
        Spatial variance of each Gaussian place field.
    place_field_means : NDArray[np.float64], shape (n_neurons,), optional
        Centre of each neuron's place field.
    max_rate : float, optional
        Peak firing rate of each place field in Hz.
    tau_d : float, optional
        AR(2) decay time constant in milliseconds.
    tau_r : float, optional
        AR(2) rise time constant in milliseconds.
    noise_std : float or NDArray[np.float64], optional
        Per-neuron Gaussian noise standard deviation on calcium traces.
    spike_model : {"poisson", "bernoulli"}, optional
        Spike generation model.
    make_inbound_outbound_neurons : bool, optional
        If True, neurons are direction-selective (active on one direction
        only).
    random_seed : int or None, optional
        Global random seed for reproducibility.

    Returns
    -------
    time : NDArray[np.float64], shape (n_frames,)
        Time in seconds at each imaging frame.
    position : NDArray[np.float64], shape (n_frames,)
        Animal position downsampled to imaging frame rate.
    imaging_frequency : float
        Frame rate (echoed for convenience).
    spikes : NDArray[np.float64], shape (n_frames, n_neurons)
        Ground-truth spike counts per imaging frame.
    calcium_clean : NDArray[np.float64], shape (n_frames, n_neurons)
        Noiseless AR(2) calcium traces at imaging rate.
    calcium_noisy : NDArray[np.float64], shape (n_frames, n_neurons)
        Noisy calcium traces at imaging rate.
    place_fields : NDArray[np.float64], shape (n_time_1ms, n_neurons)
        Firing-rate place fields at 1 ms resolution.

    """
    rng = np.random.default_rng(random_seed)

    # 1. Simulate position at 1 ms resolution (SAMPLING_FREQUENCY = 1000 Hz)
    n_samples_1ms = int(n_runs * SAMPLING_FREQUENCY * 2 * track_height / running_speed)
    time_1ms = simulate_time(n_samples_1ms, SAMPLING_FREQUENCY)
    position_1ms = simulate_position(time_1ms, track_height, running_speed)

    place_field_means = np.asarray(place_field_means, dtype=np.float64)
    n_neurons = place_field_means.shape[0] if place_field_means.ndim == 1 else place_field_means.shape[0]

    # 2. Build place fields and spike trains at 1 ms
    if not make_inbound_outbound_neurons:
        place_fields_list = [
            simulate_place_field_firing_rate(
                mean, position_1ms, max_rate=max_rate, variance=place_field_variance
            )
            for mean in place_field_means
        ]
        spike_trains = [
            simulate_calcium_neuron(pf, spike_model=spike_model, random_seed=rng.integers(2**31))
            for pf in place_fields_list
        ]
    else:
        trajectory_direction = get_trajectory_direction(position_1ms)
        place_fields_list = []
        spike_trains = []
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for mean in place_field_means:
                pf = simulate_place_field_firing_rate(
                    mean,
                    position_1ms,
                    max_rate=max_rate,
                    variance=place_field_variance,
                    is_condition=is_condition,
                )
                place_fields_list.append(pf)
                spike_trains.append(
                    simulate_calcium_neuron(
                        pf, spike_model=spike_model, random_seed=rng.integers(2**31)
                    )
                )

    place_fields = np.stack(place_fields_list, axis=1)  # (n_time_1ms, n_neurons)
    spike_train_1ms = np.stack(spike_trains, axis=1)      # (n_time_1ms, n_neurons)

    # 3. AR(2) calcium simulation + downsampling
    spikes, calcium_clean, calcium_noisy = simulate_ar2_calcium(
        spike_train_1ms,
        noise_std=noise_std,
        tau_d=tau_d,
        tau_r=tau_r,
        imaging_frequency=imaging_frequency,
        random_seed=rng.integers(2**31),
    )

    # 4. Downsample position and build frame-rate time axis
    k = round(1000.0 / imaging_frequency)
    n_frames = n_samples_1ms // k
    frame_indices = np.arange(n_frames) * k
    position_ds = position_1ms[frame_indices]
    time_ds = time_1ms[frame_indices]

    return time_ds, position_ds, imaging_frequency, spikes, calcium_clean, calcium_noisy, place_fields


# ── Calcium replay wrappers ──────────────────────────────────────────────────


def make_calcium_continuous_replay(
    imaging_frequency: float = IMAGING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = REPLAY_SPEEDUP,
    is_outbound: bool = True,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    noise_std: Union[float, NDArray[np.float64]] = CALCIUM_NOISE_STD,
    random_seed: Optional[int] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate a continuous replay event with calcium imaging.

    Wraps :func:`~replay_trajectory_classification.sorted_spikes_simulation
    .make_continuous_replay` to generate 1 ms spike trains, then applies
    AR(2) calcium dynamics and downsamples to ``imaging_frequency``.

    Parameters
    ----------
    imaging_frequency : float, optional
        Frame rate in Hz.
    track_height : float, optional
        Track length.
    running_speed : float, optional
        Baseline running speed (before speedup).
    place_field_means : NDArray[np.float64], optional
        Place field centres.
    replay_speedup : int, optional
        Multiplier for replay speed relative to running speed.
    is_outbound : bool, optional
        Direction of replay.
    tau_d : float, optional
        AR(2) decay time constant in ms.
    tau_r : float, optional
        AR(2) rise time constant in ms.
    noise_std : float or NDArray[np.float64], optional
        Per-neuron calcium noise standard deviation.
    random_seed : int or None, optional
        Random seed.

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_frames,)
        Time at each imaging frame.
    spikes : NDArray[np.float64], shape (n_frames, n_neurons)
        Ground-truth spike counts per frame.
    calcium_clean : NDArray[np.float64], shape (n_frames, n_neurons)
        Noiseless calcium trace.
    calcium_noisy : NDArray[np.float64], shape (n_frames, n_neurons)
        Noisy calcium trace.

    """
    _, spike_train_1ms = make_continuous_replay(
        sampling_frequency=SAMPLING_FREQUENCY,
        track_height=track_height,
        running_speed=running_speed,
        place_field_means=place_field_means,
        replay_speedup=replay_speedup,
        is_outbound=is_outbound,
    )

    spikes, calcium_clean, calcium_noisy = simulate_ar2_calcium(
        spike_train_1ms,
        noise_std=noise_std,
        tau_d=tau_d,
        tau_r=tau_r,
        imaging_frequency=imaging_frequency,
        random_seed=random_seed,
    )

    n_frames = spikes.shape[0]
    replay_time = np.arange(n_frames) / imaging_frequency

    return replay_time, spikes, calcium_clean, calcium_noisy


def make_calcium_hover_replay(
    imaging_frequency: float = IMAGING_FREQUENCY,
    hover_neuron_ind: Optional[int] = None,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    noise_std: Union[float, NDArray[np.float64]] = CALCIUM_NOISE_STD,
    random_seed: Optional[int] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate a stationary (hover) replay event with calcium imaging.

    Wraps :func:`~replay_trajectory_classification.sorted_spikes_simulation
    .make_hover_replay`.

    Parameters
    ----------
    imaging_frequency : float, optional
        Frame rate in Hz.
    hover_neuron_ind : int or None, optional
        Index of the neuron that fires repeatedly. Defaults to the
        middle neuron.
    place_field_means : NDArray[np.float64], optional
        Place field centres (used to determine number of neurons).
    tau_d : float, optional
        AR(2) decay time constant in ms.
    tau_r : float, optional
        AR(2) rise time constant in ms.
    noise_std : float or NDArray[np.float64], optional
        Per-neuron calcium noise standard deviation.
    random_seed : int or None, optional
        Random seed.

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_frames,)
    spikes : NDArray[np.float64], shape (n_frames, n_neurons)
    calcium_clean : NDArray[np.float64], shape (n_frames, n_neurons)
    calcium_noisy : NDArray[np.float64], shape (n_frames, n_neurons)

    """
    _, spike_train_1ms = make_hover_replay(
        hover_neuron_ind=hover_neuron_ind,
        place_field_means=place_field_means,
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    spikes, calcium_clean, calcium_noisy = simulate_ar2_calcium(
        spike_train_1ms,
        noise_std=noise_std,
        tau_d=tau_d,
        tau_r=tau_r,
        imaging_frequency=imaging_frequency,
        random_seed=random_seed,
    )

    n_frames = spikes.shape[0]
    replay_time = np.arange(n_frames) / imaging_frequency

    return replay_time, spikes, calcium_clean, calcium_noisy


def make_calcium_fragmented_replay(
    imaging_frequency: float = IMAGING_FREQUENCY,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    tau_d: float = TAU_D,
    tau_r: float = TAU_R,
    noise_std: Union[float, NDArray[np.float64]] = CALCIUM_NOISE_STD,
    random_seed: Optional[int] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate a fragmented replay event with calcium imaging.

    Wraps :func:`~replay_trajectory_classification.sorted_spikes_simulation
    .make_fragmented_replay`.

    Parameters
    ----------
    imaging_frequency : float, optional
        Frame rate in Hz.
    place_field_means : NDArray[np.float64], optional
        Place field centres.
    tau_d : float, optional
        AR(2) decay time constant in ms.
    tau_r : float, optional
        AR(2) rise time constant in ms.
    noise_std : float or NDArray[np.float64], optional
        Per-neuron calcium noise standard deviation.
    random_seed : int or None, optional
        Random seed.

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_frames,)
    spikes : NDArray[np.float64], shape (n_frames, n_neurons)
    calcium_clean : NDArray[np.float64], shape (n_frames, n_neurons)
    calcium_noisy : NDArray[np.float64], shape (n_frames, n_neurons)

    """
    _, spike_train_1ms = make_fragmented_replay(
        place_field_means=place_field_means,
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    spikes, calcium_clean, calcium_noisy = simulate_ar2_calcium(
        spike_train_1ms,
        noise_std=noise_std,
        tau_d=tau_d,
        tau_r=tau_r,
        imaging_frequency=imaging_frequency,
        random_seed=random_seed,
    )

    n_frames = spikes.shape[0]
    replay_time = np.arange(n_frames) / imaging_frequency

    return replay_time, spikes, calcium_clean, calcium_noisy
