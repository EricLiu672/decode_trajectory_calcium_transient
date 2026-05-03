# %%
"""Calcium trace simulation, deconvolution, and decoding.

Run this script from the repository root with the ``caiman`` conda environment:

    conda run -n caiman python notebooks/calcium_test/s1_decoding_with_calcium.py

The workflow intentionally mirrors the sorted-spike tutorial notebooks, but it
adds a CAIMAN/OASIS-style front-end:

1. simulate position, place fields, and latent spikes using repo-local helpers,
2. generate calcium fluorescence traces from the latent spikes,
3. deconvolve the fluorescence into inferred spike-like activity,
4. decode/classify from the inferred activity with the existing
   ``SortedSpikesDecoder`` and ``SortedSpikesClassifier``,
5. run an interval-bound sweep that compares latent-spike and calcium-inferred
   decoding as inter-neuron event spacing changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_fscore_support

try:
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
except ImportError as exc:  # pragma: no cover - script is meant for the caiman env
    raise ImportError(
        "Install/use the 'caiman' conda environment before running this script. "
        "The CAIMAN deconvolution function is required."
    ) from exc

try:
    from replay_trajectory_classification.likelihoods.calcium_likelihood import (
        estimate_calcium_place_fields,
    )
except ImportError:  # pragma: no cover - optional exploratory section
    estimate_calcium_place_fields = None
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import (
    estimate_place_fields_kde,
)

from replay_trajectory_classification import (
    Environment,
    Identity,
    RandomWalk,
    SortedSpikesClassifier,
    SortedSpikesDecoder,
    Uniform,
    estimate_movement_var,
)
from replay_trajectory_classification.calcium_interval_analysis import (
    extract_mae_interval_bounds,
    make_evenly_spaced_place_field_means,
    summarize_event_spacing,
)
from replay_trajectory_classification.simulate import (
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
    simulate_position,
    simulate_time,
)
from replay_trajectory_classification.sorted_spikes_simulation import (
    make_continuous_replay,
    make_fragmented_replay,
    make_hover_replay,
)


# %% [markdown]
# ## Global configuration
# %%
sns.set_style("white")
sns.set_context("talk")
plt.ioff()

TRACK_HEIGHT = 180.0
RUNNING_SPEED = 1.0
SAMPLING_FREQUENCY = 30
N_RUNS = 6
# Keep the baseline simulation on a linear track, but tile at least 20 neurons
# across it so the tutorial-style diagnostics stay informative.
DEFAULT_NEURON_COUNT = 20
PLACE_FIELD_VARIANCE = 6.0**2
PLACE_FIELD_MEANS = make_evenly_spaced_place_field_means(
    neuron_count=DEFAULT_NEURON_COUNT,
    track_height=TRACK_HEIGHT,
)
STATE_NAMES = ["continuous", "fragmented", "stationary"]
STATE_COLORS = {
    "continuous": "#521b65",
    "fragmented": "#ff6944",
    "stationary": "#9f043a",
}


@dataclass(frozen=True)
class CalciumTraceConfig:
    """Parameters for the calcium front-end."""

    # g controls the calcium decay; larger values produce slower decays and
    # therefore more overlap between neighboring events.
    g: float = 0.95
    # g_noise jitters that decay across neurons so the population is not
    # unrealistically homogeneous.
    g_noise: float = 0.01
    # Additive Gaussian background noise on the fluorescence trace.
    # This is the main knob to make the simulated recording less idealized.
    sn: float = 20.0
    # Poisson noise approximates photon-counting variability in the readout.
    poiss_noise_factor: float = 0.5
    # baseline offsets the fluorescence trace before adding measurement noise.
    baseline: float = 0.0
    # spike_noise perturbs latent spike amplitudes before calcium filtering.
    spike_noise: float = 0.1
    # Enable this to simulate a saturating GCaMP-like fluorescence transfer
    # function instead of a purely linear observation model.
    nonlinearity: bool = False
    # s_min is the sparsity floor passed to constrained_foopsi.
    s_min: float = 0.01
    # spike_threshold binarizes the continuous OASIS output before the
    # downstream decoder/classifier sees it as spike-like activity.
    spike_threshold: float = 0.10
    # seed keeps the calcium simulation/deconvolution examples reproducible.
    seed: int = 20


@dataclass(frozen=True)
class IntervalSweepConfig:
    """Configuration for the calcium interval-bound experiment."""

    running_speeds: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    neuron_counts: tuple[int, ...] = (10, 20, 40, 80)
    n_repeats: int = 2
    mae_ratio_threshold: float = 1.25
    base_seed: int = 101
    sampling_frequency: int = SAMPLING_FREQUENCY
    n_runs: int = N_RUNS
    track_height: float = TRACK_HEIGHT
    place_field_variance: float = PLACE_FIELD_VARIANCE
    max_rate: float = 12.0
    train_fraction: float = 0.7


def announce(step: str) -> None:
    """Emit a lightweight progress message for script runs."""

    print(f"[calcium_test] {step}", flush=True)

# %% [markdown]
# ## Calcium simulation helpers
#
# `CalciumTraceConfig.sn` controls the additive fluorescence background noise.
# Increase it when you want a more realistic, lower-SNR calcium trace.

# %%
def make_gc6f_nonlinearity(
    model: str = "dana_kim_gc6f",
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Approximate GCaMP nonlinearity from the CAIMAN example script."""

    if model == "dana_kim_gc6f":
        x_orig = np.array([-2, 0.0, 1.0, 2.0, 3.0, 5.0, 10, 20, 40, 160], dtype=float)
        y_orig = np.array(
            [-0.01, 0.0, 0.1, 0.25, 0.5, 0.9, 1.85, 3.6, 4.9, 6.1], dtype=float
        )
    else:
        raise ValueError(f"Unknown nonlinearity model: {model}")

    forward = interp1d(x_orig, y_orig, kind="quadratic", fill_value="extrapolate")
    inverse = interp1d(y_orig, x_orig, kind="quadratic", fill_value="extrapolate")
    return forward, inverse


def simulate_sorted_spike_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    max_rate: float = 12.0,
) -> dict[str, np.ndarray | float]:
    """Simulate linear-track spikes using the repo's low-level simulation logic."""

    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)
    time = simulate_time(n_samples, sampling_frequency)
    position = simulate_position(time, track_height, running_speed)

    place_fields = np.stack(
        [
            simulate_place_field_firing_rate(
                place_field_mean,
                position,
                max_rate=max_rate,
                variance=place_field_variance,
            )
            for place_field_mean in place_field_means
        ],
        axis=1,
    )
    spikes = np.stack(
        [
            simulate_neuron_with_place_field(
                place_field_mean,
                position,
                max_rate=max_rate,
                variance=place_field_variance,
                sampling_frequency=sampling_frequency,
            )
            for place_field_mean in place_field_means
        ],
        axis=1,
    )

    return {
        "time": time,
        "position": position,
        "sampling_frequency": float(sampling_frequency),
        "track_height": float(track_height),
        "running_speed": float(running_speed),
        "spikes": spikes.astype(float),
        "place_fields": place_fields,
        "place_field_means": np.asarray(place_field_means, dtype=float),
        "place_field_variance": float(place_field_variance),
        "max_rate": float(max_rate),
    }


def simulate_calcium_traces_from_spikes(
    true_spikes: np.ndarray,
    config: CalciumTraceConfig,
) -> dict[str, np.ndarray]:
    """Generate latent calcium and fluorescence traces from spike events."""

    spikes = np.asarray(true_spikes, dtype=float)
    rng = np.random.default_rng(config.seed)
    n_time, n_neurons = spikes.shape

    noisy_events = spikes.copy()
    spike_mask = noisy_events > 0
    noisy_events[spike_mask] += rng.normal(
        0.0,
        config.spike_noise,
        size=int(spike_mask.sum()),
    )
    noisy_events = np.clip(noisy_events, 0.0, None)

    g = np.full((n_neurons,), config.g, dtype=float)
    g += config.g_noise * 2.0 * (rng.random(n_neurons) - 0.5)
    g = np.clip(g, 0.01, 0.999)

    calcium_state = np.zeros_like(noisy_events)
    calcium_state[0] = noisy_events[0]
    for time_ind in range(1, n_time):
        calcium_state[time_ind] = noisy_events[time_ind] + g * calcium_state[time_ind - 1]

    clean_fluorescence = config.baseline + calcium_state
    if config.nonlinearity:
        nonlinearity, _ = make_gc6f_nonlinearity()
        transformed_signal = nonlinearity(clean_fluorescence)
    else:
        transformed_signal = clean_fluorescence.copy()

    signal_min = transformed_signal.min(axis=0, keepdims=True)
    signal_range = np.ptp(transformed_signal, axis=0, keepdims=True)
    signal_range[signal_range == 0.0] = 1.0
    scaled_signal = (transformed_signal - signal_min) / signal_range

    poisson_noise = (
        rng.poisson(np.clip(scaled_signal * 1000.0, 0.0, None)) / 1000.0
    ) * config.poiss_noise_factor
    fluorescence = (
        transformed_signal
        + config.sn * rng.standard_normal(size=transformed_signal.shape)
        + poisson_noise
    )

    return {
        "spikes": spikes,
        "events": noisy_events,
        "calcium_state": calcium_state,
        "clean_fluorescence": transformed_signal,
        "fluorescence": fluorescence,
        "g": g,
    }


def deconvolve_calcium_traces(
    fluorescence: np.ndarray,
    p: int = 1,
    s_min: float = 0.01,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Run CAIMAN/OASIS for each neuron and return trace + spike outputs."""

    fluorescence = np.asarray(fluorescence, dtype=float)
    n_time, n_neurons = fluorescence.shape

    deconvolved_calcium = np.zeros_like(fluorescence)
    inferred_spikes = np.zeros_like(fluorescence)
    baselines = np.zeros((n_neurons,), dtype=float)
    estimated_noise = np.zeros((n_neurons,), dtype=float)
    estimated_g: list[np.ndarray] = []

    for neuron_ind in range(n_neurons):
        c, bl, _c1, g, sn, s, _lam = constrained_foopsi(
            fluorescence[:, neuron_ind], p=p, s_min=s_min
        )
        deconvolved_calcium[:, neuron_ind] = bl + c
        inferred_spikes[:, neuron_ind] = np.clip(np.asarray(s), 0.0, None)
        baselines[neuron_ind] = bl
        estimated_noise[neuron_ind] = sn
        estimated_g.append(np.atleast_1d(g))

    return {
        "deconvolved_calcium": deconvolved_calcium,
        "inferred_spikes": inferred_spikes,
        "baseline": baselines,
        "estimated_noise": estimated_noise,
        "estimated_g": estimated_g,
    }


def binarize_inferred_spikes(
    inferred_spikes: np.ndarray, threshold: float
) -> np.ndarray:
    """Convert continuous deconvolved activity to spike indicators/counts."""

    return (np.asarray(inferred_spikes) > threshold).astype(float)


def split_run_data(
    time: np.ndarray,
    position: np.ndarray,
    spikes: np.ndarray,
    train_fraction: float = 0.7,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a simulated run into contiguous train/test segments."""

    split_ind = int(train_fraction * len(time))
    train = {
        "time": time[:split_ind],
        "position": position[:split_ind],
        "spikes": spikes[:split_ind],
    }
    test = {
        "time": time[split_ind:],
        "position": position[split_ind:],
        "spikes": spikes[split_ind:],
    }
    return train, test


# %% [markdown]
# ## Decoder / classifier helpers
# %%
def fit_sorted_spike_decoder(
    position: np.ndarray,
    spikes: np.ndarray,
    sampling_frequency: float,
) -> SortedSpikesDecoder:
    """Fit the standard sorted-spike decoder from tutorial 02."""

    # Match tutorial 02: estimate the movement variance from the running
    # trajectory, use its standard deviation as the 1D place-bin size, and
    # reuse the variance inside the random-walk transition model.
    movement_var = float(np.squeeze(estimate_movement_var(position, sampling_frequency)))
    movement_var = max(movement_var, 1e-3)

    environment = Environment(place_bin_size=np.sqrt(movement_var))
    transition_type = RandomWalk(movement_var=movement_var)
    decoder = SortedSpikesDecoder(
        environment=environment,
        transition_type=transition_type,
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={
            # Use all neurons together instead of chunking them into blocks.
            "block_size": None,
            # position_std controls the smoothness of the fitted place fields.
            "position_std": [3.0],
            # Keep diffusion off so the fit stays aligned with tutorial 02.
            "use_diffusion": False,
        },
    )
    decoder.fit(position, spikes)
    return decoder


def fit_sorted_spike_classifier(
    position: np.ndarray,
    spikes: np.ndarray,
    sampling_frequency: float,
) -> SortedSpikesClassifier:
    """Fit the standard sorted-spike classifier from tutorial 04."""

    movement_var = float(np.squeeze(estimate_movement_var(position, sampling_frequency)))
    movement_var = max(movement_var, 1e-3)

    environment = Environment(place_bin_size=np.sqrt(movement_var))
    continuous_transition_types = [
        [RandomWalk(movement_var=movement_var * 120), Uniform(), Identity()],
        [Uniform(), Uniform(), Uniform()],
        [RandomWalk(movement_var=movement_var * 120), Uniform(), Identity()],
    ]
    classifier = SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types=continuous_transition_types,
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={"position_std": 3.0},
    )
    classifier.fit(position, spikes)
    return classifier


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a finite correlation coefficient when possible."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_spike_recovery_metrics(
    true_spikes: np.ndarray,
    inferred_spikes_continuous: np.ndarray,
    inferred_spikes_binary: np.ndarray,
) -> dict[str, float]:
    """Summarize deconvolution quality against the latent spikes."""

    y_true = np.asarray(true_spikes).ravel().astype(int)
    y_pred = np.asarray(inferred_spikes_binary).ravel().astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "continuous_corr": safe_correlation(true_spikes, inferred_spikes_continuous),
        "binary_corr": safe_correlation(true_spikes, inferred_spikes_binary),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_latent_and_inferred_decoding(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    sampling_frequency: float,
    calcium_config: CalciumTraceConfig,
) -> dict[str, object]:
    """Compare latent-spike decoding with calcium-inferred decoding."""

    latent_decoder = fit_sorted_spike_decoder(
        train["position"],
        train["spikes"],
        sampling_frequency=sampling_frequency,
    )
    latent_results, latent_metrics = evaluate_decoder(
        latent_decoder,
        test["spikes"],
        test["position"],
        test["time"],
    )

    train_calcium = simulate_calcium_traces_from_spikes(train["spikes"], calcium_config)
    train_deconv = deconvolve_calcium_traces(
        train_calcium["fluorescence"],
        s_min=calcium_config.s_min,
    )
    train_binary = binarize_inferred_spikes(
        train_deconv["inferred_spikes"],
        threshold=calcium_config.spike_threshold,
    )

    test_calcium_config = replace(calcium_config, seed=calcium_config.seed + 1)
    test_calcium = simulate_calcium_traces_from_spikes(test["spikes"], test_calcium_config)
    test_deconv = deconvolve_calcium_traces(
        test_calcium["fluorescence"],
        s_min=calcium_config.s_min,
    )
    test_binary = binarize_inferred_spikes(
        test_deconv["inferred_spikes"],
        threshold=calcium_config.spike_threshold,
    )

    inferred_decoder = fit_sorted_spike_decoder(
        train["position"],
        train_binary,
        sampling_frequency=sampling_frequency,
    )
    inferred_results, inferred_metrics = evaluate_decoder(
        inferred_decoder,
        test_binary,
        test["position"],
        test["time"],
    )
    spike_metrics = compute_spike_recovery_metrics(
        test["spikes"],
        test_deconv["inferred_spikes"],
        test_binary,
    )

    latent_mae = max(latent_metrics["mae_cm"], np.finfo(float).eps)
    degradation_metrics = {
        "mae_delta_cm": inferred_metrics["mae_cm"] - latent_metrics["mae_cm"],
        "mae_ratio": inferred_metrics["mae_cm"] / latent_mae,
        "corr_delta": inferred_metrics["corr"] - latent_metrics["corr"],
    }

    return {
        "latent_results": latent_results,
        "latent_metrics": latent_metrics,
        "train_calcium": train_calcium,
        "train_deconv": train_deconv,
        "train_binary": train_binary,
        "test_calcium": test_calcium,
        "test_deconv": test_deconv,
        "test_binary": test_binary,
        "inferred_results": inferred_results,
        "inferred_metrics": inferred_metrics,
        "spike_metrics": spike_metrics,
        "degradation_metrics": degradation_metrics,
    }


def get_position_posterior(
    results,
    posterior_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a normalized position posterior and its position bins."""

    posterior = results[posterior_name].fillna(0.0)
    if "state" in posterior.dims:
        posterior = posterior.sum("state")

    posterior_array = np.asarray(posterior.to_numpy(), dtype=float)
    if posterior_array.ndim != 2:
        raise ValueError(
            f"{posterior_name} must be 2D after any state marginalization; "
            f"got shape {posterior_array.shape}."
        )

    row_sums = posterior_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    posterior_array = posterior_array / row_sums
    position_bins = np.asarray(posterior.position.to_numpy(), dtype=float)
    return posterior_array, position_bins


def map_position_from_results(
    results,
    posterior_name: str | None = None,
) -> np.ndarray:
    """Extract a MAP trajectory from decoder results."""

    if posterior_name is None:
        posterior_name = (
            "acausal_posterior"
            if "acausal_posterior" in results.data_vars
            else "causal_posterior"
        )

    posterior, position_bins = get_position_posterior(results, posterior_name)
    max_inds = np.nanargmax(posterior, axis=1)
    return position_bins[max_inds]


def summarize_decoder_posteriors(
    results,
    true_position: np.ndarray,
    decoder_label: str,
) -> pd.DataFrame:
    """Summarize posterior sharpness and MAP accuracy for manual inspection."""

    records = []
    for posterior_name in ("causal_posterior", "acausal_posterior"):
        if posterior_name not in results.data_vars:
            continue

        posterior, position_bins = get_position_posterior(results, posterior_name)
        map_position = map_position_from_results(results, posterior_name=posterior_name)
        posterior_mean = posterior @ position_bins
        posterior_sd = np.sqrt(
            np.sum(
                posterior * (position_bins[np.newaxis, :] - posterior_mean[:, np.newaxis]) ** 2,
                axis=1,
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_bits = -np.sum(
                np.where(posterior > 0.0, posterior * np.log2(posterior), 0.0),
                axis=1,
            )

        peak_probability = posterior.max(axis=1)
        records.append(
            {
                "decoder": decoder_label,
                "posterior": posterior_name.replace("_posterior", ""),
                "map_mae_cm": float(np.mean(np.abs(map_position - true_position))),
                "mean_peak_probability": float(np.mean(peak_probability)),
                "median_peak_probability": float(np.median(peak_probability)),
                "mean_posterior_sd_cm": float(np.mean(posterior_sd)),
                "mean_entropy_bits": float(np.mean(entropy_bits)),
            }
        )

    return pd.DataFrame.from_records(records)


def evaluate_decoder(
    decoder: SortedSpikesDecoder,
    spikes: np.ndarray,
    true_position: np.ndarray,
    time: np.ndarray,
) -> tuple[object, dict[str, float]]:
    """Run prediction and compute 1D decoding metrics."""

    results = decoder.predict(spikes, time=time)
    decoded_position = map_position_from_results(results)
    metrics = {
        "mae_cm": float(np.mean(np.abs(decoded_position - true_position))),
        "corr": safe_correlation(decoded_position, true_position),
    }
    return results, metrics


def summarize_state_probabilities(results) -> pd.Series:
    """Average state probabilities over time."""

    probabilities = results.acausal_posterior.sum("position")
    mean_probabilities = probabilities.mean("time").to_series()
    mean_probabilities.index = mean_probabilities.index.astype(str)
    return mean_probabilities


def evaluate_classification_scenarios(
    classifier: SortedSpikesClassifier,
    calcium_config: CalciumTraceConfig,
    scenario_builders: list[tuple[str, str, Callable[[], tuple[np.ndarray, np.ndarray]]]],
) -> pd.DataFrame:
    """Pass replay scenarios through the calcium front-end and classify them."""

    records = []
    for scenario_ind, (scenario_name, expected_state, builder) in enumerate(scenario_builders):
        announce(f"classifying scenario: {scenario_name}")
        replay_time, latent_spikes = builder()
        scenario_config = replace(calcium_config, seed=calcium_config.seed + 100 + scenario_ind)
        calcium = simulate_calcium_traces_from_spikes(latent_spikes, scenario_config)
        deconv = deconvolve_calcium_traces(
            calcium["fluorescence"],
            s_min=scenario_config.s_min,
        )
        inferred_binary = binarize_inferred_spikes(
            deconv["inferred_spikes"],
            threshold=scenario_config.spike_threshold,
        )
        results = classifier.predict(
            inferred_binary,
            time=replay_time,
            state_names=STATE_NAMES,
        )
        state_probabilities = summarize_state_probabilities(results)
        predicted_state = state_probabilities.idxmax()
        records.append(
            {
                "scenario": scenario_name,
                "expected_state": expected_state,
                "predicted_state": predicted_state,
                "expected_probability": float(state_probabilities[expected_state]),
                "state_probabilities": state_probabilities.reindex(STATE_NAMES, fill_value=0.0),
                "results": results,
                "replay_time": replay_time,
                "inferred_binary": inferred_binary,
            }
        )

    return pd.DataFrame(records)


def evaluate_parameter_sweep(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    sampling_frequency: float,
    configs: dict[str, CalciumTraceConfig],
) -> pd.DataFrame:
    """Quantify how calcium/deconvolution parameters affect decoding."""

    records = []
    for label, config in configs.items():
        announce(f"parameter sweep: {label}")
        comparison = evaluate_latent_and_inferred_decoding(
            train=train,
            test=test,
            sampling_frequency=sampling_frequency,
            calcium_config=config,
        )

        records.append(
            {
                "config": label,
                "g": config.g,
                "sn": config.sn,
                "poiss_noise_factor": config.poiss_noise_factor,
                "nonlinearity": config.nonlinearity,
                "s_min": config.s_min,
                "spike_threshold": config.spike_threshold,
                "latent_mae_cm": comparison["latent_metrics"]["mae_cm"],
                "latent_corr": comparison["latent_metrics"]["corr"],
                "mae_cm": comparison["inferred_metrics"]["mae_cm"],
                "corr": comparison["inferred_metrics"]["corr"],
                **comparison["spike_metrics"],
                **comparison["degradation_metrics"],
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_interval_sweep_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeat-wise interval-sweep results into condition summaries."""

    if results_df.empty:
        return pd.DataFrame()

    group_columns = [
        "neuron_count",
        "running_speed",
        "sampling_frequency",
        "place_field_spacing_cm",
        "expected_neighbor_interval_s",
        "expected_neighbor_interval_frames",
    ]
    summary_df = (
        results_df.groupby(group_columns, as_index=False)
        .agg(
            n_repeats=("repeat", "count"),
            mean_latent_mae_cm=("latent_mae_cm", "mean"),
            mean_inferred_mae_cm=("inferred_mae_cm", "mean"),
            std_inferred_mae_cm=("inferred_mae_cm", "std"),
            mean_latent_corr=("latent_corr", "mean"),
            mean_inferred_corr=("inferred_corr", "mean"),
            mean_mae_delta_cm=("mae_delta_cm", "mean"),
            mean_mae_ratio=("mae_ratio", "mean"),
            mean_corr_delta=("corr_delta", "mean"),
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_observed_cross_neuron_interval_s_median=(
                "observed_cross_neuron_interval_s_median",
                "mean",
            ),
            mean_observed_cross_neuron_interval_s_min=(
                "observed_cross_neuron_interval_s_min",
                "mean",
            ),
            mean_multi_neuron_event_bin_fraction=(
                "multi_neuron_event_bin_fraction",
                "mean",
            ),
        )
        .sort_values(["neuron_count", "expected_neighbor_interval_s"])
        .reset_index(drop=True)
    )
    return summary_df


def evaluate_interval_bound_sweep(
    calcium_config: CalciumTraceConfig,
    sweep_config: IntervalSweepConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sweep spacing-sensitive parameters and estimate the calcium time bound."""

    records = []
    for neuron_count in sweep_config.neuron_counts:
        place_field_means = make_evenly_spaced_place_field_means(
            neuron_count=neuron_count,
            track_height=sweep_config.track_height,
        )
        for running_speed in sweep_config.running_speeds:
            announce(
                "interval sweep: "
                f"{neuron_count} neurons @ {running_speed:.1f} cm/s"
            )
            for repeat in range(sweep_config.n_repeats):
                condition_seed = (
                    sweep_config.base_seed
                    + repeat
                    + neuron_count * 100
                    + int(round(running_speed * 10))
                )
                np.random.seed(condition_seed)
                run_data = simulate_sorted_spike_run_data(
                    sampling_frequency=sweep_config.sampling_frequency,
                    track_height=sweep_config.track_height,
                    running_speed=running_speed,
                    n_runs=sweep_config.n_runs,
                    place_field_variance=sweep_config.place_field_variance,
                    place_field_means=place_field_means,
                    max_rate=sweep_config.max_rate,
                )
                train_data, test_data = split_run_data(
                    time=np.asarray(run_data["time"]),
                    position=np.asarray(run_data["position"]),
                    spikes=np.asarray(run_data["spikes"]),
                    train_fraction=sweep_config.train_fraction,
                )
                spacing_metrics = summarize_event_spacing(
                    test_data["spikes"],
                    sampling_frequency=float(sweep_config.sampling_frequency),
                    place_field_means=place_field_means,
                    running_speed=running_speed,
                )
                comparison = evaluate_latent_and_inferred_decoding(
                    train=train_data,
                    test=test_data,
                    sampling_frequency=float(sweep_config.sampling_frequency),
                    calcium_config=replace(
                        calcium_config,
                        seed=calcium_config.seed + condition_seed,
                    ),
                )

                records.append(
                    {
                        "repeat": repeat,
                        "seed": condition_seed,
                        "neuron_count": neuron_count,
                        "running_speed": running_speed,
                        "sampling_frequency": sweep_config.sampling_frequency,
                        "latent_mae_cm": comparison["latent_metrics"]["mae_cm"],
                        "latent_corr": comparison["latent_metrics"]["corr"],
                        "inferred_mae_cm": comparison["inferred_metrics"]["mae_cm"],
                        "inferred_corr": comparison["inferred_metrics"]["corr"],
                        **comparison["degradation_metrics"],
                        **comparison["spike_metrics"],
                        **spacing_metrics,
                    }
                )

    results_df = pd.DataFrame.from_records(records)
    summary_df = summarize_interval_sweep_results(results_df)
    bound_df = extract_mae_interval_bounds(
        summary_df,
        slice_columns=("neuron_count", "sampling_frequency"),
        threshold=sweep_config.mae_ratio_threshold,
    ).sort_values(["neuron_count", "sampling_frequency"])
    return results_df, summary_df, bound_df


# %% [markdown]
# ## Plotting helpers
# %%
def plot_training_data(
    data: dict[str, np.ndarray | float],
    spike_matrix: np.ndarray,
    title_prefix: str,
) -> None:
    """Plot place fields, position, and a spike raster."""

    time = np.asarray(data["time"])
    position = np.asarray(data["position"])
    place_fields = np.asarray(data["place_fields"])
    spike_ind, neuron_ind = np.nonzero(spike_matrix)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    for neuron_ind_plot in range(place_fields.shape[1]):
        axes[0].plot(position, place_fields[:, neuron_ind_plot], alpha=0.35)
    axes[0].set_title(f"{title_prefix}: place fields")
    axes[0].set_ylabel("Rate / activity")

    axes[1].plot(time, position, linewidth=2)
    axes[1].set_title(f"{title_prefix}: position")
    axes[1].set_ylabel("Position [cm]")

    axes[2].scatter(time[spike_ind], neuron_ind + 1, s=4, color="black")
    axes[2].set_title(f"{title_prefix}: spike raster")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Neuron index")
    plt.show()


def get_example_neuron_ids(
    n_neurons: int,
    n_examples: int = 3,
) -> tuple[int, ...]:
    """Choose representative neurons spread across the linear track."""

    if n_neurons <= 0:
        return ()

    n_examples = min(n_examples, n_neurons)
    neuron_ids = np.linspace(0, n_neurons - 1, n_examples, dtype=int)
    return tuple(dict.fromkeys(int(neuron_ind) for neuron_ind in neuron_ids))


def plot_calcium_pipeline(
    time: np.ndarray,
    calcium: dict[str, np.ndarray],
    deconv: dict[str, np.ndarray | list[np.ndarray]],
    neuron_ids: tuple[int, ...] = (0, 5, 10),
) -> None:
    """Visualize fluorescence, deconvolution, and spike-event relationships."""

    unique_neuron_ids = tuple(dict.fromkeys(int(neuron_ind) for neuron_ind in neuron_ids))
    n_rows_per_neuron = 4
    fig, axes = plt.subplots(
        len(unique_neuron_ids) * n_rows_per_neuron,
        1,
        figsize=(14, 2.0 * len(unique_neuron_ids) * n_rows_per_neuron),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    axes = np.atleast_1d(axes)

    row_labels = (
        "fluorescence",
        "inferred spikes",
        "noisy events",
        "latent spikes",
    )
    row_colors = {
        "latent spikes": "black",
        "noisy events": "#d94801",
        "clean fluorescence": "#4c2a85",
        "noisy fluorescence": "#00a2ff",
        "denoised calcium": "#31a354",
        "inferred spikes": "#238b45",
    }

    for neuron_plot_ind, neuron_ind in enumerate(unique_neuron_ids):
        neuron_axes = axes[
            neuron_plot_ind * n_rows_per_neuron : (neuron_plot_ind + 1) * n_rows_per_neuron
        ]

        neuron_axes[0].plot(
            time,
            calcium["clean_fluorescence"][:, neuron_ind],
            color=row_colors["clean fluorescence"],
            linewidth=1.9,
            label="clean fluorescence",
        )
        neuron_axes[0].plot(
            time,
            calcium["fluorescence"][:, neuron_ind],
            color=row_colors["noisy fluorescence"],
            linewidth=1.1,
            alpha=0.8,
            label="noisy fluorescence",
        )
        neuron_axes[0].plot(
            time,
            deconv["deconvolved_calcium"][:, neuron_ind],
            color=row_colors["denoised calcium"],
            linewidth=1.5,
            label="denoised calcium",
        )

        neuron_axes[1].step(
            time,
            deconv["inferred_spikes"][:, neuron_ind],
            where="mid",
            color=row_colors["inferred spikes"],
            linewidth=1.1,
        )

        neuron_axes[2].step(
            time,
            calcium["events"][:, neuron_ind],
            where="mid",
            color=row_colors["noisy events"],
            linewidth=1.0,
        )

        neuron_axes[3].step(
            time,
            calcium["spikes"][:, neuron_ind],
            where="mid",
            color=row_colors["latent spikes"],
            linewidth=1.0,
        )

        neuron_axes[0].text(
            0.0,
            0.5,
            f"Neuron {neuron_ind}",
            transform=neuron_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        # if neuron_plot_ind > 0:
        #     neuron_axes[0].plot(
        #         [0.0, 1.0],
        #         [0.8, 0.8],
        #         transform=neuron_axes[0].transAxes,
        #         color="#7f7f7f",
        #         linewidth=1.0,
        #         clip_on=False,
        #     )

        for ax, row_label in zip(neuron_axes, row_labels):
            ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=42)
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.margins(x=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        if neuron_plot_ind == 0:
            neuron_axes[0].legend(frameon=False, loc="upper right", ncol=1)

    axes[-1].set_xlabel("Time [s]")
    plt.show()


def make_true_place_fields_on_grid(
    position_grid: np.ndarray,
    place_field_means: np.ndarray,
    place_field_variance: float,
    max_rate: float,
) -> np.ndarray:
    """Evaluate the known simulated place fields on a requested position grid."""

    return np.stack(
        [
            simulate_place_field_firing_rate(
                place_field_mean,
                position_grid,
                max_rate=max_rate,
                variance=place_field_variance,
            )
            for place_field_mean in np.asarray(place_field_means, dtype=float)
        ],
        axis=1,
    )


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    """Scale each neuron to unit peak for shape-only comparisons."""

    values = np.asarray(values, dtype=float)
    column_max = np.max(values, axis=0, keepdims=True)
    column_max[column_max == 0.0] = 1.0
    return values / column_max


def plot_place_field_comparison(
    estimated_place_fields,
    *,
    place_field_means: np.ndarray,
    place_field_variance: float,
    max_rate: float,
    sampling_frequency: float,
    title: str,
    normalize: bool = False,
) -> None:
    """Overlay fitted and true place fields in the tutorial-02 style."""

    position_grid = np.asarray(estimated_place_fields.position.to_numpy(), dtype=float)
    estimated = np.asarray(estimated_place_fields.to_numpy(), dtype=float)
    true_place_fields = make_true_place_fields_on_grid(
        position_grid=position_grid,
        place_field_means=place_field_means,
        place_field_variance=place_field_variance,
        max_rate=max_rate,
    )

    if normalize:
        estimated_to_plot = _normalize_columns(estimated)
        true_to_plot = _normalize_columns(true_place_fields)
        y_label = "Normalized activity"
    else:
        estimated_to_plot = estimated * sampling_frequency
        true_to_plot = true_place_fields
        y_label = "Firing Rate"

    n_neurons = estimated_to_plot.shape[1]
    n_cols = min(5, n_neurons)
    n_rows = int(np.ceil(n_neurons / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for neuron_ind, ax in enumerate(axes):
        if neuron_ind >= n_neurons:
            ax.axis("off")
            continue

        ax.plot(
            position_grid,
            estimated_to_plot[:, neuron_ind],
            color="red",
            linewidth=2,
            alpha=0.9,
            label="Predicted",
        )
        ax.plot(
            position_grid,
            true_to_plot[:, neuron_ind],
            color="black",
            linewidth=2,
            label="True",
        )
        ax.set_title(f"Neuron {neuron_ind + 1}")
        sns.despine(ax=ax)

    axes[0].legend(frameon=False, loc="upper right")
    for ax in axes[::n_cols]:
        ax.set_ylabel(y_label)
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Position [cm]")
    fig.suptitle(title)
    plt.show()


def plot_spike_recovery_validation(
    time: np.ndarray,
    true_spikes: np.ndarray,
    inferred_spikes_continuous: np.ndarray,
    inferred_spikes_binary: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Validate how well deconvolution recovers spike timing and magnitude."""

    true_spikes = np.asarray(true_spikes)
    inferred_spikes_continuous = np.asarray(inferred_spikes_continuous)
    inferred_spikes_binary = np.asarray(inferred_spikes_binary)

    true_rate = true_spikes.sum(axis=1)
    inferred_rate = inferred_spikes_binary.sum(axis=1)
    continuous_rate = inferred_spikes_continuous.sum(axis=1)
    true_counts = true_spikes.sum(axis=0)
    inferred_counts = inferred_spikes_binary.sum(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    axes[0, 0].plot(time, true_rate, label="latent spikes", linewidth=2)
    axes[0, 0].plot(time, continuous_rate, label="inferred continuous", alpha=0.8)
    axes[0, 0].plot(time, inferred_rate, label="inferred binary", alpha=0.8)
    axes[0, 0].set_title("Population activity over time")
    axes[0, 0].set_ylabel("Population activity")
    axes[0, 0].legend()

    max_count = max(float(np.max(true_counts)), float(np.max(inferred_counts)), 1.0)
    axes[0, 1].scatter(true_counts, inferred_counts, color="#521b65", alpha=0.8)
    axes[0, 1].plot([0, max_count], [0, max_count], "--", color="black", linewidth=1)
    axes[0, 1].set_title("Per-neuron recovered event counts")
    axes[0, 1].set_xlabel("Latent spike count")
    axes[0, 1].set_ylabel("Inferred binary count")

    event_values = inferred_spikes_continuous[true_spikes > 0]
    noise_values = inferred_spikes_continuous[true_spikes == 0]
    if event_values.size:
        sns.histplot(
            event_values,
            ax=axes[1, 0],
            color="#ff6944",
            label="true spike bins",
            stat="density",
            bins=30,
            alpha=0.6,
        )
    if noise_values.size:
        sns.histplot(
            noise_values,
            ax=axes[1, 0],
            color="#61c5e6",
            label="non-spike bins",
            stat="density",
            bins=30,
            alpha=0.5,
        )
    axes[1, 0].set_title("Deconvolved amplitude separation")
    axes[1, 0].set_xlabel("Inferred spike amplitude")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    metric_text = "\n".join(
        [
            "Spike recovery metrics",
            f"continuous corr: {metrics['continuous_corr']:.3f}",
            f"binary corr: {metrics['binary_corr']:.3f}",
            f"precision: {metrics['precision']:.3f}",
            f"recall: {metrics['recall']:.3f}",
            f"F1: {metrics['f1']:.3f}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=12)
    plt.show()


def plot_state_transition_matrix(
    decoder: SortedSpikesDecoder,
    title: str,
) -> None:
    """Visualize the decoder's 1D random-walk transition model."""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    edge1, edge2 = np.meshgrid(
        decoder.environment.place_bin_edges_,
        decoder.environment.place_bin_edges_,
    )
    vmax = np.percentile(decoder.state_transition_, 99.9)
    ax.pcolormesh(edge1, edge2, decoder.state_transition_.T, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel("Position at time t-1 [cm]")
    ax.set_xlabel("Position at time t [cm]")
    ax.axis("square")
    plt.show()


def plot_posterior_diagnostics(
    time: np.ndarray,
    spikes: np.ndarray,
    true_position: np.ndarray,
    results,
    title_prefix: str,
) -> None:
    """Visualize activity plus causal/acausal posteriors like tutorial 02."""

    fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(15, 8))

    spike_ind, neuron_ind = np.nonzero(spikes)
    axes[0].scatter(
        time[spike_ind],
        neuron_ind + 1,
        color="black",
        s=6,
        clip_on=False,
    )
    axes[0].set_yticks((1, spikes.shape[1]))
    axes[0].set_ylabel("Cells")
    axes[0].set_title(f"{title_prefix}: inferred spike raster")

    for ax, posterior_name, panel_title in zip(
        axes[1:],
        ("causal_posterior", "acausal_posterior"),
        ("Causal posterior", "Acausal posterior"),
    ):
        results[posterior_name].plot(
            x="time",
            y="position",
            ax=ax,
            cmap="bone_r",
            vmin=0.0,
            robust=True,
            clip_on=False,
        )
        ax.plot(
            time,
            true_position,
            color="magenta",
            linestyle="--",
            linewidth=2.5,
            clip_on=False,
        )
        ax.set_ylabel("Position [cm]")
        ax.set_title(f"{title_prefix}: {panel_title}")

    axes[1].set_xlabel("")
    axes[2].set_xlabel("Time [s]")
    sns.despine(offset=5)
    plt.show()


def plot_decoder_results(
    time: np.ndarray,
    true_position: np.ndarray,
    inferred_spikes: np.ndarray,
    inferred_decoder: SortedSpikesDecoder,
    baseline_results,
    inferred_results,
    baseline_metrics: dict[str, float],
    inferred_metrics: dict[str, float],
) -> None:
    """Compare MAP trajectories, transition dynamics, and posterior heatmaps."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
    baseline_map = map_position_from_results(baseline_results)
    inferred_map = map_position_from_results(inferred_results)

    ax.plot(time, true_position, label="true position", linewidth=3)
    ax.plot(time, baseline_map, label="latent spike decode", alpha=0.9)
    ax.plot(time, inferred_map, label="inferred spike decode", alpha=0.9)
    ax.set_ylabel("Position [cm]")
    ax.set_xlabel("Time [s]")
    ax.set_title(
        "MAP decoding comparison\n"
        f"latent MAE={baseline_metrics['mae_cm']:.2f} cm, "
        f"inferred MAE={inferred_metrics['mae_cm']:.2f} cm"
    )
    ax.legend()
    plt.show()

    plot_state_transition_matrix(
        inferred_decoder,
        title="Random Walk State Transition Matrix (calcium-driven decoder)",
    )
    plot_posterior_diagnostics(
        time=time,
        spikes=inferred_spikes,
        true_position=true_position,
        results=inferred_results,
        title_prefix="Calcium-driven decoder",
    )


def plot_classification(
    replay_time: np.ndarray,
    test_spikes: np.ndarray,
    results,
    title: str,
) -> None:
    """Plot replay classification results following tutorial 04."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    axes[0].scatter(
        replay_time[spike_time_ind],
        neuron_ind,
        color="black",
        marker="|",
        s=100,
        linewidth=2,
    )
    axes[0].set_ylabel("Neuron index")
    axes[0].set_title(title)

    replay_probability = results.acausal_posterior.sum("position")
    for state, prob in replay_probability.groupby("state"):
        axes[1].plot(
            prob.time,
            prob.values,
            linewidth=3,
            label=str(state),
            color=STATE_COLORS[str(state)],
        )
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim((-0.01, 1.05))
    axes[1].legend(frameon=False, loc="upper right")

    results.acausal_posterior.sum("state").plot(
        x="time", y="position", robust=True, vmin=0.0, ax=axes[2]
    )
    axes[2].set_ylabel("Position [cm]")
    axes[2].set_xlabel("Time [s]")
    plt.show()


def plot_classification_summary(classification_df: pd.DataFrame) -> None:
    """Show state probabilities for every replay scenario."""

    probability_records = []
    for _, row in classification_df.iterrows():
        for state, probability in row["state_probabilities"].items():
            probability_records.append(
                {
                    "scenario": row["scenario"],
                    "state": state,
                    "probability": probability,
                }
            )
    probability_df = pd.DataFrame(probability_records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    sns.barplot(
        data=probability_df,
        x="scenario",
        y="probability",
        hue="state",
        palette=STATE_COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Average state probability by scenario")
    axes[0].set_ylabel("Probability")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, title="state")

    summary_df = classification_df[
        ["scenario", "expected_state", "predicted_state", "expected_probability"]
    ].copy()
    summary_df["correct"] = summary_df["expected_state"] == summary_df["predicted_state"]
    sns.barplot(
        data=summary_df,
        x="scenario",
        y="expected_probability",
        hue="correct",
        palette={True: "#2ca25f", False: "#de2d26"},
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_title("Probability assigned to the expected state")
    axes[1].set_ylabel("Expected-state probability")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, title="correct")
    plt.show()


def plot_parameter_sweep_results(results_df: pd.DataFrame) -> None:
    """Visualize the sweep summary."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.barplot(data=results_df, x="config", y="mae_cm", ax=axes[0], color="#521b65")
    axes[0].set_title("Decoder MAE")
    axes[0].set_ylabel("MAE [cm]")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=results_df, x="config", y="f1", ax=axes[1], color="#ff6944")
    axes[1].set_title("Spike recovery F1")
    axes[1].set_ylabel("F1")
    axes[1].tick_params(axis="x", rotation=30)

    sns.barplot(
        data=results_df,
        x="config",
        y="continuous_corr",
        ax=axes[2],
        color="#61c5e6",
    )
    axes[2].set_title("Continuous spike correlation")
    axes[2].set_ylabel("Correlation")
    axes[2].tick_params(axis="x", rotation=30)
    plt.show()


def plot_interval_bound_results(
    summary_df: pd.DataFrame,
    bound_df: pd.DataFrame,
    mae_ratio_threshold: float,
) -> None:
    """Visualize interval spacing, decoder degradation, and derived bounds."""

    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["expected_neighbor_interval_ms"] = 1000.0 * plot_df["expected_neighbor_interval_s"]
    bound_plot_df = bound_df.copy()
    if not bound_plot_df.empty:
        bound_plot_df["bound_interval_ms"] = 1000.0 * bound_plot_df["bound_interval_s"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sns.lineplot(
        data=plot_df,
        x="expected_neighbor_interval_ms",
        y="mean_mae_ratio",
        hue="neuron_count",
        marker="o",
        palette="viridis",
        ax=axes[0],
    )
    axes[0].axhline(
        mae_ratio_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold = {mae_ratio_threshold:.2f}",
    )
    axes[0].set_title("Calcium decoding degradation vs interval")
    axes[0].set_xlabel("Expected neighbor interval [ms]")
    axes[0].set_ylabel("Inferred / latent MAE")
    axes[0].legend(frameon=False, title="neurons")

    heatmap_df = plot_df.pivot(
        index="neuron_count",
        columns="running_speed",
        values="mean_mae_ratio",
    )
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="mako",
        cbar_kws={"label": "Mean MAE ratio"},
        ax=axes[1],
    )
    axes[1].set_title("MAE ratio across speed and population size")
    axes[1].set_xlabel("Running speed [cm/s]")
    axes[1].set_ylabel("Neuron count")

    bound_plot_df = bound_plot_df.dropna(subset=["bound_interval_ms"])
    if not bound_plot_df.empty:
        sns.pointplot(
            data=bound_plot_df,
            x="neuron_count",
            y="bound_interval_ms",
            color="#521b65",
            ax=axes[2],
        )
        axes[2].set_title("Estimated calcium interval bound")
        axes[2].set_xlabel("Neuron count")
        axes[2].set_ylabel("Bound interval [ms]")
    else:
        axes[2].axis("off")

    plt.show()


# %% [markdown]
# ## End-to-end demo runner
# %%
def build_scenario_builders(
    sampling_frequency: int,
    place_field_means: np.ndarray,
) -> list[tuple[str, str, Callable[[], tuple[np.ndarray, np.ndarray]]]]:
    """Construct replay scenarios compatible with the chosen neuron count."""

    return [
        (
            "continuous replay",
            "continuous",
            partial(
                make_continuous_replay,
                sampling_frequency=sampling_frequency,
                replay_speedup=10,
                place_field_means=place_field_means,
            ),
        ),
        (
            "hover replay",
            "stationary",
            partial(
                make_hover_replay,
                sampling_frequency=sampling_frequency,
                place_field_means=place_field_means,
            ),
        ),
        (
            "fragmented replay",
            "fragmented",
            partial(
                make_fragmented_replay,
                sampling_frequency=sampling_frequency,
                place_field_means=place_field_means,
            ),
        ),
    ]


def build_sweep_configs(base_config: CalciumTraceConfig) -> dict[str, CalciumTraceConfig]:
    """Default parameter sweep for calcium reliability experiments."""

    return {
        "baseline": base_config,
        "higher-background-noise": replace(base_config, sn=0.24, seed=23),
        "nonlinear": replace(
            base_config,
            nonlinearity=True,
            poiss_noise_factor=0.10,
            seed=43,
        ),
        "low-threshold": replace(base_config, spike_threshold=0.05, seed=47),
        "slow-decay": replace(base_config, g=0.98, seed=53),
    }


def make_default_demo_config(
    baseline_config: CalciumTraceConfig | None = None,
) -> CalciumTraceConfig:
    """Provide the default calcium front-end configuration for the demo."""

    if baseline_config is not None:
        return baseline_config

    return CalciumTraceConfig(
        # These defaults keep the traces noisy enough to expose the
        # calcium-front-end degradation, while still producing readable
        # tutorial-style place-field and posterior diagnostics.
        g=0.95,
        sn=0.16,
        poiss_noise_factor=0.05,
        s_min=0.01,
    )


def make_default_interval_sweep_config(
    sweep_config: IntervalSweepConfig | None = None,
) -> IntervalSweepConfig:
    """Provide the default interval-bound sweep configuration."""

    if sweep_config is not None:
        return sweep_config

    return IntervalSweepConfig()


def run_simulation_step(
    *,
    sampling_frequency: int,
    n_runs: int,
    place_field_means: np.ndarray,
) -> dict[str, object]:
    """Simulate the latent running data and visualize the place-cell baseline."""

    announce("simulating latent place-cell spikes")
    run_data = simulate_sorted_spike_run_data(
        sampling_frequency=sampling_frequency,
        n_runs=n_runs,
        place_field_means=place_field_means,
    )
    plot_training_data(run_data, run_data["spikes"], title_prefix="Latent training data")
    return {"run_data": run_data}


def run_split_step(run_data: dict[str, object]) -> dict[str, object]:
    """Split the latent run into train and test segments."""

    announce("splitting train and test segments")
    train_data, test_data = split_run_data(
        time=np.asarray(run_data["time"]),
        position=np.asarray(run_data["position"]),
        spikes=np.asarray(run_data["spikes"]),
    )
    return {"train_data": train_data, "test_data": test_data}


def run_calcium_frontend_step(
    *,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    run_data: dict[str, object],
    baseline_config: CalciumTraceConfig,
    place_field_means: np.ndarray,
) -> dict[str, object]:
    """Generate calcium traces, deconvolve them, and visualize recovery quality."""

    announce("generating calcium traces and running OASIS deconvolution")
    train_calcium = simulate_calcium_traces_from_spikes(train_data["spikes"], baseline_config)
    train_deconv = deconvolve_calcium_traces(
        train_calcium["fluorescence"], s_min=baseline_config.s_min
    )
    train_inferred_binary = binarize_inferred_spikes(
        train_deconv["inferred_spikes"], baseline_config.spike_threshold
    )

    test_calcium = simulate_calcium_traces_from_spikes(
        test_data["spikes"], replace(baseline_config, seed=baseline_config.seed + 1)
    )
    test_deconv = deconvolve_calcium_traces(
        test_calcium["fluorescence"], s_min=baseline_config.s_min
    )
    test_inferred_binary = binarize_inferred_spikes(
        test_deconv["inferred_spikes"], baseline_config.spike_threshold
    )

    spike_metrics = compute_spike_recovery_metrics(
        test_data["spikes"],
        test_deconv["inferred_spikes"],
        test_inferred_binary,
    )
    print(pd.Series(spike_metrics).round(3))
    plot_calcium_pipeline(
        test_data["time"],
        test_calcium,
        test_deconv,
        neuron_ids=get_example_neuron_ids(len(place_field_means)),
    )
    # plot_spike_recovery_validation(
    #     time=test_data["time"],
    #     true_spikes=test_data["spikes"],
    #     inferred_spikes_continuous=test_deconv["inferred_spikes"],
    #     inferred_spikes_binary=test_inferred_binary,
    #     metrics=spike_metrics,
    # )
    plot_training_data(
        {
            **run_data,
            "time": test_data["time"],
            "position": test_data["position"],
            "place_fields": run_data["place_fields"][len(train_data["time"]) :],
        },
        test_inferred_binary,
        title_prefix="Inferred spike test data",
    )

    return {
        "train_calcium": train_calcium,
        "train_deconv": train_deconv,
        "train_inferred_binary": train_inferred_binary,
        "test_calcium": test_calcium,
        "test_deconv": test_deconv,
        "test_inferred_binary": test_inferred_binary,
        "spike_metrics": spike_metrics,
    }


def run_calcium_likelihood_step(
    *,
    train_data: dict[str, np.ndarray],
    train_calcium: dict[str, np.ndarray],
    train_deconv: dict[str, object],
    train_inferred_binary: np.ndarray,
    run_data: dict[str, object],
    run_calcium_likelihood_section: bool,
    place_field_signal: str = "inferred_spikes",
) -> dict[str, object]:
    """Optionally estimate calcium-derived place fields and compare them to truth."""

    calcium_place_fields = None
    calcium_scales = None
    place_field_estimator = None
    place_field_signal_label = None

    if run_calcium_likelihood_section:
        reference_decoder = fit_sorted_spike_decoder(
            train_data["position"],
            train_data["spikes"],
            sampling_frequency=float(run_data["sampling_frequency"]),
        )
        position = np.asarray(train_data["position"])[:, np.newaxis]
        environment = reference_decoder.environment

        if place_field_signal == "inferred_spikes":
            announce("estimating place fields from inferred spikes with the sorted-spike KDE helper")
            calcium_place_fields = estimate_place_fields_kde(
                position=position,
                spikes=np.asarray(train_inferred_binary),
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
                edges=environment.edges_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
                position_std=[3.0],
                use_diffusion=False,
                block_size=None,
            )
            place_field_estimator = "sorted-spike KDE"
            place_field_signal_label = "inferred spikes"
        elif place_field_signal in {"fluorescence", "deconvolved_calcium"}:
            if estimate_calcium_place_fields is None:
                raise ImportError(
                    "estimate_calcium_place_fields is unavailable, so continuous calcium "
                    "place-field estimation cannot be run in this environment."
                )

            calcium_activity = (
                np.asarray(train_calcium["fluorescence"])
                if place_field_signal == "fluorescence"
                else np.asarray(train_deconv["deconvolved_calcium"])
            )
            announce(
                "estimating place fields from continuous calcium activity with the gamma helper"
            )
            calcium_place_fields, calcium_scales = estimate_calcium_place_fields(
                position=position,
                calcium_activity=calcium_activity,
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
            )
            place_field_estimator = "gamma calcium likelihood"
            place_field_signal_label = (
                "raw fluorescence"
                if place_field_signal == "fluorescence"
                else "denoised calcium trace"
            )
        else:
            raise ValueError(
                "place_field_signal must be one of "
                "{'inferred_spikes', 'fluorescence', 'deconvolved_calcium'}."
            )

        title = (
            f"Place fields from {place_field_signal_label}\n"
            f"estimator: {place_field_estimator}"
        )
        if calcium_scales is not None:
            title += f", mean scale={calcium_scales.mean():.3f}"
        plot_place_field_comparison(
            calcium_place_fields,
            place_field_means=np.asarray(run_data["place_field_means"]),
            place_field_variance=float(run_data["place_field_variance"]),
            max_rate=float(run_data["max_rate"]),
            sampling_frequency=float(run_data["sampling_frequency"]),
            title=title,
            normalize=place_field_signal != "inferred_spikes",
        )

    return {
        "calcium_place_fields": calcium_place_fields,
        "calcium_scales": calcium_scales,
        "place_field_estimator": place_field_estimator,
        "place_field_signal": place_field_signal_label,
    }


def run_decoder_step(
    *,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    train_inferred_binary: np.ndarray,
    test_inferred_binary: np.ndarray,
    run_data: dict[str, object],
) -> dict[str, object]:
    """Fit latent and inferred-spike decoders and compare their trajectories."""

    announce("fitting decoders on latent and inferred activity")
    latent_decoder = fit_sorted_spike_decoder(
        train_data["position"],
        train_data["spikes"],
        sampling_frequency=float(run_data["sampling_frequency"]),
    )
    inferred_decoder = fit_sorted_spike_decoder(
        train_data["position"],
        train_inferred_binary,
        sampling_frequency=float(run_data["sampling_frequency"]),
    )

    latent_results, latent_metrics = evaluate_decoder(
        latent_decoder,
        test_data["spikes"],
        test_data["position"],
        test_data["time"],
    )
    inferred_results, inferred_metrics = evaluate_decoder(
        inferred_decoder,
        test_inferred_binary,
        test_data["position"],
        test_data["time"],
    )
    print(
        pd.DataFrame(
            [latent_metrics, inferred_metrics],
            index=["latent spikes", "inferred spikes"],
        ).round(3)
    )
    posterior_summary = pd.concat(
        [
            summarize_decoder_posteriors(
                latent_results,
                true_position=test_data["position"],
                decoder_label="latent spikes",
            ),
            summarize_decoder_posteriors(
                inferred_results,
                true_position=test_data["position"],
                decoder_label="inferred spikes",
            ),
        ],
        ignore_index=True,
    )
    print(posterior_summary.round(3))
    plot_decoder_results(
        time=test_data["time"],
        true_position=test_data["position"],
        inferred_spikes=test_inferred_binary,
        inferred_decoder=inferred_decoder,
        baseline_results=latent_results,
        inferred_results=inferred_results,
        baseline_metrics=latent_metrics,
        inferred_metrics=inferred_metrics,
    )

    return {
        "latent_decoder": latent_decoder,
        "inferred_decoder": inferred_decoder,
        "latent_results": latent_results,
        "inferred_results": inferred_results,
        "latent_metrics": latent_metrics,
        "inferred_metrics": inferred_metrics,
        "posterior_summary": posterior_summary,
    }


def run_classifier_step(
    *,
    train_data: dict[str, np.ndarray],
    train_inferred_binary: np.ndarray,
    sampling_frequency: int,
    place_field_means: np.ndarray,
    baseline_config: CalciumTraceConfig,
) -> dict[str, object]:
    """Fit the replay classifier and evaluate all calcium-degraded scenarios."""

    announce("fitting classifier on inferred activity")
    classifier = fit_sorted_spike_classifier(
        train_data["position"],
        train_inferred_binary,
        sampling_frequency=float(sampling_frequency),
    )

    scenario_builders = build_scenario_builders(
        sampling_frequency=sampling_frequency,
        place_field_means=place_field_means,
    )
    classification_df = evaluate_classification_scenarios(
        classifier,
        calcium_config=baseline_config,
        scenario_builders=scenario_builders,
    )
    print(
        classification_df[
            ["scenario", "expected_state", "predicted_state", "expected_probability"]
        ].round(3)
    )
    plot_classification_summary(classification_df)
    for _, row in classification_df.iterrows():
        plot_classification(
            row["replay_time"],
            row["inferred_binary"],
            row["results"],
            title=f"{row['scenario']} after calcium front-end",
        )

    return {
        "classifier": classifier,
        "classification_df": classification_df,
    }


def run_parameter_sweep_step(
    *,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    run_data: dict[str, object],
    baseline_config: CalciumTraceConfig,
    run_parameter_sweep: bool,
) -> dict[str, object]:
    """Optionally evaluate decoding sensitivity to calcium front-end parameters."""

    sweep_results = None
    if run_parameter_sweep:
        announce("running calcium parameter sweep")
        sweep_results = evaluate_parameter_sweep(
            train=train_data,
            test=test_data,
            sampling_frequency=float(run_data["sampling_frequency"]),
            configs=build_sweep_configs(baseline_config),
        )
        print(sweep_results.round(3))
        plot_parameter_sweep_results(sweep_results)

    return {"sweep_results": sweep_results}


def run_interval_bound_step(
    *,
    baseline_config: CalciumTraceConfig,
    interval_sweep_config: IntervalSweepConfig,
    run_interval_bound_experiment: bool,
) -> dict[str, object]:
    """Run the spacing sweep and derive a calcium-decoding interval bound."""

    interval_sweep_results = None
    interval_sweep_summary = None
    interval_bound_df = None

    if run_interval_bound_experiment:
        announce("running calcium interval-bound experiment")
        (
            interval_sweep_results,
            interval_sweep_summary,
            interval_bound_df,
        ) = evaluate_interval_bound_sweep(
            calcium_config=baseline_config,
            sweep_config=interval_sweep_config,
        )
        print(
            interval_sweep_summary[
                [
                    "neuron_count",
                    "running_speed",
                    "expected_neighbor_interval_s",
                    "mean_mae_ratio",
                    "mean_mae_delta_cm",
                    "mean_inferred_mae_cm",
                ]
            ].round(3)
        )
        print(interval_bound_df.round(3))
        plot_interval_bound_results(
            interval_sweep_summary,
            interval_bound_df,
            mae_ratio_threshold=interval_sweep_config.mae_ratio_threshold,
        )

    return {
        "interval_sweep_results": interval_sweep_results,
        "interval_sweep_summary": interval_sweep_summary,
        "interval_bound_df": interval_bound_df,
    }


def finalize_demo_step(
    *,
    spike_metrics: dict[str, float],
    latent_metrics: dict[str, float],
    inferred_metrics: dict[str, float],
    interval_bound_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build the final summary table for the demo."""

    announce("finished script")
    metric_records = [
        {"metric": "spike_f1", "value": spike_metrics["f1"]},
        {"metric": "spike_continuous_corr", "value": spike_metrics["continuous_corr"]},
        {"metric": "latent_decoder_mae_cm", "value": latent_metrics["mae_cm"]},
        {"metric": "inferred_decoder_mae_cm", "value": inferred_metrics["mae_cm"]},
    ]
    if interval_bound_df is not None and not interval_bound_df.empty:
        finite_bounds = interval_bound_df["bound_interval_s"].dropna()
        if not finite_bounds.empty:
            metric_records.extend(
                [
                    {"metric": "min_interval_bound_s", "value": float(finite_bounds.min())},
                    {"metric": "max_interval_bound_s", "value": float(finite_bounds.max())},
                ]
            )

    summary = pd.DataFrame(metric_records)
    print(summary.round(3))
    return {"summary": summary}


def run_demo(
    *,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_runs: int = N_RUNS,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    baseline_config: CalciumTraceConfig | None = None,
    run_calcium_likelihood_section: bool = True,
    place_field_signal: str = "inferred_spikes",
    run_parameter_sweep: bool = False,
    run_interval_bound_experiment: bool = True,
    interval_sweep_config: IntervalSweepConfig | None = None,
) -> dict[str, object]:
    """Execute the full calcium simulation / decoding workflow."""

    baseline_config = make_default_demo_config(baseline_config)
    interval_sweep_config = make_default_interval_sweep_config(interval_sweep_config)
    demo_results: dict[str, object] = {}

    demo_results.update(
        run_simulation_step(
            sampling_frequency=sampling_frequency,
            n_runs=n_runs,
            place_field_means=place_field_means,
        )
    )
    demo_results.update(run_split_step(demo_results["run_data"]))
    demo_results.update(
        run_calcium_frontend_step(
            train_data=demo_results["train_data"],
            test_data=demo_results["test_data"],
            run_data=demo_results["run_data"],
            baseline_config=baseline_config,
            place_field_means=place_field_means,
        )
    )
    demo_results.update(
        run_calcium_likelihood_step(
            train_data=demo_results["train_data"],
            train_calcium=demo_results["train_calcium"],
            train_deconv=demo_results["train_deconv"],
            train_inferred_binary=demo_results["train_inferred_binary"],
            run_data=demo_results["run_data"],
            run_calcium_likelihood_section=run_calcium_likelihood_section,
            place_field_signal=place_field_signal,
        )
    )
    demo_results.update(
        run_decoder_step(
            train_data=demo_results["train_data"],
            test_data=demo_results["test_data"],
            train_inferred_binary=demo_results["train_inferred_binary"],
            test_inferred_binary=demo_results["test_inferred_binary"],
            run_data=demo_results["run_data"],
        )
    )
    demo_results.update(
        run_classifier_step(
            train_data=demo_results["train_data"],
            train_inferred_binary=demo_results["train_inferred_binary"],
            sampling_frequency=sampling_frequency,
            place_field_means=place_field_means,
            baseline_config=baseline_config,
        )
    )
    demo_results.update(
        run_parameter_sweep_step(
            train_data=demo_results["train_data"],
            test_data=demo_results["test_data"],
            run_data=demo_results["run_data"],
            baseline_config=baseline_config,
            run_parameter_sweep=run_parameter_sweep,
        )
    )
    demo_results.update(
        run_interval_bound_step(
            baseline_config=baseline_config,
            interval_sweep_config=interval_sweep_config,
            run_interval_bound_experiment=run_interval_bound_experiment,
        )
    )
    demo_results.update(
        finalize_demo_step(
            spike_metrics=demo_results["spike_metrics"],
            latent_metrics=demo_results["latent_metrics"],
            inferred_metrics=demo_results["inferred_metrics"],
            interval_bound_df=demo_results["interval_bound_df"],
        )
    )

    return demo_results


# %% [markdown]
# ## Notebook-style demo cells
# %%
RUN_DEMO_CELLS = __name__ == "__main__"
DEMO_CONFIG = {
    "sampling_frequency": SAMPLING_FREQUENCY,
    "n_runs": N_RUNS,
    "place_field_means": PLACE_FIELD_MEANS,
    "baseline_config": make_default_demo_config(),
    "run_calcium_likelihood_section": True,
    "place_field_signal": "inferred_spikes",
    "run_parameter_sweep": False,
    "run_interval_bound_experiment": True,
    "interval_sweep_config": make_default_interval_sweep_config(),
}
DEMO_STATE: dict[str, object] = {}
# %% [markdown]
# ### 1. Simulate latent place-cell spikes
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_simulation_step(
            sampling_frequency=DEMO_CONFIG["sampling_frequency"],
            n_runs=DEMO_CONFIG["n_runs"],
            place_field_means=DEMO_CONFIG["place_field_means"],
        )
    )
# %% [markdown]
# ### 2. Split train and test segments
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(run_split_step(DEMO_STATE["run_data"]))
# %% [markdown]
# ### 3. Simulate calcium traces, deconvolve them, and inspect recovery
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_calcium_frontend_step(
            train_data=DEMO_STATE["train_data"],
            test_data=DEMO_STATE["test_data"],
            run_data=DEMO_STATE["run_data"],
            baseline_config=DEMO_CONFIG["baseline_config"],
            place_field_means=DEMO_CONFIG["place_field_means"],
        )
    )
# %% [markdown]
# ### 4. Estimate place fields for the selected calcium-derived signal and compare them to ground truth
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_calcium_likelihood_step(
            train_data=DEMO_STATE["train_data"],
            train_calcium=DEMO_STATE["train_calcium"],
            train_deconv=DEMO_STATE["train_deconv"],
            train_inferred_binary=DEMO_STATE["train_inferred_binary"],
            run_data=DEMO_STATE["run_data"],
            run_calcium_likelihood_section=DEMO_CONFIG["run_calcium_likelihood_section"],
            place_field_signal=DEMO_CONFIG["place_field_signal"],
        )
    )
# %% [markdown]
# ### 5. Decode position, inspect the transition model, and compare causal/acausal posteriors
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_decoder_step(
            train_data=DEMO_STATE["train_data"],
            test_data=DEMO_STATE["test_data"],
            train_inferred_binary=DEMO_STATE["train_inferred_binary"],
            test_inferred_binary=DEMO_STATE["test_inferred_binary"],
            run_data=DEMO_STATE["run_data"],
        )
    )
# %% [markdown]
# ### 6. Classify replay scenarios after the calcium front-end
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_classifier_step(
            train_data=DEMO_STATE["train_data"],
            train_inferred_binary=DEMO_STATE["train_inferred_binary"],
            sampling_frequency=DEMO_CONFIG["sampling_frequency"],
            place_field_means=DEMO_CONFIG["place_field_means"],
            baseline_config=DEMO_CONFIG["baseline_config"],
        )
    )
# %% [markdown]
# ### 7. Run the optional calcium front-end parameter sweep
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_parameter_sweep_step(
            train_data=DEMO_STATE["train_data"],
            test_data=DEMO_STATE["test_data"],
            run_data=DEMO_STATE["run_data"],
            baseline_config=DEMO_CONFIG["baseline_config"],
            run_parameter_sweep=DEMO_CONFIG["run_parameter_sweep"],
        )
    )
# %% [markdown]
# ### 8. Estimate the calcium interval bound
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        run_interval_bound_step(
            baseline_config=DEMO_CONFIG["baseline_config"],
            interval_sweep_config=DEMO_CONFIG["interval_sweep_config"],
            run_interval_bound_experiment=DEMO_CONFIG["run_interval_bound_experiment"],
        )
    )
# %% [markdown]
# ### 9. Summarize downstream performance
# %%
if RUN_DEMO_CELLS:
    DEMO_STATE.update(
        finalize_demo_step(
            spike_metrics=DEMO_STATE["spike_metrics"],
            latent_metrics=DEMO_STATE["latent_metrics"],
            inferred_metrics=DEMO_STATE["inferred_metrics"],
            interval_bound_df=DEMO_STATE["interval_bound_df"],
        )
    )

# %%
