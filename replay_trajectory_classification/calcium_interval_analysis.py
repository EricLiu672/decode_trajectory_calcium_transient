"""Utilities for calcium-decoding interval-bound analyses.

These helpers stay independent of CAIMAN so the core interval and bound logic
can be reused from notebooks and covered by unit tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def make_evenly_spaced_place_field_means(
    neuron_count: int,
    track_height: float,
) -> NDArray[np.float64]:
    """Create evenly tiled 1D place-field centers across the track."""

    if neuron_count < 2:
        raise ValueError("neuron_count must be at least 2 to define an interval.")
    if track_height <= 0.0:
        raise ValueError("track_height must be positive.")

    return np.linspace(0.0, float(track_height), int(neuron_count), dtype=np.float64)


def _nearest_cross_neuron_intervals(
    event_time_ind: NDArray[np.int64],
    event_neuron_ind: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Find the nearest event interval, in frames, to a different neuron."""

    if event_time_ind.size < 2:
        return np.asarray([], dtype=np.float64)

    sort_ind = np.argsort(event_time_ind, kind="stable")
    event_time_ind = np.asarray(event_time_ind[sort_ind], dtype=np.int64)
    event_neuron_ind = np.asarray(event_neuron_ind[sort_ind], dtype=np.int64)
    nearest_intervals = []

    for event_ind, (time_ind, neuron_ind) in enumerate(zip(event_time_ind, event_neuron_ind)):
        best_interval = np.inf

        prev_ind = event_ind - 1
        while prev_ind >= 0:
            interval = float(time_ind - event_time_ind[prev_ind])
            if interval >= best_interval:
                break
            if event_neuron_ind[prev_ind] != neuron_ind:
                best_interval = interval
                break
            prev_ind -= 1

        next_ind = event_ind + 1
        while next_ind < event_time_ind.size:
            interval = float(event_time_ind[next_ind] - time_ind)
            if interval >= best_interval:
                break
            if event_neuron_ind[next_ind] != neuron_ind:
                best_interval = interval
                break
            next_ind += 1

        if np.isfinite(best_interval):
            nearest_intervals.append(best_interval)

    return np.asarray(nearest_intervals, dtype=np.float64)


def summarize_event_spacing(
    spikes: NDArray[np.float64],
    sampling_frequency: float,
    place_field_means: NDArray[np.float64] | None = None,
    running_speed: float | None = None,
) -> dict[str, float]:
    """Summarize expected and observed inter-neuron event spacing."""

    spike_array = np.asarray(spikes)
    if spike_array.ndim != 2:
        raise ValueError("spikes must be a 2D array with shape (time, neuron).")
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be positive.")

    event_time_ind, event_neuron_ind = np.nonzero(spike_array > 0)
    active_counts = np.bincount(event_time_ind, minlength=spike_array.shape[0])
    active_counts = active_counts[active_counts > 0]
    unique_event_time_ind = np.unique(event_time_ind)
    population_intervals = np.diff(unique_event_time_ind).astype(np.float64)
    cross_neuron_intervals = _nearest_cross_neuron_intervals(
        event_time_ind.astype(np.int64),
        event_neuron_ind.astype(np.int64),
    )

    summary = {
        "n_events": float(event_time_ind.size),
        "n_event_bins": float(unique_event_time_ind.size),
        "mean_active_neurons_per_event_bin": float(active_counts.mean())
        if active_counts.size
        else np.nan,
        "max_active_neurons_per_event_bin": float(active_counts.max())
        if active_counts.size
        else np.nan,
        "multi_neuron_event_bin_fraction": float(np.mean(active_counts > 1))
        if active_counts.size
        else np.nan,
        "observed_population_interval_frames_median": float(np.median(population_intervals))
        if population_intervals.size
        else np.nan,
        "observed_population_interval_frames_min": float(np.min(population_intervals))
        if population_intervals.size
        else np.nan,
        "observed_cross_neuron_interval_frames_median": float(
            np.median(cross_neuron_intervals)
        )
        if cross_neuron_intervals.size
        else np.nan,
        "observed_cross_neuron_interval_frames_min": float(np.min(cross_neuron_intervals))
        if cross_neuron_intervals.size
        else np.nan,
    }

    for key in (
        "observed_population_interval_frames_median",
        "observed_population_interval_frames_min",
        "observed_cross_neuron_interval_frames_median",
        "observed_cross_neuron_interval_frames_min",
    ):
        summary[key.replace("_frames_", "_s_")] = (
            summary[key] / sampling_frequency
            if np.isfinite(summary[key])
            else np.nan
        )

    if place_field_means is not None:
        ordered_means = np.sort(np.asarray(place_field_means, dtype=np.float64))
        place_field_spacing = np.diff(ordered_means)
        summary["place_field_spacing_cm"] = (
            float(np.median(place_field_spacing)) if place_field_spacing.size else np.nan
        )
    else:
        summary["place_field_spacing_cm"] = np.nan

    if running_speed is not None and np.isfinite(summary["place_field_spacing_cm"]):
        summary["expected_neighbor_interval_s"] = (
            summary["place_field_spacing_cm"] / float(running_speed)
            if running_speed > 0.0
            else np.nan
        )
    else:
        summary["expected_neighbor_interval_s"] = np.nan

    summary["expected_neighbor_interval_frames"] = (
        summary["expected_neighbor_interval_s"] * sampling_frequency
        if np.isfinite(summary["expected_neighbor_interval_s"])
        else np.nan
    )

    return summary


def extract_mae_interval_bounds(
    summary_df: pd.DataFrame,
    *,
    slice_columns: Sequence[str],
    interval_column: str = "expected_neighbor_interval_s",
    frame_interval_column: str = "expected_neighbor_interval_frames",
    mae_ratio_column: str = "mean_mae_ratio",
    threshold: float = 1.25,
) -> pd.DataFrame:
    """Extract the smallest acceptable interval within each parameter slice."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    if not slice_columns:
        raise ValueError("slice_columns must contain at least one column.")
    if summary_df.empty:
        return pd.DataFrame()

    required_columns = [*slice_columns, interval_column, mae_ratio_column]
    missing_columns = [column for column in required_columns if column not in summary_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for bound extraction: {missing_columns}")

    bound_records = []
    groupby_columns = list(slice_columns)

    for group_key, group in summary_df.groupby(groupby_columns, dropna=False, sort=True):
        group = group.sort_values(interval_column, kind="mergesort").reset_index(drop=True)
        acceptable = group[group[mae_ratio_column] <= threshold]

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = dict(zip(groupby_columns, group_key))
        record.update(
            {
                "mae_ratio_threshold": float(threshold),
                "smallest_tested_interval_s": float(group[interval_column].min()),
                "largest_tested_interval_s": float(group[interval_column].max()),
                "n_tested_conditions": int(len(group)),
                "n_acceptable_conditions": int(len(acceptable)),
            }
        )

        if acceptable.empty:
            record.update(
                {
                    "bound_interval_s": np.nan,
                    "bound_interval_frames": np.nan,
                    "bound_mae_ratio": np.nan,
                    "bound_status": "not_reached",
                }
            )
        else:
            bound_row = acceptable.iloc[0]
            bound_interval = float(bound_row[interval_column])
            record.update(
                {
                    "bound_interval_s": bound_interval,
                    "bound_interval_frames": float(bound_row[frame_interval_column])
                    if frame_interval_column in bound_row.index
                    else np.nan,
                    "bound_mae_ratio": float(bound_row[mae_ratio_column]),
                    "bound_status": (
                        "all_tested_acceptable"
                        if len(acceptable) == len(group)
                        else "threshold_crossed"
                    ),
                }
            )

        for column in group.columns:
            if column in groupby_columns or column in {
                interval_column,
                frame_interval_column,
                mae_ratio_column,
            }:
                continue
            if column.startswith("mean_"):
                record[f"bound_{column[5:]}"] = (
                    float(acceptable.iloc[0][column]) if not acceptable.empty else np.nan
                )

        bound_records.append(record)

    return pd.DataFrame.from_records(bound_records)
