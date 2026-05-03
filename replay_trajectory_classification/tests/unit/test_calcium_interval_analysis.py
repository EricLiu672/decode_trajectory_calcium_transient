import numpy as np
import pandas as pd
import pytest

from replay_trajectory_classification.calcium_interval_analysis import (
    extract_mae_interval_bounds,
    make_evenly_spaced_place_field_means,
    summarize_event_spacing,
)


def test_make_evenly_spaced_place_field_means_tiles_track():
    means = make_evenly_spaced_place_field_means(neuron_count=5, track_height=180.0)

    np.testing.assert_allclose(means, np.array([0.0, 45.0, 90.0, 135.0, 180.0]))


@pytest.mark.parametrize("neuron_count", [0, 1])
def test_make_evenly_spaced_place_field_means_requires_multiple_neurons(neuron_count):
    with pytest.raises(ValueError):
        make_evenly_spaced_place_field_means(neuron_count=neuron_count, track_height=180.0)


def test_summarize_event_spacing_reports_expected_and_observed_intervals():
    spikes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    summary = summarize_event_spacing(
        spikes,
        sampling_frequency=10.0,
        place_field_means=np.array([0.0, 20.0, 40.0]),
        running_speed=10.0,
    )

    assert summary["n_events"] == 5.0
    assert summary["n_event_bins"] == 4.0
    assert summary["place_field_spacing_cm"] == 20.0
    assert summary["expected_neighbor_interval_s"] == 2.0
    assert summary["expected_neighbor_interval_frames"] == 20.0
    assert summary["observed_population_interval_frames_median"] == 1.0
    assert summary["observed_cross_neuron_interval_frames_min"] == 0.0
    assert summary["multi_neuron_event_bin_fraction"] == 0.25


def test_extract_mae_interval_bounds_picks_smallest_acceptable_interval():
    summary_df = pd.DataFrame(
        {
            "neuron_count": [8, 8, 8, 12, 12],
            "sampling_frequency": [30, 30, 30, 30, 30],
            "expected_neighbor_interval_s": [0.60, 0.40, 0.20, 0.50, 0.25],
            "expected_neighbor_interval_frames": [18.0, 12.0, 6.0, 15.0, 7.5],
            "mean_mae_ratio": [1.05, 1.18, 1.40, 1.10, 1.22],
            "mean_inferred_mae_cm": [8.0, 9.0, 12.0, 7.5, 8.5],
        }
    )

    bounds = extract_mae_interval_bounds(
        summary_df,
        slice_columns=("neuron_count", "sampling_frequency"),
        threshold=1.25,
    ).sort_values("neuron_count")

    assert list(bounds["bound_status"]) == ["threshold_crossed", "all_tested_acceptable"]
    np.testing.assert_allclose(bounds["bound_interval_s"], np.array([0.40, 0.25]))
    np.testing.assert_allclose(bounds["bound_interval_frames"], np.array([12.0, 7.5]))
    np.testing.assert_allclose(bounds["bound_mae_ratio"], np.array([1.18, 1.22]))


def test_extract_mae_interval_bounds_returns_not_reached_when_needed():
    summary_df = pd.DataFrame(
        {
            "neuron_count": [16, 16],
            "sampling_frequency": [30, 30],
            "expected_neighbor_interval_s": [0.30, 0.15],
            "expected_neighbor_interval_frames": [9.0, 4.5],
            "mean_mae_ratio": [1.30, 1.45],
        }
    )

    bounds = extract_mae_interval_bounds(
        summary_df,
        slice_columns=("neuron_count", "sampling_frequency"),
        threshold=1.25,
    )

    assert bounds.loc[0, "bound_status"] == "not_reached"
    assert np.isnan(bounds.loc[0, "bound_interval_s"])
