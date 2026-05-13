import numpy as np

from replay_trajectory_classification.calcium_sorted_spikes_decoding import (
    deconvolve_and_binarize,
    fit_sorted_spikes_decoder,
    median_decoding_error,
)
from replay_trajectory_classification.simulate_calcium import (
    CALCIUM_CONTINUOUS_N_FRAMES,
    CALCIUM_FRAGMENTED_N_FRAMES,
    CALCIUM_HOVER_N_FRAMES,
    make_continuous_replay,
    make_fragmented_replay,
    make_simulated_run_data,
    make_hover_replay,
    make_theta_sweep,
    simulate_calcium_from_spikes,
)


def test_simulate_calcium_from_spikes_subsamples_spikes_and_trace():
    """Core calcium simulation returns subsampled spikes and trace outputs."""
    spikes_1ms = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    true_spikes, clean_calcium, noisy_calcium = simulate_calcium_from_spikes(
        spikes_1ms=spikes_1ms,
        sigma=0.0,
        tau_d=400.0,
        tau_r=1.0,
        subsample_factor=2,
        rng=np.random.default_rng(0),
    )

    expected_spikes = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    assert true_spikes.shape == clean_calcium.shape == noisy_calcium.shape == (3, 2)
    np.testing.assert_array_equal(true_spikes, expected_spikes)
    np.testing.assert_allclose(clean_calcium, noisy_calcium)
    assert np.all(clean_calcium >= 0.0)
    assert np.any(clean_calcium[1:] > 0.0)


def test_make_simulated_run_data_returns_aligned_calcium_outputs():
    """Simulated calcium run data stays aligned across time, position, and neurons."""
    (
        time,
        position,
        sampling_frequency,
        calcium_traces,
        true_spikes,
        place_fields,
    ) = make_simulated_run_data(
        sampling_frequency=20,
        track_height=50.0,
        running_speed=10.0,
        n_runs=1,
        place_field_means=np.array([10.0, 40.0]),
        sigma=0.0,
        rng=np.random.default_rng(1),
    )

    n_time = time.shape[0]
    assert time.shape == position.shape == (n_time,)
    assert sampling_frequency == 20
    assert (
        calcium_traces.shape
        == true_spikes.shape
        == place_fields.shape
        == (n_time, 2)
    )
    assert np.all(np.diff(time) > 0.0)
    assert np.all(position >= 0.0)
    assert np.all(position <= 50.0)
    assert np.all(calcium_traces >= 0.0)
    assert np.all(place_fields >= 0.0)


def test_make_hover_replay_returns_calcium_for_the_hovered_neuron():
    """Hover replay stays aligned and concentrates spikes on one neuron."""
    replay_time, test_spikes, calcium_traces = make_hover_replay(
        hover_neuron_ind=1,
        place_field_means=np.array([0.0, 50.0, 100.0]),
        sampling_frequency=20,
        sigma=0.0,
    )

    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape
    assert test_spikes.shape[1] == 3
    assert np.argmax(test_spikes.sum(axis=0)) == 1
    assert np.all(np.diff(replay_time) > 0.0)
    assert np.all(calcium_traces >= 0.0)


def test_make_hover_replay_default_duration_matches_calcium_timescale():
    """Default hover replay should span a meaningful number of calcium frames."""
    replay_time, test_spikes, calcium_traces = make_hover_replay(sigma=0.0)

    assert replay_time.shape == (CALCIUM_HOVER_N_FRAMES,)
    assert test_spikes.shape == calcium_traces.shape == (CALCIUM_HOVER_N_FRAMES, 19)
    assert np.all(np.diff(replay_time) > 0.0)


def test_make_hover_replay_respects_spike_interval():
    """Hover replay exposes the calcium-frame spike interval as part of its API."""
    replay_time, test_spikes, calcium_traces = make_hover_replay(
        hover_neuron_ind=1,
        place_field_means=np.array([0.0, 50.0, 100.0]),
        sampling_frequency=20,
        sigma=0.0,
        n_frames=12,
        spike_interval=4,
    )

    assert replay_time.shape == (12,)
    assert test_spikes.shape == calcium_traces.shape == (12, 3)
    np.testing.assert_array_equal(np.flatnonzero(test_spikes[:, 1]), np.array([0, 4, 8]))
    assert np.count_nonzero(test_spikes[:, [0, 2]]) == 0


def test_make_continuous_replay_default_duration_matches_calcium_timescale():
    """Default continuous replay should span many calcium frames, not just a few."""
    replay_time, test_spikes, calcium_traces = make_continuous_replay(sigma=0.0)

    assert replay_time.shape == (CALCIUM_CONTINUOUS_N_FRAMES,)
    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape == (CALCIUM_CONTINUOUS_N_FRAMES, 19)
    assert np.all(np.diff(replay_time) > 0.0)


def test_make_continuous_replay_respects_frame_limited_realworld_speed():
    """Continuous calcium replay only activates neurons within the traversed range."""
    replay_time, test_spikes, calcium_traces = make_continuous_replay(
        sampling_frequency=20,
        internal_sampling_frequency=1000,
        track_height=100.0,
        running_speed=50.0,
        place_field_means=np.array([0.0, 10.0, 20.0, 25.0, 40.0]),
        n_frames=12,
        replay_speedup=1.0,
        sigma=0.0,
    )

    assert replay_time.shape == (12,)
    assert test_spikes.shape == calcium_traces.shape == (12, 5)
    np.testing.assert_array_equal(test_spikes.sum(axis=0), np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
    assert np.all(np.diff(np.flatnonzero(test_spikes.sum(axis=1))) >= 0)


def test_make_fragmented_replay_default_duration_matches_calcium_timescale():
    """Default fragmented replay should produce a non-empty calcium-timescale event."""
    replay_time, test_spikes, calcium_traces = make_fragmented_replay(sigma=0.0)

    assert replay_time.shape[0] >= 30
    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape
    assert np.count_nonzero(test_spikes.sum(axis=1)) >= 5


def test_make_fragmented_replay_respects_spike_interval():
    """Fragmented replay exposes sparse jump timing at the calcium frame scale."""
    replay_time, test_spikes, calcium_traces = make_fragmented_replay(
        place_field_means=np.array([0.0, 50.0, 100.0, 150.0]),
        sampling_frequency=20,
        sigma=0.0,
        n_frames=12,
        spike_interval=4,
        rng=np.random.default_rng(0),
    )

    assert replay_time.shape == (12,)
    assert test_spikes.shape == calcium_traces.shape == (12, 4)
    np.testing.assert_array_equal(np.flatnonzero(test_spikes.sum(axis=1)), np.array([0, 4, 8]))
    np.testing.assert_array_equal(np.argmax(test_spikes[[0, 4, 8]], axis=1), np.array([3, 2, 2]))
    np.testing.assert_array_equal(test_spikes.sum(axis=1)[[0, 4, 8]], np.ones(3))


def test_make_simulated_run_data_is_reproducible_with_seeded_rng():
    """Seeded generators reproduce the same simulated calcium dataset."""
    kwargs = dict(
        sampling_frequency=20,
        track_height=50.0,
        running_speed=10.0,
        n_runs=1,
        place_field_means=np.array([10.0, 40.0]),
        sigma=0.05,
    )

    outputs_1 = make_simulated_run_data(rng=np.random.default_rng(7), **kwargs)
    outputs_2 = make_simulated_run_data(rng=np.random.default_rng(7), **kwargs)

    for output_1, output_2 in zip(outputs_1[:-1], outputs_2[:-1]):
        np.testing.assert_allclose(output_1, output_2)
    np.testing.assert_allclose(outputs_1[-1], outputs_2[-1])


def test_make_theta_sweep_returns_aligned_calcium_outputs():
    """Theta sweep returns aligned time, spikes, and calcium outputs."""
    replay_time, test_spikes, calcium_traces = make_theta_sweep(
        sampling_frequency=20,
        internal_sampling_frequency=1000,
        sigma=0.0,
        n_sweeps=2,
    )

    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape
    assert test_spikes.shape[1] > 1
    assert np.all(np.diff(replay_time) > 0.0)
    assert np.all(calcium_traces >= 0.0)
    assert np.any(test_spikes.sum(axis=0) > 0.0)


def test_make_theta_sweep_respects_place_field_means():
    """Theta sweep matches the requested neuron layout."""
    replay_time, test_spikes, calcium_traces = make_theta_sweep(
        sampling_frequency=20,
        internal_sampling_frequency=1000,
        place_field_means=np.array([0.0, 60.0, 120.0, 180.0]),
        sigma=0.0,
        n_sweeps=2,
    )

    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape
    assert test_spikes.shape[1] == 4


def test_make_theta_sweep_handles_endpoint_inclusive_place_fields():
    """Theta sweep supports evenly spaced place fields including track end."""
    replay_time, test_spikes, calcium_traces = make_theta_sweep(
        sampling_frequency=200,
        internal_sampling_frequency=1000,
        track_height=180.0,
        running_speed=15.0,
        place_field_means=np.linspace(0.0, 180.0, 20),
        replay_speedup=145,
        sigma=0.1,
        rng=np.random.default_rng(42),
        n_sweeps=6,
    )

    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape == (150, 20)


def test_make_theta_sweep_handles_odd_internal_replay_lengths():
    """Theta sweep remains valid when the internal continuous replay length is odd."""
    replay_time, test_spikes, calcium_traces = make_theta_sweep(
        sampling_frequency=200,
        internal_sampling_frequency=1000,
        track_height=180.0,
        running_speed=15.0,
        place_field_means=np.linspace(0.0, 180.0, 20),
        replay_speedup=51,
        sigma=0.1,
        rng=np.random.default_rng(42),
        n_sweeps=2,
    )

    assert replay_time.shape == (test_spikes.shape[0],)
    assert test_spikes.shape == calcium_traces.shape
    assert test_spikes.shape[1] == 20


def test_deconvolve_and_binarize_returns_binary_spike_matrix():
    """Deconvolution returns a binary spike matrix aligned with the calcium input."""
    (
        _time,
        _position,
        _sampling_frequency,
        calcium_traces,
        _true_spikes,
        _place_fields,
    ) = make_simulated_run_data(
        sampling_frequency=30,
        track_height=50.0,
        running_speed=10.0,
        n_runs=1,
        place_field_means=np.array([10.0, 25.0, 40.0]),
        sigma=0.1,
        rng=np.random.default_rng(11),
    )

    inferred_spikes = deconvolve_and_binarize(calcium_traces)

    assert inferred_spikes.shape == calcium_traces.shape
    assert np.issubdtype(inferred_spikes.dtype, np.integer)
    assert set(np.unique(inferred_spikes)).issubset({0, 1})
    assert inferred_spikes.sum() > 0


def test_fit_sorted_spikes_decoder_decodes_binarized_calcium_run_data():
    """A decoder fit from calcium-derived spikes can recover position on a new run."""
    run_a = make_simulated_run_data(
        sampling_frequency=30,
        track_height=120.0,
        running_speed=10.0,
        n_runs=2,
        place_field_means=np.linspace(0.0, 120.0, 12),
        sigma=1.0,
        rng=np.random.default_rng(0),
    )
    run_b = make_simulated_run_data(
        sampling_frequency=30,
        track_height=120.0,
        running_speed=10.0,
        n_runs=2,
        place_field_means=np.linspace(0.0, 120.0, 12),
        sigma=1.0,
        rng=np.random.default_rng(1),
    )

    (
        time_a,
        position_a,
        sampling_frequency,
        calcium_a,
        _true_spikes_a,
        _place_fields_a,
    ) = run_a
    (
        time_b,
        position_b,
        _sampling_frequency_b,
        calcium_b,
        _true_spikes_b,
        _place_fields_b,
    ) = run_b

    decoder, inferred_spikes_a = fit_sorted_spikes_decoder(
        position=position_a,
        calcium_traces=calcium_a,
        sampling_frequency=sampling_frequency,
        position_std=3.0,
    )
    inferred_spikes_b = deconvolve_and_binarize(calcium_b)
    results = decoder.predict(inferred_spikes_b, time=time_b)

    assert inferred_spikes_a.shape == calcium_a.shape
    assert results.causal_posterior.shape[0] == time_b.shape[0]
    assert results.acausal_posterior.shape[0] == time_b.shape[0]
    np.testing.assert_allclose(results.causal_posterior.sum("position"), 1.0)
    np.testing.assert_allclose(results.acausal_posterior.sum("position"), 1.0)
    assert median_decoding_error(results.acausal_posterior, position_b) < 5.0


def test_state_space_classifier_separates_realworld_calcium_events_with_true_event_spikes():
    """State-space event labels stay distinct when replay timing matches calcium timescales."""
    from replay_trajectory_classification import (
        DiagonalDiscrete,
        Environment,
        Identity,
        RandomWalk,
        SortedSpikesClassifier,
        Uniform,
    )
    from replay_trajectory_classification.continuous_state_transitions import (
        estimate_movement_var,
    )

    track_height = 200.0
    place_field_means = np.linspace(0.0, track_height, 20)
    sampling_frequency = 30
    internal_sampling_frequency = 1000
    running_speed = 15.0
    sigma = 1.0
    state_names = ["Local", "Stationary", "Jump"]

    (
        _time,
        position,
        _sampling_frequency,
        calcium_traces,
        _true_spikes,
        _place_fields,
    ) = make_simulated_run_data(
        sampling_frequency=sampling_frequency,
        internal_sampling_frequency=internal_sampling_frequency,
        track_height=track_height,
        running_speed=running_speed,
        n_runs=2,
        place_field_means=place_field_means,
        sigma=sigma,
        rng=np.random.default_rng(0),
    )
    inferred_spikes = deconvolve_and_binarize(calcium_traces)
    movement_var = float(
        estimate_movement_var(position.reshape(-1, 1), sampling_frequency)
    )

    classifier = SortedSpikesClassifier(
        environments=[Environment(environment_name="", place_bin_size=1.0)],
        continuous_transition_types=[
            [RandomWalk(movement_var=movement_var), Identity(), Uniform()],
            [RandomWalk(movement_var=movement_var), Identity(), Uniform()],
            [Uniform(), Uniform(), Uniform()],
        ],
        discrete_transition_type=DiagonalDiscrete(),
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={
            "position_std": 3.0,
            "use_diffusion": False,
            "block_size": None,
        },
    )
    classifier.fit(position, inferred_spikes)

    def mean_state_probability(event_fn, seed, **kwargs):
        time, true_spikes, _calcium_traces = event_fn(
            sampling_frequency=sampling_frequency,
            internal_sampling_frequency=internal_sampling_frequency,
            place_field_means=place_field_means,
            sigma=sigma,
            rng=np.random.default_rng(seed),
            **kwargs,
        )
        results = classifier.predict(true_spikes, time=time, state_names=state_names)
        return results.acausal_posterior.sum("position").mean("time").values

    continuous_probability = mean_state_probability(
        make_continuous_replay,
        2,
        track_height=track_height,
        running_speed=running_speed,
        n_frames=CALCIUM_CONTINUOUS_N_FRAMES,
        replay_speedup=1.0,
    )
    stationary_probability = mean_state_probability(
        make_hover_replay,
        3,
        n_frames=CALCIUM_HOVER_N_FRAMES,
    )
    fragmented_probability = mean_state_probability(
        make_fragmented_replay,
        4,
        n_frames=CALCIUM_FRAGMENTED_N_FRAMES,
    )

    assert continuous_probability[0] > continuous_probability[2]
    assert stationary_probability[1] > stationary_probability[0]
    assert stationary_probability[1] > stationary_probability[2]
    assert fragmented_probability[2] > fragmented_probability[0]
    assert fragmented_probability[2] > fragmented_probability[1]
