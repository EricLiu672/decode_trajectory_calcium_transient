"""Replay trajectory classification package.

A Python package for decoding spatial position from neural activity and
classifying trajectory types in hippocampal replay events using state-space
modeling approaches.

The main classes are:

- SortedSpikesClassifier/Decoder: For spike-sorted neural data
- ClusterlessClassifier/Decoder: For clusterless (unsorted) neural data
- Environment: Spatial environment representation with discrete grids

Examples
--------
>>> from replay_trajectory_classification import SortedSpikesClassifier
>>> classifier = SortedSpikesClassifier()
"""

# flake8: noqa
from track_linearization import (
    get_linearized_position,
    make_actual_vs_linearized_position_movie,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)

from replay_trajectory_classification.classifier import (
    CalciumClassifier,
    ClusterlessClassifier,
    SortedSpikesClassifier,
)
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
    estimate_movement_var,
)
from replay_trajectory_classification.decoder import (
    ClusterlessDecoder,
    SortedSpikesDecoder,
)
from replay_trajectory_classification.discrete_state_transitions import (
    DiagonalDiscrete,
    RandomDiscrete,
    UniformDiscrete,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from replay_trajectory_classification.observation_model import ObservationModel
from replay_trajectory_classification.simulate_calcium import (
    make_continuous_replay,
    make_fragmented_replay,
    make_hover_replay,
    make_simulated_run_data,
    simulate_calcium_from_spikes,
    compute_ar2_coefficients,
)

__version__ = "1.4.1"
