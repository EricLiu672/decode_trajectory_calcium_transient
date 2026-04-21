# Copilot instructions for `replay_trajectory_classification`

## Build, test, and lint commands

Use the conda environment from `environment.yml`; this repository depends on `conda-forge`, `franklab`, and `edeno` packages such as `track_linearization` and `regularized_glm`.

```bash
conda env create -f environment.yml
conda activate replay_trajectory_classification
pip install -e .
```

Install optional toolsets as needed:

```bash
pip install -e '.[dev]'
pip install -e '.[test]'
pip install -e '.[docs]'
```

Lint:

```bash
ruff check replay_trajectory_classification/
flake8 replay_trajectory_classification/ --max-line-length=88 --select=E9,F63,F7,F82 --show-source --statistics
```

Unit tests:

```bash
pytest replay_trajectory_classification/tests -v
pytest replay_trajectory_classification/tests/unit -v -m "not gpu"
pytest replay_trajectory_classification/tests/unit/test_api_basic.py::test_main_api_imports -v
```

Pytest markers from `pytest.ini`:

```bash
pytest replay_trajectory_classification/tests -v -m "not slow"
pytest replay_trajectory_classification/tests -v -m "not gpu"
```

Notebook integration tests (these are the CI-critical tests in `.github/workflows/PR-test.yml`):

```bash
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute notebooks/tutorial/01-Introduction_and_Data_Format.ipynb --output-dir=/tmp

for nb in notebooks/tutorial/*.ipynb; do
  jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 --execute "$nb"
done
```

Build distributions:

```bash
pip install build
python -m build --wheel
python -m build --sdist
```

Documentation:

```bash
pip install -e '.[docs]'
make -C docs html
```

## High-level architecture

`replay_trajectory_classification/__init__.py` is the public API surface. It re-exports the main decoder/classifier classes, transition models, `Environment`, `ObservationModel`, and several `track_linearization` helpers, so compatibility changes often need to be reflected there.

The package is organized around state-space decoding:

- `decoder.py` implements single-dynamics spatial decoding through `_DecoderBase`, `SortedSpikesDecoder`, and `ClusterlessDecoder`.
- `classifier.py` implements multi-state trajectory classification through `_ClassifierBase`, `SortedSpikesClassifier`, and `ClusterlessClassifier`.
- `core.py` contains the low-level causal/acausal Bayesian routines used by both decoders and classifiers. These paths are performance-sensitive and include CPU and GPU variants.

Spatial geometry is handled by `environments.py`. `Environment` is a dataclass that discretizes position into bins, either by inferring a grid from position samples or by constructing a 1D layout from a `track_graph`. Classifiers can hold multiple `Environment` instances at once.

Observation/state wiring is split across:

- `observation_model.py`, where `ObservationModel(environment_name, encoding_group)` links a classifier state to an environment and encoding group.
- `continuous_state_transitions.py`, which defines movement models such as `RandomWalk`, `EmpiricalMovement`, `RandomWalkDirection1`, `RandomWalkDirection2`, `Identity`, and `Uniform`.
- `discrete_state_transitions.py`, which defines switching among states with `DiagonalDiscrete`, `RandomDiscrete`, `UniformDiscrete`, and `UserDefinedDiscrete`.

Likelihood estimation is registry-driven. `likelihoods/__init__.py` maps algorithm strings to `(fit_fn, estimate_fn)` pairs for sorted spikes, clusterless multiunit data, and calcium data. Decoder/classifier classes choose implementations through parameters like `sorted_spikes_algorithm` and `clusterless_algorithm` instead of hard-coding one likelihood.

The typical flow is:

1. Fit environment bins from position.
2. Fit observation model parameters from spikes or multiunit marks.
3. Fit continuous/discrete transition models and initial conditions.
4. Run `predict(...)`, which calls the shared Bayesian core and returns labeled results.

`predict()` methods return `xarray.Dataset` objects, not plain NumPy arrays or pandas DataFrames. Downstream code expects named dimensions such as time, position bins, and states.

## Key conventions

There are parallel sorted-spike and clusterless pathways throughout the codebase. If you change shared behavior in `decoder.py`, `classifier.py`, or `likelihoods/`, check whether the corresponding sorted and clusterless classes or algorithms need matching updates.

Likelihood APIs follow a fit/estimate split. Registry entries pair a training-time function with an inference-time function; new likelihoods should fit that pattern so they can plug into the existing algorithm maps cleanly.

Scikit-learn-style estimator conventions matter here. The main classes inherit from `sklearn.base.BaseEstimator`, expose `fit(...)`/`predict(...)`, and support persistence helpers like `save_model()` / `load_model()`.

Environment names are the join key for multi-environment classification. `ObservationModel.environment_name` is how classifier states are associated with particular `Environment` instances, so keep those names aligned when adding or reworking multi-environment logic.

GPU support is optional, not a separate product line. GPU implementations live in `*_gpu.py` modules and are selected by algorithm name or `use_gpu=True`; CPU behavior must remain correct when CuPy is unavailable.

Notebook tutorials in `notebooks/tutorial/` are not just examples; CI executes them end-to-end. Keep them runnable when changing public APIs, defaults, result shapes, or import paths.

Unit tests under `replay_trajectory_classification/tests/unit/` use small synthetic datasets and sometimes skip edge cases around inferred track boundaries. For low-level behavior changes, run targeted pytest cases plus the affected tutorial notebook.

The repository uses modern packaging through `pyproject.toml`; prefer `pip install -e .` and `python -m build ...` workflows, not legacy `setup.py` commands.
