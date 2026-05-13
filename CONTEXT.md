# Context: replay_trajectory_classification (calcium decoding)

## Domain terms

### Signals

**Continuous OASIS output**  
The OASIS `s` array: a continuous, non-negative float representing the inferred calcium event amplitude per time bin, produced by `oasis.functions.deconvolve` *before* any thresholding. This is the signal fed to the ZIG likelihood model (`deconv_calcium_likelihood`).

**Binarized OASIS spikes**  
The output of `deconvolve_and_binarize()`: binary 0/1 int8 array produced by thresholding the OASIS continuous output at zero. This is the signal fed to the Poisson KDE likelihood model (`spiking_likelihood_kde`) via `SortedSpikesClassifier`.

**Deconvolved spikes** *(user-facing, imprecise)*  
Informal term that can refer to either the continuous or binarized OASIS output depending on context. Prefer the precise terms above in code and comments.

**Calcium trace / ΔF/F**  
The raw fluorescence signal returned by `make_simulated_run_data` as `calcium_traces`. This is the *input* to OASIS deconvolution, not directly used as a likelihood input.

### Models

**Poisson KDE classifier**  
A `SortedSpikesClassifier` using `sorted_spikes_algorithm="spiking_likelihood_kde"`, trained and predicted on binarized OASIS spikes. Treats each binarized event as a Poisson spike.

**ZIG classifier**  
A `CalciumClassifier` using `calcium_algorithm="deconv_calcium_likelihood"`, trained and predicted on continuous OASIS output. Models the calcium signal as Zero-Inflated Gamma (ZIG) distributed.

**Gamma classifier** *(excluded from current work)*  
A `CalciumClassifier` using `calcium_algorithm="calcium_likelihood"`. Takes raw calcium traces. Explicitly excluded from the ZIG vs Poisson KDE comparison notebooks.

### States

**Local**  
Smooth position evolution modelled by `RandomWalk`. Corresponds to normal running.

**Stationary**  
Position stays in one bin, modelled by `Identity`. Corresponds to hover/pause events.

**Jump**  
Non-local position transitions, modelled by `Uniform`. Corresponds to fragmented replay.

### Simulation contract

**Calcium replay simulation contract**  
The three primary calcium replay functions (`make_continuous_replay`, `make_hover_replay`, `make_fragmented_replay`) follow the same pipeline as run-data generation: replay position → place-field firing rate → Poisson spikes → AR(2) calcium traces. Sparsity is controlled through event duration and dwell length, not replay-specific rate changes.  
_Avoid_: hand-placed isolated spike events, replay-specific rate inflation.

**Replay position trajectory**  
The position signal fed into place-field rate computation for a replay event. Shape determines the replay class: monotonic ramp for continuous replay, constant value for stationary replay, piecewise-constant with non-local jumps for fragmented replay.

## Relationships

- **Replay position trajectory** is processed through place-field rates to produce **Calcium trace / ΔF/F**
- **Calcium trace / ΔF/F** is transformed by OASIS into either **Continuous OASIS output** or **Binarized OASIS spikes**
- **ZIG classifier** consumes **Continuous OASIS output**; **Poisson KDE classifier** consumes **Binarized OASIS spikes**
- The replay class (Local / Stationary / Jump) is determined solely by the shape of the **Replay position trajectory**, not by how spikes are placed

## Flagged ambiguities

- "GT spikes" was used informally to mean the ground-truth replay-generating signal — resolved: the canonical output is Poisson spikes drawn from place-field rates at the replay position, same as run data.
