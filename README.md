# Ukumog Engine

Search-first engine for an arithmetic-pattern placement game on an 11x11 board. The current codebase is closest to a Stockfish-style hybrid: exact rules, tactical search, alpha-beta/PVS, heavy instrumentation, and only light ML assistance where search can stay in control.

## Overview

Game rules:

* Two players alternate placing one stone on an empty cell.
* A move wins immediately if it creates a valid pattern of five.
* A move loses immediately if it creates a valid pattern of four and not five.
* A pattern is an arithmetic progression on the board; stones do not need to be adjacent.

Repo goal:

* keep the engine search-led and tactically exact
* improve depth and efficiency before spending more complexity on ML
* use ML as a helper for quiet-node evaluation only after it proves value in equal-time matches

## What Works Now

The current repo is not just scaffolding. These parts are implemented and covered by tests:

* exact rule engine with precomputed 4-mask and 5-mask tables
* tactical analysis for immediate wins, poison moves, forced blocks, safe threats, and double threats
* alpha-beta negamax with iterative deepening, PVS, aspiration windows, killer/history ordering, quiet-move caps, and tactical quiescence
* reversible incremental state with make/unmake and lookup-backed mask evaluation
* selective tactical proof solver for sharp positions
* CLI play and engine-vs-engine benchmarking with structured search summaries
* desktop GUI play with a mouse-driven board, live engine configuration panels, and on-demand position analysis
* quiet-position self-play data generation, mask-state value training, dataset inspection, and checkpoint loading
* legacy CNN checkpoint loading for comparison runs against older experiments and baseline tests
* regression coverage for rules, tactical motifs, incremental parity, search behavior, and ML plumbing

Current verification:

* `python -m pytest -q` passes: `94 passed`
* local depth-7 opening/quiet-heavy suite on `2026-04-23`: `3.243s` average over 10 fixed positions, down from `7.888s` on the safety baseline
* fixed depth-6 class split on `2026-04-23`: `initial` `0.657s`, `tactical_midgame` `4.558s`, `quiet_midgame` `1.283s`, `restricted_threat_midgame` `0.153s`, `restricted_threat_repro` `0.158s`
* representative search summaries show tactics, quiescence, and proof work dominate wall time
* current documented ML baseline is root-policy on CPU, while quiet-value remains experimental

## What Is Outdated

These ideas or narratives should stop being treated as active priorities:

* the old phase-by-phase milestone story in the previous dev logs
* Stage 7 work packages that are already complete but still read like live roadmap items
* any suggestion that static evaluation speed is still the main bottleneck
* the previous policy-first board-CNN workflow as a primary direction; it is now legacy comparison-only

The bottleneck has shifted. On real search traces, the expensive buckets are tactical analysis, quiescence, and tactical proof/selectivity. The right next push is deeper and cleaner search, not more eval micro-optimization.

## How To Run

Human vs engine:

```powershell
python play_cli.py --mode human-vs-engine --depth 4 --time 10
```

Desktop GUI:

```powershell
python play_gui.py
```

Engine vs engine with instrumentation:

```powershell
python play_cli.py --mode engine-vs-engine --depth 4 --time 0.5 --device cpu --search-summary --time-trace
```

Fixed benchmark suite:

```powershell
python tools/search_benchmark.py --depth 6
```

Engine vs engine with the best current documented ML checkpoint:

```powershell
python play_cli.py --mode engine-vs-engine --depth 4 --time 0.5 --device cpu --model checkpoints/phase7_baseline.pt --ml-mode auto
```

Current strongest local equal-time profile:

```powershell
python play_cli.py --mode engine-vs-engine --depth 6 --time 0.5 --device cpu --model checkpoints/phase7_ml_bigger_dense.pt --ml-mode auto
```

Fair paired-color ML match suite against pure search:

```powershell
python tools/ml_match_suite.py --candidate-name Phase7Baseline --model checkpoints/phase7_baseline.pt --ml-mode auto --device cpu --depth 4
```

Fair paired-color search or search+ML suite with different candidate/baseline depths:

```powershell
python tools/ml_match_suite.py --candidate-name Depth6Dense --model checkpoints/phase7_ml_bigger_dense.pt --ml-mode auto --candidate-depth 6 --baseline-name Depth5 --baseline-depth 5 --time-controls 0.5
```

Diagnostic-only repeated batch from the initial position:

```powershell
python play_cli.py --mode engine-vs-engine --games 20 --shuffle-colors --seed 20260421 --seed-step 1 --temperature 0 --black-temperature 0 --white-temperature 0 --black-model checkpoints/phase7_baseline.pt --black-ml-mode auto --depth 4 --time 0.5 --device cpu
```

Generate root-policy training data:

```powershell
python -m ukumog_engine.ml.generate_root_policy_data --games 400 --play-depth 2 --label-depth 6 --label-time-ms 1500 --sample-every 2 --sample-start-ply 2 --temperature 0.8 --temperature-plies 8 --sample-top-k 4 --min-candidate-moves 4 --min-score-gap 50 --max-score-gap 4000 --output data/root_policy_v1.npz
```

Append new deduplicated root-policy samples to an existing dataset:

```powershell
python -m ukumog_engine.ml.generate_root_policy_data --games 200 --play-depth 2 --label-depth 6 --label-time-ms 1500 --sample-every 2 --sample-start-ply 2 --temperature 0.8 --temperature-plies 8 --sample-top-k 4 --min-candidate-moves 4 --min-score-gap 50 --max-score-gap 4000 --append-output --output data/root_policy_v1.npz
```

Inspect a dataset:

```powershell
python -m ukumog_engine.ml.inspect_data --data data/root_policy_v1.npz
```

Train a root-policy model:

```powershell
python -m ukumog_engine.ml.train_root_policy --data data/root_policy_v1.npz --output checkpoints/root_policy_v1.pt --epochs 12 --batch-size 128 --learning-rate 3e-4 --trunk-channels 56 --blocks 5 --policy-channels 24 --split-group auto --device auto
```

Experimental quiet-value run:

```powershell
python play_cli.py --mode engine-vs-engine --depth 4 --time 0.5 --device cpu --model checkpoints/quiet_value_v2.pt --ml-mode auto --learned-weight 0.1
```

Notes:

* `play_cli.py` now supports `--ml-mode auto` and `--device {cpu,cuda,auto}`. `auto` resolves quiet-value checkpoints to `quiet-value` and policy checkpoints to `root-policy`.
* `play_gui.py` keeps the existing engine configuration surface interactive: mouse play on the board, black/white engine panels, `Analyze` for current-position eval plus top moves, and `Start Autoplay` for engine-vs-engine games.
* the GUI now defaults to `human-vs-human` plain-board mode so you can build a position move by move, jump around the move timeline with `Start` / `Prev` / `Next` / `End`, and analyze the current board state without the engine auto-replying.
* analysis now surfaces a dedicated recommended move line and highlights that move directly on the board.
* the right-side engine settings area is both scrollable and collapsible to behave better at tighter window sizes or higher zoom.
* the GUI currently keeps board size fixed at `11x11`; the engine core, tests, and ML stack are still written around that size.
* `play_cli.py` supports `--search-summary`, `--stats-jsonl`, `--time-trace`, side-specific depth/time/model overrides, and opening sampling controls.
* `tools/search_benchmark.py` runs the fixed search-tuning position set: `initial`, `tactical_midgame`, `immediate_double_threat`, `quiet_midgame`, and the poisoned-line regression family `restricted_threat_midgame*` / `restricted_threat_repro*`.
* `tools/ml_match_suite.py` is the recommended strength check. It uses 8 deterministic opening-prefix positions plus `quiet_midgame` and `tactical_midgame`, pairs both color assignments at `0.5s` and `1.0s`, and now supports separate `--candidate-depth` and `--baseline-depth` overrides.
* `temperature=0` engine batches are useful for debugging or reproducing one opening, but they are no longer the recommended strength-evaluation command.
* the primary documented ML runtime path is now root-policy on CPU, with `checkpoints/phase7_baseline.pt` as the current documented checkpoint until a new `root_policy_v1` model clears the suite.
* the strongest local equal-time result currently on record is `checkpoints/phase7_ml_bigger_dense.pt` at `depth=6`, which scored `13-7` (`65%`) against pure `depth=5` on the full 20-game paired suite at `0.5s`.
* `ukumog_engine.ml.generate_root_policy_data` builds `root_policy_v1` from ambiguous root decisions rather than generic board snapshots, and `--append-output` deduplicates against existing `canonical_hashes` before extending a dataset.
* `ukumog_engine.ml.train_root_policy` trains a policy-only dense CNN over the existing 15-plane encoder with symmetry augmentation and canonical-hash validation splits.
* the quiet-value path remains available, but it is experimental and should be run with an explicit low blend such as `--learned-weight 0.1`.

## Compatibility

Legacy checkpoints and their client/runtime compatibility are intentionally preserved:

* `phase7_baseline.pt` stays in place as the baseline comparison model
* older policy/value checkpoints still load through the current evaluator path
* quiet-value tooling remains available for experimental comparisons
* the current command-line workflow remains supported alongside the new desktop GUI entrypoint

Archived legacy workflow notes:

* [LEGACY_ML_WORKFLOWS.md](docs/archive/LEGACY_ML_WORKFLOWS.md)
* [README_HISTORY_20260421.md](docs/archive/README_HISTORY_20260421.md)
* [SEARCH_ENGINE_DEVLOG.md](docs/SEARCH_ENGINE_DEVLOG.md)
* [GUI_DEVLOG.md](docs/GUI_DEVLOG.md)

## Archived Logs

The previous working logs were intentionally moved out of the repo root:

* [SEARCH_ENGINE_DEVLOG.md](docs/SEARCH_ENGINE_DEVLOG.md)
* [DEV_AGENDA.md](docs/archive/DEV_AGENDA.md)
* [DEV_NEXT_STAGE.md](docs/archive/DEV_NEXT_STAGE.md)
* [LEGACY_ML_WORKFLOWS.md](docs/archive/LEGACY_ML_WORKFLOWS.md)
* [README_HISTORY_20260421.md](docs/archive/README_HISTORY_20260421.md)
* [GUI_DEVLOG.md](docs/GUI_DEVLOG.md)

The archive files remain useful as historical notes. `README.md` stays the top-level project document, and `docs/SEARCH_ENGINE_DEVLOG.md` is the active search-specific planning log.
