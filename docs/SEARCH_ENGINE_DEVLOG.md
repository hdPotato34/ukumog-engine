# Search Engine Dev Log

Active search-engine roadmap and measurement log for the current repo state.

Historical context lives in [README_HISTORY_20260421.md](archive/README_HISTORY_20260421.md). That archive is still worth keeping around because it records how Stages 1-5 were reached, but this file is the active search-specific planning document going forward.

Repo surface note:

* a desktop GUI now ships via `play_gui.py`, but this log remains search-focused
* the GUI reuses the same shared controller/runtime path as `play_cli.py`, so search and ML compatibility still live in one place
* board size remains fixed at `11x11`; the GUI did not change the engine's board assumptions

## Current Standing

The engine is already structurally in a strong place:

* alpha-beta negamax with iterative deepening, PVS, aspiration windows, TT, killer/history ordering, and LMR
* tactical quiescence with a dedicated q-search TT
* conservative quiet-node futility and late-move pruning
* selective proof-solver gating for forcing branches
* incremental tactical bookkeeping with a `BASIC` vs `ORDERING` tactical-detail split and bulk future-win derivation
* benchmark and test coverage that is good enough to support search-first iteration

The big result from the Stage 1-5 work, as documented in the archived README, still holds today:

* proof search is no longer the primary bottleneck
* static evaluation is not the primary bottleneck
* the hard class is still the wide tactical midgame, where many non-forcing tactical continuations survive long enough to keep both tactical-summary generation and quiescence expensive

In other words, the search engine is already "Stockfish-like" in broad shape, but not yet in tactical throughput. The next gains will not come from one magic trick. They will come from driving down the effective branching factor and the tactical recomputation cost inside the surviving nodes.

Stage 7 status:

* a conservative non-PV quiescence narrowing pass has now landed
* the change is intentionally narrow: it only tightens safe-threat-only quiescence behavior away from PV lines and leaves forced blocks and double threats alone
* repeated local benchmarks on the representative tactical midgame were mixed and did not materially reduce node counts
* practical reading: this was a safe exploratory pass, not a breakthrough, and the Stage 6 baseline remains the last clearly large search-speed jump

Fine-grained q-search follow-up:

* the benchmark tool and search summaries now expose q-search mix data such as single-forced-block nodes, safe-threat-only nodes, and q-skip reasons
* on the representative tactical midgame at depth 7, the key q-search fact is now clear: the engine sees thousands of single-forced-block q-nodes and almost no safe-threat-only q-nodes
* practical reading: the previous safe-threat-focused Stage 7 heuristic was aimed at a real q-search category, but not at the dominant one on the benchmark that matters most
* a stronger single-forced-block chain follower has now landed on top of that profiling pass
* repeated local tactical-benchmark runs are still somewhat noisy, but they now show a more consistent structural win: q-search time drops materially when the benchmark is dominated by forced-block chains
* a follow-up tactical-summary plumbing cleanup has also landed: paired summaries and move-map generation now reuse already-known ordered candidate sets and candidate-bit masks instead of recomputing them through helper calls
* a deeper tactical bulk-pass has now landed: paired summaries derive black/white immediate wins, future wins, and poison sets in shared passes, and search ordering now uses count-only move-map data when it only needs lengths
* a broader ordering hot-path cleanup has now landed: the tactical-biased continuation-ordering experiment was backed out, static move-order bonuses are precomputed, and ranking now avoids extra per-move count dicts when snapshot move maps already exist
* another broad tactical-kernel pass has now landed: the hot 4-mask and 5-mask scans stop re-checking empty counts that are already implied by the incremental mask counts, and `move_map_counts` / `move_maps` now derive opponent winning replies and future wins from one shared 5-mask pass instead of two
* a strength-oriented ordering pass has now landed: quiet-history updates are finally scaled strongly enough to matter, and quiet moves searched before a quiet beta cutoff now receive history maluses instead of being forgotten

## Current Baseline

Verification run on `2026-04-22` after the quiet-history ordering pass:

* `python -m pytest -q` -> `78 passed`
* `python tools/search_benchmark.py --depth 7 --positions tactical_midgame quiet_midgame initial`
* `python tools/search_benchmark.py --depth 8 --time-ms 60000 --positions tactical_midgame`
* `python tools/ml_match_suite.py --candidate-name Depth6Dense --model checkpoints/phase7_ml_bigger_dense.pt --ml-mode auto --candidate-depth 6 --baseline-name Depth5 --baseline-depth 5 --time-controls 0.5`

Observed benchmark ranges from local spot checks under varying machine load:

* `initial` depth 7: about `1.36s` to `1.40s`, `7162` total nodes
* `quiet_midgame` depth 7: about `1.94s` to `2.00s`, `12807` total nodes
* `tactical_midgame` depth 7: about `6.1s` to `7.7s`, `33051` total nodes
* `tactical_midgame` depth 8 with a `60000ms` budget: about `18.8s` to `26.6s`, `109638` total nodes

Time-share observations from those same runs:

* on the representative tactical midgame at depth 7, tactics time is about `3.3s` to `4.2s` and quiescence time is about `2.8s` to `3.6s`
* at the depth-8 probe, tactics time is about `10.5s` to `15.2s` and quiescence time about `9.0s` to `13.1s`
* proof time stays near zero in all of those runs, which confirms the archived conclusion that proof search is no longer the wall
* the q-search mix changed meaningfully on the tactical benchmark because the engine is now finding earlier cutoffs: at depth 7 the run produced about `2755` single-forced-block q-nodes instead of the older `4529`
* this is no longer just a per-node cleanup stage; the quiet-history change materially reduced node count on the tactical benchmark

Interpretation:

* efficient depth 8 is already realistic on easy and quiet positions
* time-limited depth 8 is now much more comfortable on the wide tactical benchmark position than it was before the quiet-history pass
* the current blocker is still tactical summary generation plus quiescence expansion, not static eval
* inside q-search specifically, the current benchmark still looks much more like a forced-block chain problem than a broad safe-threat expansion problem
* the new quiet-history stage proves something important for strength: turning compute into Elo is not just about nominal depth, it is about getting stronger ordering signals so the extra search actually reaches the right lines
* pure search is still under-converting some of that gain at equal time, but search+policy guidance is now clearly more promising than either raw depth or ML-alone stories

## What Stage 1-8 Actually Bought Us

The archived README records the sequence well, and the repo still reflects it:

* Stage 1: light tactical snapshots stopped paying for move maps up front
* Stage 2: conservative futility and late-move pruning dramatically lowered quiet tail work
* Stage 3: proof-solver gating prevented deep non-PV tactical proof explosions
* Stage 4: q-search TT reuse cleaned up repeated horizon work
* Stage 5: tactical summaries moved to explicit `BASIC` vs `ORDERING` tiers, and exact move maps are now computed for a smaller ranked subset
* Stage 6: `BASIC` tactical summaries now derive future wins in one pass over the five-masks, and opponent-win-after bookkeeping no longer iterates masks per move
* Stage 7: non-PV quiescence now narrows safe-threat-only frontier handling more aggressively, but the measured gain so far is modest and noisy
* Fine-grained q-search pass: the engine now exposes q-search mix counters and has a direct single-forced-block chain follower, which clarifies that the tactical benchmark is dominated by forced-block chains and can cut q-search time materially on repeat runs
* Confirmed follow-up benchmark: the representative tactical midgame now finishes a `depth=8`, `time-ms=60000` probe in about `44.4s`
* Tactical-summary plumbing cleanup: internal incremental helpers now reuse ordered candidates and candidate-bit masks instead of recomputing them, which trims tactical overhead without changing search behavior
* Paired tactical bulk-pass: shared 5-mask and 4-mask scans now derive both colors' winning, future-win, and poison bookkeeping together, and ordering now uses count-only move-map data when full tuples would be thrown away
* Broader ordering hot-path cleanup: the tactical-ordering experiment was backed out for the live baseline, static move-order bonuses are now precomputed, and ranking no longer builds extra count dicts when snapshot move maps already exist
* Broad tactical-kernel cleanup: hot 4-mask and 5-mask scans now trust count-implied empties, and the single-color move-map path now derives opponent winning replies and future wins from one scan instead of two
* Strength-oriented quiet-history pass: quiet cutoffs now create a strong positive ordering signal, and earlier quiet misses in the same node now receive a negative signal instead of being treated neutrally

Those were the right stages. They got the engine from "depth 6 is painful" to "depth 7 is practical and depth 8 now fits inside a 60-second tactical benchmark budget with meaningful slack." The next phase should stay on that same axis rather than resetting the strategy, but it still needs stricter measurement because future node-reduction experiments can help one position class while hurting another.

## Present Bottlenecks

The current standing view, combining the archive with the fresh measurements, is:

1. `paired_tactical_summaries` is still a hot path even after the new paired bulk pass.
2. Within that path, the shared poison scan now looks like the clearest remaining broad tactical hotspot.
3. The one-pass move-map refactor helped the `move_map_counts` path, so that piece is less urgent than it was before this stage.
4. Quiescence is still expensive, but the dominant benchmark shape is now much clearer: many single-forced-block chains, not broad safe-threat-only tails.
5. The new quiet-history pass shows that node-count reduction can be worth keeping when it survives strength checks, not just speed checks.
6. Pure search still does not convert all of that extra tactical efficiency into a dominant equal-time edge over depth 5 by itself.
7. The strongest current match result comes from pairing the faster search with a root-policy checkpoint, which suggests the near-term determinant is search quality first and ML-guided root ordering second.
8. On the current depth-8 tactical benchmark, tactics time is still larger than quiescence time, so another tactical-kernel cleanup is still a plausible next broad-speed win.

That means the next curriculum should be broad-speed-first again: cheaper hot paths that help the whole engine, with node-reduction experiments treated as opt-in and heavily measured.

## Current Next Priority

The most practical next-step ordering from the new baseline is:

1. Keep the new candidate/baseline-depth match-suite workflow in the acceptance path for future strength work.
2. Re-profile the shared poison scan inside `paired_tactical_summaries` now that the move-map path is cheaper again.
3. Keep the fixed suite split by position class: opening-like, quiet, and tactical benchmarks should all stay visible for every future search change.
4. Treat node-reduction ideas as experiments behind both speed and strength guardrails rather than as the default next move.
5. If one more Python-side tactical-kernel pass does not move throughput enough, start a scoped native-acceleration experiment for the hottest tactical kernels rather than polishing around the edges.

## Strength Guardrails

Recent strength-side checks matter for the next decision:

* the latest quiet-history pass is not a low-risk per-node cleanup; it is a real node-shape change, so the strength checks matter more than they did on the last stage
* that risk showed up clearly: pure depth 6 and depth 7 no longer look automatically stronger than depth 5 just because the tactical benchmark got faster
* the bigger tactical-risk frontier is still move ordering: anything that changes which quiet moves are searched first can interact with futility pruning, late-move pruning, and proof gating
* that makes the new candidate/baseline-depth match-suite path part of the real regression gate, not just a nice-to-have diagnostic

Small fixed-time search checks on `tactical_midgame` are useful here:

* pure search with `max_time_ms=500` completed `depth=5` and chose move `59`
* pure search with `max_time_ms=1000` also completed `depth=5` and chose move `59`
* pure search with `max_time_ms=2000` completed `depth=6` and switched to move `36`
* the current `phase7_baseline.pt` root-policy run still completed only `depth=5` at `max_time_ms=2000` and stayed on move `59`

Practical reading:

* higher search efficiency does buy tactically meaningful extra depth once it crosses the next completed-depth boundary
* that benefit is not guaranteed at very short controls; at `500ms` there was no move change on the representative tactical benchmark
* raw nominal depth is still not the full story; the search needs strong enough ordering to make the extra effort land on the right lines
* the current root-policy models can now matter again when paired with the faster search, especially at the root where ordering quality is most valuable

Small opening-prefix equal-time checks reinforce that caution:

* on a small 8-game paired suite, pure depth 7 vs pure depth 5 split `4/8`, which is not enough to call it consistently stronger
* on the full 20-game paired suite at `0.5s`, `Depth6 + phase7_ml_bigger_dense.pt` scored `13/20` (`65%`) against pure depth 5
* on that same path, `phase7_baseline.pt` was only a modest edge over depth 5, so checkpoint choice still matters

Implication for the roadmap:

* the default next stage should still stay search-first, but "search-first" now includes explicit strength-gate measurements
* node-count reduction via tactical ordering is worth keeping only when it survives those strength gates
* ML is not the determinant by itself yet, but it is now a credible multiplier on top of the faster search
* before or alongside any future node-reduction pass, the repo should keep:
  a solver-backed forcing regression position beyond the immediate double-threat case
  and the tiny fixed-time opening-prefix strength gate so speed gains are not mistaken for strength gains

## Curriculum For Efficient Depth 8 And Beyond

### Next Broad-Speed Stage: Tactical Kernel Cleanup

Goal:
Lower per-node tactical cost across opening-like, quiet, and tactical positions together.

Priority ideas:

* re-profile the shared poison scan inside `paired_tactical_summaries` before touching quiescence again
* look for another paired-summary pass that reduces classification work without re-widening the code path for quiet positions
* keep the one-pass `move_map_counts` / `move_maps` helper, but only revisit it if new profiling says it is still disproportionately expensive
* keep static ordering work cheap and avoid reintroducing dynamic per-move bookkeeping into `_rank_moves`
* prefer improvements that leave node counts unchanged while reducing wall time across the full benchmark set

Why first:

* the latest stage clearly improved the deep timed tactical probe and real node count on the tactical benchmark
* the quiet benchmark did not improve by the same margin, which means the next broad pass should be chosen with that guardrail in mind
* the present tactical benchmark still spends more time in tactics than in quiescence
* broad ML data-generation throughput benefits more from cheaper nodes everywhere than from a narrower tactical-only tree win

Success target:

* move `initial`, `quiet_midgame`, and `tactical_midgame` forward together from the current baseline, with `quiet_midgame` explicitly treated as a no-regression gate

### Guarded Experimental Stage: Node-Count Reduction

Goal:
Lower total nodes on the hard tactical benchmark without adopting a narrower live baseline that hurts the broader engine.

Priority ideas:

* revisit tactical ordering only after the broad-speed baseline is stable
* reuse hash/PV preferences and tactical buckets carefully, but treat continuation-style ideas as experiments rather than defaults
* keep the solver-backed forcing regression and tiny fixed-time opening-prefix strength gate in the acceptance workflow
* reject any node-reduction pass that helps `tactical_midgame` while clearly regressing `initial` or `quiet_midgame`

Why second:

* the recent tactical-ordering experiment proved node reduction is possible
* it also proved that a faster tactical benchmark alone is not enough for the broader-speed target

Success target:

* lower `tactical_midgame` node count meaningfully while preserving or improving the opening-like and quiet benchmarks

### Escalation Stage: Native Boundary Decision

Goal:
Decide whether Python-level search work still has enough headroom, or whether the hottest tactical loops should be moved behind a faster implementation boundary.

Priority ideas:

* keep the fixed benchmark suite as the gate for all new search changes
* add one or two profiling snapshots around `paired_tactical_summaries` and q-search candidate generation before each major refactor
* only consider native acceleration after the tactical pipeline is cleaner, so we are not freezing a wasteful algorithm in a faster language

Why later:

* algorithmic wins are still being found
* native acceleration before the next tactical cleanup risks locking in the wrong work

## Immediate Next Moves

These are the next concrete steps I would take from the current repo state:

1. Keep q-search profiling in the standard benchmark output, but treat it as supporting data rather than the first knob to turn again.
2. Keep the new `tools/ml_match_suite.py` depth-split workflow in the standard strength gate for pure-search and search+ML candidates.
3. Re-profile `paired_tactical_summaries`, especially the shared poison path, on `initial`, `quiet_midgame`, and `tactical_midgame` after each tactical-kernel change.
4. Run the full position-class split after every broad-speed stage:
   depth 7 on `initial`, `quiet_midgame`, and `tactical_midgame`, plus the time-limited depth 8 tactical probe.
5. Keep node-reduction ideas in a side lane until the broad tactical kernels stop moving enough to matter.
6. If one more broad tactical-kernel pass still leaves ML data generation uncomfortably slow, promote the native-acceleration decision earlier instead of waiting for a perfect pure-Python endpoint.

## What Not To Do Next

The archived README was right to warn about these, and the current measurements do not change that:

* do not jump to null-move pruning yet
* do not assume more generic pruning is the best next move
* do not go back to static-eval micro-optimization as the main strategy
* do not try to solve search throughput by pushing ML further into the critical path

## Practical Goalposts

For the next search phase, the right definition of progress is:

* hold tactical correctness and search regressions steady
* keep `initial` depth 7 around or below about `1.35s`
* keep `quiet_midgame` depth 7 around or below about `2.0s`
* keep `tactical_midgame` depth 7 around or below about `6.5s`
* keep time-limited depth 8 tactical probes around or below about `20-27s` depending on local load
* keep `Depth6 + phase7_ml_bigger_dense.pt` clearly above `50%` against pure depth 5 on the full paired suite
* reject any future speed stage that wins only by helping the tactical benchmark while giving back too much on the opening-like or quiet cases

If those move in the right direction, then efficient depth 8 and beyond becomes a realistic engineering path rather than a slogan.
