# Four-person work plan

These are assigned responsibilities, not a claim that each teammate authored the initial scaffold. The initial implementation was generated with Codex at Shreya's request on 22 November 2026. Git commits preserve actual commit times and the configured account identity. Each teammate should review and commit their own subsequent changes using their own Git identity.

| Member | Workstream | Files to own/review | Acceptance criteria |
|---|---|---|---|
| Meghna | Dataset, provenance, quality | `scripts/download_data.py`, `configs/euroc.json`, `docs/DATASET.md` | Verify archive checksum and sources; inspect units, cadence, missing values, reference coverage and dataset terms; freeze disjoint sequence splits; document sensor/reference uncertainty |
| Pranjal | Preprocessing and navigation | `src/io_drift/data.py`, `navigation.py`, `inference.py`, relevant tests | Confirm frame/quaternion conventions; no gap-crossing windows or reference features; independently verify gravity and integration; implement a streaming buffer and real initialization as next tasks |
| Shreya | Deep learning and training | `src/io_drift/model.py`, `scripts/train.py`, config training fields | Reproduce saved CNN; compare small GRU/TCN on validation only; log seeds/checkpoints; conduct ablations without selecting on test |
| Ritesh | Evaluation, plots, results | `scripts/evaluate.py`, `scripts/plot_results.py`, `metrics.py`, results report | Reproduce identical-endpoint baselines; report per-sequence velocity and position errors, initialization and failures; keep test frozen; measure deployment latency separately |

## Proposed next four workdays (future plan)

1. **Day 1:** Meghna validates references and splits; Pranjal reviews loaders; Shreya reproduces training; Ritesh verifies metric definitions and baseline protocol.
2. **Day 2:** Pranjal develops a causal orientation/initialization strategy; Shreya compares CNN against a small recurrent model on validation; Meghna audits another drone dataset; Ritesh adds stationary/short-trajectory diagnostics.
3. **Day 3:** Freeze the final configuration using validation. Ritesh runs final held-out evaluation; the others independently reproduce their modules and review results.
4. **Day 4:** Joint code review, report, presentation and limitations. Every member commits only work they actually performed or reviewed with a clear description.

## Honest contribution workflow

Each collaborator clones the repository, creates a feature branch, makes/reviews substantive changes, runs relevant tests, commits using their own configured identity, and opens a pull request. Use co-author trailers only for people who actually contributed and provide their correct GitHub-associated email. Never fabricate past activity or authorship. The initial scaffold's workstream commits are organizational divisions, not evidence of teammate contributions.

No assigned item is automatically complete simply because starter code exists. Full IO validation, generalization, deployment timing, streaming acquisition and improved orientation handling remain research work.

## Optimization implementation update

Codex implemented a bounded eight-configuration DL search plus two seed repetitions and training-derived fixed calibration. This does not change authorship assignments: Shreya should review/reproduce model selection, Pranjal should audit calibration and reference-initialization assumptions, Meghna should organize fresh external flight data, and Ritesh should independently verify matched-endpoint comparisons and position-baseline failures. None of these reviews is claimed complete on their behalf.
