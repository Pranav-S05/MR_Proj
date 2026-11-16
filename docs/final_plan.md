# Final plan: improve VGGT-Long with calibration (revision 2, 15 Nov 2026, 19:00)

This file replaces `PLAN.md` and `PLAN_broad_archive.md`. Follow it step by step. Every design choice is already
made here; if something is not written here, do not add it. Numbers so far: `notes/findings.md`.

Changes in revision 2 (after the 60/30 numbers and the user's instructions of 15 Nov evening):
1. The small setting is now chunk 30 / overlap 15, not 32/16. Reason to give when asked: 30/15 and 60/30 are the
   two chunk settings of VGGT-Long's own results table, and 75/30 is its released KITTI config. 32/16 was only the
   mid-submission setting; it stays as the DEVELOPMENT setting (where the method was designed) and leaves the final
   tables. Everything is rerun at 30/15.
2. Qualitative results are added (step F8: trajectories, scale-drift curves, maps).
3. The upstream VGGT-Long test is run at 30/15 as well as at 75/30.
4. The decision rules (F7) now give one label per setting. Thresholds are unchanged.
5. The method (section 2) is NOT changed. Section 1 explains why the 60/30 numbers do not require a change.
6. The GPU order puts 30/15 first (stage table in section 4).

Revision 2a (15 Nov, 19:15, three small additions, no change to method, runs or rules):
a. F1 has a rule for storage read errors, and records that drive 09's loop descriptors were recomputed.
b. Figure Q2 prints the end-of-drive scale of both lines, and F9 reports it (new item 17).
c. F7's list of notes gains one about the result without loop closure.

Revision 2b (16 Nov, 01:00, after all three settings of our pipeline and the first upstream runs; no change to the
method, the labels or the thresholds). Read section 4b: it replaces stages G5 to G7 of the stage table and adds
three items to F9. In short: finish R4 first; the full 75/30 upstream queue becomes a five-drive agreement check;
upstream C1 is dropped; the summary must show why the gain shrinks with chunk size and must say plainly that plain
VGGT-Long at 75/30 is still more accurate than calibrated 30/15.

Revision 2c (16 Nov, 10:45, after reading all results; no change to the method, the labels or the thresholds). Read
section 4c: a smaller RAM margin so drives 00 and 02 can finish, three corrections to how the summary presents the
upstream runs, and the cleanup list for our files in RAM.

Revision 3 (16 Nov, 14:00, closing; no new experiment, no GPU, nothing changes in the method, labels or results).
All runs of the plan are done and were re-checked independently (section 4d). One short laptop pass adds four
reporting items to the summary, then the study is closed. Read section 4d.

Revision 3a (16 Nov, evening; wording only, after checking the cited papers, `notes/citation_check.md`). VGGT-Long's
results table (v2) has four chunk rows (30, 60, 90, 120), each with overlap = half the chunk. Wherever this plan says
"the two chunk settings of VGGT-Long's own results table", read: 30/15 and 60/30 are two of its four rows; 75/30 is the
setting of v1 of the paper and of the released KITTI config. The paper's chunk-30 row averages 44.713 m; our plain
upstream runs at 30/15 average 44.70 m (per drive within 1.3%). The Phoenix folder was deleted on 16 Nov at the user's
request (`kavish/pixi`, `kavish/repro`, `kavish/mr_audit_arrays` kept).

## 0. Scope lock
Project: "Calibrated Confidence for Kilometre-Scale Feed-Forward SLAM". Baseline: VGGT-Long (Deng et al.).
Goal of this plan: a lower trajectory error than VGGT-Long on KITTI, obtained by calibration, with no extra
network inference, shown (a) in our pipeline at 30/15, 60/30 and 75/30 and (b) inside VGGT-Long's own code at
30/15 and 75/30, with quantitative tables and qualitative figures.

In scope: steps F1 to F10 below, nothing else.
Out of scope (do not start, do not queue, do not ask about): reproducing VGGT-Bridge or VGGT-Align, skip-edge
graphs, last-frame reference runs, Virtual KITTI, Pi3, the LiDAR audit (its arrays hold only absolute errors, so
it cannot be done), a rotation correction, new features, new correction models, tuning any constant in section 2,
any further run at 32/16.

Rules for every step:
1. Anything fitted is fitted on OTHER drives only (leave one drive out), at the SAME chunk setting.
2. Every drive is reported, plus the mean. No drive is dropped from a table.
3. The judge is ATE (Sim(3)-aligned RMSE in metres, `mr_cpu.evaluation.ate`), with loops unless stated.
4. The method was designed at 32/16 and its evaluation at 60/30 and 75/30 was fixed before those results existed
   (`findings.md` section 10). Nothing is changed after seeing a result. A bad result is reported as it is.
5. All work stays in `scale_study/`. `MR_Proj-main/` is read only.
6. Phoenix rules of `../Phoenix.md`: one GPU (GPU 0), tmux, pixi only, no tokens, never touch other users' jobs.
7. Course rule: the team writes the report text. Code, tables, figures and notes may be generated.

## 1. What is known on 15 Nov, 19:00
Development setting 32/16, 10 drives (`results/first_w32_o16.csv`), ATE with loops: VGGT-Long 41.73 m, calibrated
(`auto`, picked C4 on every drive) 25.13 m: 39.8% lower, geometric-mean ratio 0.678, 8 better / 2 worse (00 by 1.4%,
03 by 20%), Wilcoxon p = 0.020, bootstrap 95% interval of the mean change -52% to -16%. True-scale bound 13.78 m.

60/30, 10 drives, first full look (`results/prelim_first_w60_o30_10drives.csv`; 09 still to come), with loops:

| seam scale | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-Long (`none`) | 8.54 | 57.18 | 54.61 | 8.98 | 4.46 | 9.09 | 4.63 | 2.68 | 74.29 | 27.80 | 25.23 |
| C1 constant | 8.54 | 76.56 | 49.01 | 5.79 | 3.43 | 9.04 | 4.70 | 2.59 | 51.79 | 21.12 | 23.26 |
| C4 | 8.45 | 84.65 | 48.07 | 15.09 | 2.45 | 5.48 | 3.28 | 1.81 | 52.38 | 12.90 | 23.46 |
| `auto` | 8.45 | 99.22 | 48.07 | 15.09 | 2.45 | 5.59 | 3.08 | 2.20 | 52.38 | 13.95 | 25.05 |
| true scale (bound) | 8.24 | 27.59 | 41.17 | 3.09 | 0.82 | 3.70 | 2.85 | 2.30 | 40.91 | 10.04 | 14.07 |

Reading: 8 of 10 drives improve (geometric-mean ratio 0.80 for C4, 0.84 for `auto`), but 01 and 03 get clearly
worse, so the mean moves only 7% (C4) and 0.7% (`auto`), and no test is significant. By the rules of F7 this is
"no reliable gain" at 60/30 on the mean. Models fitted at 32/16 and applied unchanged at 60/30 fail (C4: 53 m): the
travel curve depends on the chunk size, so coefficients must always be fitted at the same setting.

Why the method is not changed because of this:
- Larger chunks leave less to gain. Chain drift without loops is 21x to 52x on the long drives at 32/16 but 4x to
  5x at 60/30, and the true-scale bound is 67% below VGGT-Long at 32/16 against 44% at 60/30.
- Drive 01 cannot judge any correction: it has 35 seams at 60/30 with a mean scale error of +0.004 and a standard
  error of 0.019 (67 seams, +0.009, standard error 0.014 at 32/16). Its ATE went down with C4 at 32/16 and up at
  60/30. That is noise, not a pattern to design against.
- Drive 03 is a real weakness: the depth-ratio cue under-corrects there at both settings, while the constant does
  well. It is reported by name. Fixing it now would be tuning on test drives.
- So the honest expectation for the final tables is: a clear improvement at 30/15, and at 60/30 and 75/30 most
  drives better but no reliable gain in the mean. F7 has the exact wording for each outcome.

Also done: oracle breakdown, weights study, simulation with real drive errors, middle-reference ablation and the
pixel-weight study, all at 32/16 (they are rerun at 30/15 below); `stats.py` bootstrap column; the nested C4
condition in `exp_weights.py`; the upstream patch with full C4 (`phoenix/vggt_long_calib.diff`, 30 added lines,
compiles) and the coefficient export, checked to reproduce our C4 corrections exactly.

## 2. The method, exactly ("calibrated seam scale"), unchanged from revision 1
VGGT-Long is unchanged except for one number per seam. For the seam between window i and the next window j, with
VGGT-Long's Sim(3) estimate (s, R, t) mapping window j into window i:

    log s_new = log s + c,      t_new = t + (s - s_new) * R @ m

where m is the mean camera centre of window j's overlap frames (so that point keeps its image). Graph weights stay
identity. Loop edges are not touched.

The correction c of the named method C4 has two stages, both fitted on the seams of the other drives at the same
chunk setting against the true log-scale error y (component 6 of the seam label, `lib.load(...)["err"]`):
- Feature x1 (depth-ratio scale minus point-cloud scale). For each shared frame: over pixels with depth
  confidence above 0.1 * min(median conf of both windows) in both windows and positive depth in both, the median
  of log(depth_i / depth_j). A frame counts if it has at least 30 such pixels. x1 = median over the counting
  frames, minus log s. Defined if at least 4 frames count.
- Stage 1 (fusion): c1 = a + b * x1, ridge regression of y on the standardised x1 with penalty 1e-3 * n, intercept
  not penalised. Where x1 is not defined, c1 = a0 = mean of y on the fitting seams.
- Feature x2 (travel per window, `speed`): log(path length of window j's overlap cameras / median depth of window
  j's valid points). Defined only if the overlap has at least 4 steps, all steps positive in both windows, and
  path length of window i's overlap cameras / median depth / s > 0.3.
- Stage 2 (calibration curve): isotonic regression (`mr_cpu.calibration.Isotonic`, `floor=-inf`, direction from the
  sign of the Spearman correlation on the fitting seams) of the stage-1 residual y - c1 against x2. Where x2 is
  defined, c = c1 + g(x2); otherwise c = c1.
Other versions kept only as table rows: C1 (c = a0), C2 (isotonic on x2 alone), C3 (stage 1 alone), C3i, C3p.
"auto" = for each test drive, pick among those six by an inner leave-one-out on the fitting drives (lowest mean
log ATE ratio to `none`, with loops). Code: `study/exp_main.py` (do not edit).

Headline row of every table of our pipeline: `auto`. Named single method, and the one patched into upstream: C4.

## 3. What exists (do not rewrite)
- Laptop, `scale_study/study/`: `lib.py`, `gpu_run.py`, `exp_profile.py`, `exp_oracle.py`, `exp_main.py`,
  `exp_weights.py` (with the C4 condition), `sim_real_drives.py`, `sim_bias_vs_weights.py`, `stats.py` (with
  `ci95`), `fig_ate.py`, `x5_pixel.py`, `x5_score.py`, `fit_upstream_coeffs.py` (exports C1, C3, C4).
  Python: `../MR_Proj-main/.venv/Scripts/python.exe` (called PY below), always run from `scale_study/`.
- Laptop, `scale_study/phoenix/`: `run_queue.sh`, `run_upstream_queue.sh`, `make_upstream_calib.py`,
  `vggt_long_calib.diff`, `fetch_seq_from_zip.py`.
- Laptop, `scale_study/runs/`: `first_w32_o16` and `middle_w32_o16` (00-08, 10), `first_w60_o30` (00-08, 10).
- Laptop, `scale_study/results/`: 32/16 development outputs and `prelim_*` files of 60/30. Leave them.
- Phoenix, `/scratch2/vpr-team/kavish/scale_study/` (called M below): same code, KITTI images 00-10 (drive 09
  re-fetched and verified), `runs/`, `logs/`, `external/VGGT-Long` at commit c160869 with weights and the patched
  `vggt_long_calib.py`.
- Phoenix tmux at 19:00: `kavish-scale-big` (now running `first` at 75/30) and `kavish-scale-09` (waiting).

## 4. Steps

### GPU stage table (GPU 0 only, one tmux session at a time, each started when the one before has ended)
All `run_queue.sh` and `run_upstream_queue.sh` calls skip finished runs, so a stopped stage is restarted with the
same line. D11 = `first:04 first:07 first:03 first:10 first:06 first:01 first:09 first:05 first:00 first:08 first:02`.
U(X) = `X:07 X:04 X:03 X:06 X:10 X:01 X:09 X:05 X:08 X:00 X:02` (for example U(C4) = `C4:07 C4:04 ...`).

| stage | command inside `cd M` (wrap as `tmux new-session -d -s kavish-g<N> "cd M && <command>"`) | needs first | about |
|---|---|---|---|
| G1 | `W=60 O=30 bash phoenix/run_queue.sh 0 "first:09" >> logs/run_big.log 2>&1` | - | 10 min |
| G2 | `W=30 O=15 bash phoenix/run_queue.sh 0 "D11" >> logs/run_big.log 2>&1` | - | 1.8 h |
| G3a | `W=30 O=15 V=first_w30_o15 bash phoenix/run_upstream_queue.sh 0 "plain:07 C4:07" >> logs/run_up.log 2>&1` | F6 6b at 30/15 | 12 min |
| G3b | same script, `"U(C4) U(plain)"` with `W=30 O=15 V=first_w30_o15` | 6c passed at 30/15 | 3.6 h |
| G4 | `W=75 O=30 bash phoenix/run_queue.sh 0 "D11" >> logs/run_big.log 2>&1` | - | 2.8 h |
| G5a | `W=75 O=30 V=first_w75_o30 bash phoenix/run_upstream_queue.sh 0 "plain:07 C4:07" >> logs/run_up.log 2>&1` | F6 6b at 75/30 | 15 min |
| G5b | same script, `"U(C4) U(plain)"` with `W=75 O=30 V=first_w75_o30` | 6c passed at 75/30 | 3.3 h |
| G6 | `W=30 O=15 bash phoenix/run_queue.sh 0 "<D11 with middle: in place of first:>" >> logs/run_big.log 2>&1` | - | 1.8 h |
| G7 | upstream `"U(C1)"` at `W=30 O=15 V=first_w30_o15`, then at `W=75 O=30 V=first_w75_o30` | G3b, G5b | 3.5 h |

G1 and G2 may be chained in one session. If the GPU is still not free when a stage would start (another user), wait;
never use a second GPU. If time runs out, the stages are dropped from the end (G7 first, then G6) and the summary
says which were not run.

### F1. Switch the running queue to the stage table (do this first)
1. `ssh -o BatchMode=yes phoenix 'tmux kill-session -t kavish-scale-09; tmux kill-session -t kavish-scale-big'`
   (both are ours; the 75/30 drive that is running is lost and redone in G4; finished 75/30 drives are kept).
2. Wait until `nvidia-smi` shows GPU 0 under 300 MiB, then start G1 and G2 in one session:
   `tmux new-session -d -s kavish-g2 "cd M && W=60 O=30 bash phoenix/run_queue.sh 0 'first:09' >> logs/run_big.log 2>&1; W=30 O=15 bash phoenix/run_queue.sh 0 'D11' >> logs/run_big.log 2>&1"`
3. Check with `tail -5 logs/run_big.log`. A stage is done when its tmux session is gone.
Failure rules (all stages):
- `GPU 0 still busy: stopping the queue`: wait for GPU 0 under 300 MiB, restart the same stage.
- A drive `FAILED` with `out of memory`: rerun it once on an empty GPU 0. If it fails again at 75/30, our pipeline
  has no 75/30 column: write that in `findings.md`, skip the 75/30 parts of F3, and run G5 with `V=first_w60_o30`.
  At 30/15 or 60/30 an out-of-memory error means another process is on the card: wait and rerun.
- A drive `FAILED` with `Input/output error` (storage fault, as on drive 09 on 15 Nov): fetch its images again
  with `python phoenix/fetch_seq_from_zip.py <seq> data/kitti/sequences/<seq>/image_2_new` (inside the pixi env;
  it must end with `0 unreadable`), swap the folder in (`mv image_2 image_2_bad; mv image_2_new image_2;
  rm -rf image_2_bad`), delete `runs/descriptors/<seq>*.npy` and every run folder of that drive that was
  produced after the fault, then rerun. (Done for 09: images re-fetched, `runs/descriptors/09.npy` deleted and
  recomputed in G1, so no 09 result uses the damaged copy.)
- Any other failure: read the last 20 lines of `logs/first_w<W>_o<O>_<seq>.log`; fix only a file or path problem and
  rerun; otherwise report the drive as missing.

### F2. Copy results back (after G2, again after G4 and G6)
From `scale_study/` in Git Bash:
```
for v in first_w30_o15 first_w60_o30 first_w75_o30 middle_w30_o15; do for s in 00 01 02 03 04 05 06 07 08 09 10; do
  [ -f runs/$v/$s/manifest.json ] && continue
  ssh -o BatchMode=yes phoenix "test -f /scratch2/vpr-team/kavish/scale_study/runs/$v/$s/edges.npz" || continue
  mkdir -p runs/$v/$s
  for f in edges.npz extras.npz manifest.json; do
    scp -q -o BatchMode=yes phoenix:/scratch2/vpr-team/kavish/scale_study/runs/$v/$s/$f runs/$v/$s/ || echo "MISSING $v $s $f"
  done; done; done
```
A setting is analysed only when all 11 drives of it are on the laptop (or a drive is declared missing by F1's rules).

### F3. Analysis per setting (laptop CPU, about 15 minutes per setting; run for a setting as soon as F2 has it)
For V in `first_w30_o15` (after G2), `first_w60_o30` (after G1), `first_w75_o30` (after G4):
```
$PY study/exp_profile.py --variant $V            > results/profile_$V.txt
$PY study/exp_oracle.py  --variant $V            > results/oracle_$V.txt
$PY study/exp_main.py    --variant $V            > results/main_$V.txt        # writes results/$V.csv
$PY study/exp_weights.py --variant $V            > results/weights_$V.txt
$PY study/sim_real_drives.py --variant $V        > results/sim_real_drives_$V.txt
```
After 30/15 exists, the transfer check (secondary), for V in `first_w60_o30`, `first_w75_o30`:
`$PY study/exp_main.py --variant $V --transfer-from first_w30_o15 --no-auto > results/transfer_$V.txt`
(writes `results/<V>_from_first_w30_o15.csv`; only its `T:` rows are used).
When all three settings are done:
```
$PY study/stats.py results/first_w30_o15.csv results/first_w60_o30.csv results/first_w75_o30.csv > results/stats.txt
$PY study/fig_ate.py results/first_w30_o15.csv results/first_w60_o30.csv results/first_w75_o30.csv
```
After G6: `$PY study/exp_profile.py --variant middle_w30_o15 > results/profile_middle_w30_o15.txt`.
Checks (stop and find the cause if one fails):
- The last number of each row of `results/profile_$V.txt` ("loops (VGGT-Long residual)") against upstream
  VGGT-Long from `MR_Proj-main/results/mid/upstream.csv`, drives 00 / 02 / 05 / 07: 30/15: 9.07 / 99.65 / 10.24 /
  3.98; 60/30: 8.63 / 52.78 / 9.85 / 2.66; 75/30: 10.83 / 55.90 / 10.15 / 2.84. Within 5% is a pass. A larger gap
  is written into `findings.md`; nothing is changed.
- 60/30, row `none`, drives 00 01 02 03 04 05 06 07 08 10 must read 8.54 57.18 54.61 8.98 4.46 9.09 4.63 2.68
  74.29 27.80 (they do not depend on fitting).
- `results/fig_ate.png` opens and has three panels with 11 drives each.

### F4. Bias against chunk size (no script, one small table)
From the three `results/profile_<V>.txt` files: setting, frames per step (W - O: 15, 30, 45), pooled mean seam
log-scale error (mean of the per-drive means weighted by seam count), that mean divided by frames per step, mean
drift factor. Add the 32/16 development row (step 16, pooled +0.0119). One sentence under it stating only what the
numbers show.

### F5. Pixel-weight study at 30/15 (Phoenix CPU, can run while the GPU works; after G2)
On Phoenix: edit `phoenix/x5.sh` so its last line is
`nice -n 10 python study/x5_pixel.py --variant first_w30_o15 --seqs 00 01 02 03 04 05 06 07 08 09 10 --workers 4`
and run `tmux new-session -d -s kavish-x5 "bash phoenix/x5.sh > logs/x5_30.log 2>&1"`. When the log ends with
`X5_DONE`: copy `runs/first_w30_o15/x5_pixel.npz` to the same path on the laptop and run
`$PY study/x5_score.py --variant first_w30_o15 > results/x5_first_w30_o15.txt`.
Report the four `loops+C1` means (base, calib, calib_d, depth_only). Rule: if `calib` or `calib_d` is at least 5%
below `base` and better on at least 7 of 11 drives, write "pixel weights in the alignment look promising, not part
of the method, future work"; otherwise "they do not help". In both cases also state whether `depth_only` (no
calibration) does about as well as `calib_d`: at 32/16 it did (30.61 against 29.50 m, base 34.52 m), so the gain
there came from the depth term, not from calibration. No further work on X5.

### F6. The test inside upstream VGGT-Long (stages G3, G5, G7)
Steps 6a (patch) and the code of 6b (export) are done. Before the first upstream run, one edit in
`phoenix/run_upstream_queue.sh` so the runs also keep a point-cloud sample for F8: after the line
`cp $OUT/camera_poses.txt $DST/; [ -f $OUT/loop_closures.txt ] && cp $OUT/loop_closures.txt $DST/` add
`    [ -f $OUT/pcd/combined_pcd.ply ] && python -W ignore -m mr_cpu.qualitative ply2npz $OUT/pcd/combined_pcd.ply $DST/pointcloud_sample.npz`
and copy the script to Phoenix.

Step 6b, per setting S in (`first_w30_o15`, `first_w75_o30`), when that setting has all drives on the laptop:
`$PY study/fit_upstream_coeffs.py --variant S`, then
`scp runs/upstream/coeffs_S.json phoenix:/scratch2/vpr-team/kavish/scale_study/runs/upstream/`.

Step 6c, the drive-07 check (stages G3a, G5a). Copy back
`runs/upstream/{plain,C4}_w<W>_o<O>/07/{camera_poses.txt,run.log.gz,wall_seconds}` and run 6e and the new file
`study/check_patch.py`:
```
"""python study/check_patch.py --variant first_w30_o15 --seq 07 --log runs/upstream/C4_w30_o15/07/run.log.gz"""
import argparse, gzip, re
import numpy as np
import lib
from exp_main import Scorer, load_variant
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True); ap.add_argument("--seq", default="07"); ap.add_argument("--log", required=True)
a = ap.parse_args()
seqs, data, feats, y, ok = load_variant(a.variant)
ours = Scorer(data, feats, y, ok).corr(a.seq, "C4_fuse_depth+speed", [o for o in seqs if o != a.seq])
pat = re.compile(r"Seam scale calibration: log s ([+-][0-9.]+) -> ([+-][0-9.]+)")
up = np.array([float(m.group(2)) - float(m.group(1)) for m in map(pat.search, gzip.open(a.log, "rt")) if m])
n = min(len(ours), len(up))
print("seams ours / upstream:", len(ours), len(up))
print("correlation %.3f, mean difference %+.4f" % (np.corrcoef(ours[:n], up[:n])[0, 1], (up[:n] - ours[:n]).mean()))
```
Pass conditions, all three, before the full queue of that setting (G3b, G5b):
1. `plain` ATE on 07 within 3% of the mid-submission upstream value (3.98 m at 30/15, 2.84 m at 75/30).
2. `check_patch.py`: same seam count, correlation above 0.9, absolute mean difference below 0.005. If not, compare
   x1 and x2 seam by seam between the patch and `lib.seam_features`, fix the patch (never `lib.py`), rerun
   `phoenix/make_upstream_calib.py` on Phoenix, delete the `C4` run of 07, repeat the check.
3. `C4` `wall_seconds` within 5% of `plain` (the correction adds no inference).

Step 6e, scoring. New file `study/score_upstream.py`:
```
"""python study/score_upstream.py --w 30 --o 15  ->  results/upstream_w30_o15.csv (read by study/stats.py)"""
import argparse, csv
import lib
from mr_cpu import evaluation, kitti
from mr_cpu.trajectory import read_poses
NAMES = {"plain": "none", "C1": "C1_const", "C4": "C4_fuse_depth+speed"}
ap = argparse.ArgumentParser(); ap.add_argument("--w", type=int, required=True); ap.add_argument("--o", type=int, required=True)
a = ap.parse_args()
rows = []
for tag, name in NAMES.items():
    for s in lib.ALL:
        f = lib.ROOT / "runs" / "upstream" / f"{tag}_w{a.w}_o{a.o}" / s / "camera_poses.txt"
        if not f.exists():
            continue
        est, gt = read_poses(f), kitti.gt_cam2_poses(lib.KITTI, s)
        assert len(est) == len(gt), (f, len(est), len(gt))
        ate = evaluation.ate(est[:, :3, 3], gt[:, :3, 3])
        rows.append([name, s, f"{ate:.4f}", f"{ate:.4f}", "", "", ""])
        print(tag, s, f"{ate:.2f}")
with open(lib.ROOT / "results" / f"upstream_w{a.w}_o{a.o}.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["method", "seq", "ate_chain", "ate_loops", "resid_mean", "resid_std", "auto_pick"]); w.writerows(rows)
```
After each of G3b, G5b, G7: copy back every `runs/upstream/<tag>_w<W>_o<O>/<seq>/{camera_poses.txt,wall_seconds}`,
run the script for that setting, then `$PY study/stats.py results/upstream_w<W>_o<O>.csv > results/stats_upstream_w<W>_o<O>.txt`.
`stats.py` needs the same drives in every method: if a run is missing, remove that drive's rows from the CSV for all
methods and say so in the summary.

### F7. Fixed decision rules (apply as written; do not soften or strengthen)
Five results, each `<method>` against `none`, ATE with loops, 11 drives:
R1 ours 30/15 (`auto`), R2 ours 60/30 (`auto`), R3 ours 75/30 (`auto`), R4 upstream 30/15 (C4), R5 upstream 75/30 (C4).
Each gets exactly one label:
- "clear improvement": mean ATE at least 15% lower, geometric-mean ratio at most 0.90, Wilcoxon p below 0.05, and
  no drive worse by more than 25%.
- "improvement": mean ATE at least 10% lower, geometric-mean ratio below 1, at least 7 of 11 drives better.
- "no reliable gain": anything else.
What may be claimed (pick the first line that applies):
1. R1 and R4 at least "improvement", and R3 and R5 at least "improvement": "calibrating the seam scale improves
   VGGT-Long at its small-chunk setting and at its released setting, in its own code, with no extra inference".
2. R1 and R4 at least "improvement", R3 or R5 "no reliable gain": "calibrating the seam scale improves VGGT-Long at
   its small-chunk setting (30/15), in our pipeline and in its own code, with no extra inference. At larger chunks
   k of 11 drives still improve (geometric-mean ratio g) but the mean does not, because of drives <names>; the gain
   shrinks as chunks grow, since VGGT-Long's scale drift shrinks too (table of F4)."
3. R1 at least "improvement", R4 "no reliable gain": "improves our reproduction at 30/15; not confirmed in upstream
   code", with the 6c agreement numbers.
4. R1 "no reliable gain": the result is the diagnosis (oracle, weights, simulation) plus the 32/16 development
   result, stated as such.
Always written next to the claim:
- The method was designed at 32/16 on these same KITTI drives; 30/15 is a rerun at VGGT-Long's nearest published
  setting, not an independent test. The 60/30 and 75/30 evaluation was fixed before its results existed.
- 11 drives of one dataset; corrections fitted on KITTI drives at the same chunk setting; models do not transfer
  between chunk settings (the `T:` rows).
- The drives that got worse, by name and amount, at every setting.
- KITTI ground-truth height errors on 02, 08, 09 (documented by SemanticKITTI).
- Without loop closure the gain is smaller and uneven: give the chain ATE of `none` and `auto` and name the
  drives where the calibrated chain over-corrects (item 17 of F9). Reason to state: the correction lowers the
  seam noise, but its remaining error keeps one sign along a drive, so it adds up in a chain and is removed only
  where loops exist. At 32/16 drive 02 went from shrinking about 30 times to growing about 10 times without
  loops (chain ATE 181.5 to 208.0 m), while with loops it went from 106.9 to 47.1 m.
Secondary (reported, never headline): C1 and C4 rows of our pipeline, chain ATE, the `T:` rows, upstream C1, the
C4 condition of `exp_weights.py` (if `calibrated` is at least 3% below `equal` in the mean and better on at least 7
of 11 drives, write "after the mean is calibrated, calibrated variances help too"; otherwise "they still do not").
Published numbers of other papers (VGGT-Bridge 22.61 m, VGGT-Align 19.99 m, VGGT-Long reruns 24.85 to 29.41 m, all
75/30) may appear in one row marked "copied from the papers, different hardware and reruns", for orientation only.

### F8. Qualitative results (laptop; three figures, fixed design)
Colours everywhere: ground truth `#1f2328` (dark), VGGT-Long `#d1242f` (red), calibrated `#2f6fdb` (blue).
All trajectories are drawn after Sim(3) alignment to ground truth (`evaluation.sim3_align`, `transform_points`),
top view = x against z of the camera positions, equal axis scale. Every figure is saved as PNG (200 dpi) and PDF in
`results/`. Each figure is opened and looked at before it is called done (no overlapping labels, all panels filled).

Figure Q1, trajectories, one file per setting. New file `study/fig_traj.py`:
```
"""python study/fig_traj.py --variant first_w30_o15   -> results/fig_traj_<variant>.png / .pdf
Top view of every drive: ground truth, VGGT-Long, VGGT-Long + calibrated seam scale (the `auto` pick), with loops."""
import argparse, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lib
from exp_main import Scorer, load_variant
from mr_cpu import evaluation
from mr_cpu.geometry import transform_points
GT, BASE, OURS, MUTED = "#1f2328", "#d1242f", "#2f6fdb", "#59636e"
def aligned(est, gt):
    return transform_points(evaluation.sim3_align(est[:, :3, 3], gt[:, :3, 3]), est[:, :3, 3])
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); a = ap.parse_args()
seqs, data, feats, y, ok = load_variant(a.variant)
sc = Scorer(data, feats, y, ok)
picks = {r["seq"]: r["auto_pick"] for r in csv.DictReader(open(lib.ROOT / "results" / f"{a.variant}.csv")) if r["method"] == "auto"}
fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
for ax, s in zip(axes.flat, seqs):
    d = data[s]; gt = d["gt"][:, :3, 3]
    a0, _, e0 = lib.solve(d, None, loops=True, return_nodes=True)
    z = lib.rescale_seams(d, sc.corr(s, picks[s], [o for o in seqs if o != s]))
    a1, _, e1 = lib.solve(d, z, loops=True, return_nodes=True)
    p0, p1 = aligned(e0, d["gt"]), aligned(e1, d["gt"])
    ax.plot(gt[:, 0], gt[:, 2], color=GT, lw=1.6, label="ground truth")
    ax.plot(p0[:, 0], p0[:, 2], color=BASE, lw=1.0, label="VGGT-Long")
    ax.plot(p1[:, 0], p1[:, 2], color=OURS, lw=1.0, label="+ calibrated seam scale")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_title(f"{s}: {a0:.1f} m -> {a1:.1f} m", fontsize=10)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ax in axes.flat[len(seqs):]:
    ax.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=10)
m = a.variant.split("_")
fig.suptitle(f"KITTI 00-10, chunk {m[1][1:]} / overlap {m[2][1:]}: top view (m), ATE with loops", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(lib.ROOT / "results" / f"fig_traj_{a.variant}.{ext}", dpi=200)
```
Run it for the three settings. The titles must equal the `none` and `auto` rows of `results/<variant>.csv`.

Figure Q2, scale drift along the drive, 30/15 only. New file `study/fig_drift.py`, same loading and the same two
solves as Q1 but with `loops=False` (the chain, where the drift is visible), not aligned. For each of the two
trajectories: step lengths of the estimate and of the ground truth; sums over a sliding window of 100 frames
(`np.convolve(x, np.ones(100), "valid")`); ratio estimate / ground truth where the ground-truth sum exceeds 1 m,
else NaN; divide by the median of its first 100 finite values. Plot ratio against frame number on a log y axis,
one panel per drive in the same 3 x 4 grid, red and blue as above, a thin grey line at 1, y label "map scale
relative to the start", panel title = drive, legend in the empty twelfth panel, figure title "Scale of the
reconstruction along the drive, without loop closure (chunk 30 / overlap 15)". Save as `results/fig_drift_first_w30_o15`.
The script (`study/fig_drift.py`, written) must also print one line per drive: drive, end scale of VGGT-Long,
end scale of the calibrated chain, where end scale = median of the last 100 finite values of the plotted curve.
Add these lines at the end of the drive loop: keep the two curves as `c0`, `c1` and
`print(s, "%.3f %.3f" % tuple(np.median(c[np.isfinite(c)][-100:]) for c in (c0, c1)))`; redirect the output to
`results/drift_first_w30_o15.txt`. The figure is used as it comes out: a drive where the blue line leaves 1 in
the other direction stays in the figure.

Figure Q3, maps from upstream VGGT-Long at 30/15, after G3b. Drives: 02, 08, and the drive with the worst
C4 / plain ATE ratio in `results/upstream_w30_o15.csv` (if that is 02 or 08, the next worst). Copy back
`pointcloud_sample.npz` of the `plain` and `C4` runs of those three drives. New file `study/fig_maps.py`: grid of
3 rows (drives) x 2 columns (VGGT-Long, + calibrated seam scale). Per panel: read `camera_poses.txt`
(`read_poses`) and the sample (`xyz`, `rgb`); T = `evaluation.sim3_align(est positions, gt positions)`; draw
`mr_cpu.qualitative._bev(ax, transform_points(T, xyz), rgb, gt_positions, title)` and add the aligned estimated
trajectory in blue (`lw=0.8`); title = `"<drive> <column name>: ATE <x.x> m"`. Figure size 11 x 14, saved as
`results/fig_maps_upstream_w30_o15`.

### F9. Write up the findings (files, not report text)
1. `results/SUMMARY.md` with exactly these parts, in this order:
   (1) the label of R1 to R5 and the claim line of F7 that applies, with its "always written" notes;
   (2) per setting (30/15, 60/30, 75/30): drives as columns, rows `none`, `C1`, `C4`, `auto`, `oracle`, with loops;
   (3) the same without loops;
   (4) `stats.txt` lines of `auto` and `C4` for each setting;
   (5) upstream tables at 30/15 and 75/30 (plain, C1 if run, C4) with their stats and the 6c check numbers;
   (6) cost table: per setting, peak GPU memory (largest `peak_gpu_mem_gb` in the drives' `manifest.json`), summed
       `wall_seconds`, mean ATE of `none` and of `auto`; one sentence comparing calibrated 30/15 with uncalibrated
       60/30 and 75/30, stating only what the numbers show;
   (7) oracle breakdown means per setting;
   (8) weights table means per setting (four bias conditions and the C4 condition) and F7's sentence on variances;
   (9) simulation with real drives at 30/15 (largest gain without bias; drives where weights are worse with bias);
   (10) F4's table; (11) F5's numbers and sentence; (12) middle against first reference at 30/15 (if G6 ran);
   (13) the `T:` rows; (14) drives that got worse, per setting; (15) list of figures; (16) stages not run, if any;
   (17) end-of-drive chain scale per drive at 30/15, VGGT-Long and calibrated (from `results/drift_first_w30_o15.txt`),
        with the count of drives whose scale ends closer to 1 after calibration and the names of those that end
        further from 1.
2. `notes/findings.md`: append section 11 with the same numbers in short form and the date. Add one line that the
   32/16 numbers of sections 9 and 10 are development results.
3. Figures: `results/fig_ate.*`, `results/fig_traj_*`, `results/fig_drift_*`, `results/fig_maps_*`.

### F10. Close
1. Phoenix: `tmux ls` shows no `kavish-*` session. Leave `M` in place until the user says the study is finished
   (when deleting, keep `kavish/pixi/`, `kavish/mr_audit_arrays/`, `kavish/repro/`).
2. Nothing is committed, pushed or copied into `MR_Proj-main/`.
3. Final message to the user: labels of R1 to R5 with their numbers, the claim line, the drives that got worse,
   what failed or was not run, and where the files are. No more than one screen.

## 4b. Revision 2b: remaining work after the disk filled (overrides the stage table and section 5 where they differ)

State at 01:00 on 16 Nov. Our pipeline is complete at 30/15, 60/30, 75/30 (11 drives each): R1 "improvement"
(45.66 to 35.35 m, -22.6%; C4 alone 28.17 m, -38.3%), R2 and R3 "no reliable gain". Upstream at 30/15: plain on 6
drives, C4 on 7, all within about 1% of our pipeline; patch check passed (72 of 72 seams, correlation 1.000).
/scratch2 is full because of other users; `kavish-resume` waits for space.

Accepted deviations of the implementer (keep them): temporary files of upstream runs in RAM with the two guards
(`SHM=1`), the per-drive disk rule, the changed stage order, the deleted overlap maps (so X5 cannot be rerun; it
is finished), the resume script, the RAM-against-disk check on drive 04.

Remaining GPU work, in this order, each only when /scratch2 has at least 10 GB free and GPU 0 is idle:
1. Rest of upstream 30/15 (R4): plain 09; C4 and plain for 05, 08, 00, 02. This is the most important remaining
   item: R4 decides the claim line.
2. `plainram:04` (RAM against disk). Pass: ATE within 1% of the disk run of plain 04 (11.22 m). Report the number.
3. Upstream at 75/30 as an agreement check only: plain and C4 for 07, 04, 03, 06, 10 (G5a's three conditions on
   07 still apply). The full 11-drive queue G5b is not run. Reason: R3 is already "no reliable gain", so claim
   line 2 of F7 applies whatever R5 would be, and upstream has matched our pipeline within about 1% on every run.
   R5 is reported as "not run in full; agreement with our pipeline on five drives: <numbers>".
4. G6 (`middle` at 30/15, with `SUB=0`), 11 drives.
G7 (upstream C1) is dropped. If /scratch2 has not recovered by 18:00 on 16 Nov, stop waiting: kill
`kavish-resume`, finalise with what exists, and list every missing run in part 16 of the summary. If only R4 is
incomplete then, report R4 on the drives that have both runs, state the count, and label it "incomplete".

Additions to F9 (`results/SUMMARY.md`); numbers come from files that already exist:
- Part 10 gains a table of the per-drive mean seam scale error at the three settings (from
  `results/profile_<V>.txt`) and this reading, with the numbers filled in: at 30/15 every drive has a positive
  bias (+0.006 to +0.046), so one correction learned on other drives points the right way for all of them; at
  75/30 the drive means run from -0.006 to +0.032 with three drives at or below zero (01, 03, 05) and the pooled
  bias is +0.0086 with a standard error of 0.0039. At large chunks the seam scale error is mostly random from seam
  to seam, not a shared bias, and a calibration of the mean cannot remove random error. The oracle rows (true
  scale about 15 m at every setting) show the room that a better seam scale estimate would still have.
- Part 6 (cost) must state the comparison across settings in plain words: plain VGGT-Long at 75/30 (24.61 m) is
  more accurate than calibrated 30/15 (35.35 m with `auto`, 28.17 m with C4). What calibration offers at 30/15 is
  a smaller gap at lower peak memory (9.5 GB against 13.1 GB), not a better result than large chunks.
- Part 1 must note for R1 that `auto` chose C1 for drive 01 only (156.67 m; C4 would give 77.64 m) and C4 for the
  other ten, and give the C4 row's statistics next to it. `auto` stays the headline of our pipeline; R4 is C4, as
  fixed before.

Nothing else changes. In particular: no new correction, no retuning for drives 01 and 03, no switch of the
headline row.

## 4c. Revision 2c: finishing R4 and presenting the upstream runs (overrides 4b where they differ)

State at 10:40 on 16 Nov: everything of 4b is done except upstream 30/15 on drives 00 and 02. C4:00 started when
RAM allowed and is running; plain:00, C4:02 and plain:02 remain. Upstream 30/15 on the nine finished drives
(01 03 04 05 06 07 08 09 10): plain mean 42.55 m, C4 mean 27.88 m; C4 better on 7, equal on 09, worse on 07 (x1.10).

1. RAM margin for the three remaining runs. Change the start condition of RAM-mode runs from "peak + 40 GB
   available" to "peak + 30 GB available"; the watchdog stays at 20 GB. Reasons: over the morning the available RAM
   at the retries was 57 to 64 GB (other users' use is steady), the runs need 67 GB under the old margin and 57 GB
   under the new one, and the measured RAM use of a run agrees with the estimate (about 27 GB for drive 08). With
   the new margin at least 27 GB stay available during a run, and our run is the one that is stopped if that falls
   below 20 GB. How: `sed -i 's/peak + 40/peak + 30/g' /dev/shm/kavish_run_upstream_queue.sh` on Phoenix (`sed -i`
   writes a new file, so the running instance is not disturbed; the retry loop of `kavish-ramq2` picks the new
   value up at its next attempt) and the same edit in `phoenix/run_upstream_queue.sh` on the laptop. Do not restart
   or kill `kavish-ramq2` while a run is in progress. The stop rule stays: if R4 is not complete at 18:00, finalise
   with the drives that have both runs and label R4 "incomplete".
2. Summary part 5, 30/15: next to the upstream rows add our pipeline's rows for the same drives (`none` and C4 of
   `results/first_w30_o15.csv`), so the agreement between VGGT-Long's code and our pipeline is visible per drive.
   Until all 11 drives have both runs, print the paired statistics with the heading "interim, <n> of 11 drives".
3. Summary part 5, 75/30: this is an agreement check on five short drives, not a result. Print the five drives with
   upstream plain, our `none`, upstream C4, our C4 side by side, and the sentence "R5 was not run in full; on these
   five drives upstream and our pipeline agree within <largest gap> and C4 moves both in the same direction". Do
   NOT print paired statistics or a mean change for these five drives: they are the short drives only, and their
   mean says nothing about 75/30 as a whole (our full 75/30 result R3 is "no reliable gain").
4. Condition 3 of step 6c is replaced by what was actually checked and passed: the plain and the C4 log of drive 07
   show the same number of chunks, network calls and alignments (30/15: 73, 76, 73; 75/30: 24, 27, 24). Wall time is
   reported next to it without a pass band, because it depends on the load of the shared machine.
5. Cleanup at F10, in this order, only after `kavish-ramq2` has ended or been stopped at 18:00: (a) copy to the
   laptop every `camera_poses.txt`, `wall_seconds`, `run.log.gz`, `loop_closures.txt` still only in
   `/dev/shm/kavish_out`, and `pointcloud_sample.npz` of the three map drives of Q3 only; (b) copy
   `/dev/shm/kavish_ramq*.log` and `/dev/shm/kavish_logs/` into `scale_study/phoenix_logs/` on the laptop; (c) on
   Phoenix `rm -rf /dev/shm/kavish_*`, then check that `ls /dev/shm | grep kavish` prints nothing and `tmux ls`
   shows no `kavish-*` session. The folder on /scratch2 is left as it is (the user decides when to delete it).
6. Q3 is drawn when R4 is complete or at the stop time, with the rule as written (02, 08 and the worst C4 / plain
   ratio; if 02 has no pair, use 08, the worst ratio, and the best ratio among the remaining drives, and say so in
   the caption).

Expected outcome, stated as expectation: if drives 00 and 02 behave in upstream as they do in our pipeline (00:
9.0 to 10.5 m, 02: 110 to 48 m), R4 meets every condition of "clear improvement" and claim line 2 of F7 applies.

## 4d. Revision 3: closing pass and conclusion

Final labels (verified): R1 "improvement" (our 30/15, `auto`: 45.66 to 35.35 m), R2 and R3 "no reliable gain", R4
"clear improvement" (upstream VGGT-Long 30/15 with C4: 44.70 to 28.09 m, -37.2%, 8 better / 2 worse, Wilcoxon
p 0.019), R5 not run in full. Claim line 2 of F7 applies.

Independent re-check done on 16 Nov (keep these lines in `notes/findings.md`):
- every drive's upstream coefficients were fitted on the ten other drives (the `fitted_on` lists of both
  coefficient files); plain logs contain no calibrated seam, C4 logs one per seam;
- all 22 upstream 30/15 trajectories rescored with a separately written Umeyama alignment: same ATE as the project
  code; the statistics of R4 recomputed from those scores: same values;
- upstream plain at 30/15 reproduces the mid-submission upstream values to three decimals (00 9.071, 02 99.646,
  05 10.236, 07 3.984), and these match the VGGT-Long paper's chunk-30 numbers quoted in
  `MR_Proj-main/docs/status.md` (00 9.06, 02 99.95, 05 10.20, 07 4.00): the baseline is faithful;
- the summary's table means equal the CSV files.

Closing pass (laptop only; append to `study/make_summary.py` and regenerate; about 15 minutes):
1. Secondary metrics, as in the mid-submission tables. New file `study/rel_errors.py`: for upstream 30/15 (plain
   and C4, from the saved `camera_poses.txt`) and for our pipeline at the three settings (`none`, C4 and `auto`,
   trajectories from the same solves as `study/fig_traj.py`), compute `mr_cpu.evaluation.kitti_relative_errors(est,
   gt)` and write `results/rel_errors.csv` with columns source, setting, method, seq, t_rel_percent,
   r_rel_deg_per_100m. Summary part 18: one table per source and setting with the per-drive `t_rel_percent` and the
   mean, and the mean `r_rel`. These numbers are reported as they come out, get no label, and change no claim; the
   judge stays ATE. One sentence states in which direction each mean moved.
2. Summary part 1, under R4, two robustness lines computed from `results/upstream_w30_o15.csv`: the median ATE
   (plain 19.53, C4 18.04) and the mean without the two largest gains, drives 02 and 08 (28.00 to 22.39 m, -20%).
   Wording: "the mean gain is carried by the long, high-drift drives; without the two largest it is still -20%; the
   median moves little because the short drives were already accurate".
3. Summary part 12: add the comparison sentence, computed, not typed: pooled seam bias of `middle` against `first`
   at 30/15, the number of drives where `middle` is larger, and the mean ATE with and without loops of both
   (expected +0.0218 against +0.0136, 11 of 11, 52.43 against 45.66 m, 95.17 against 77.48 m).
4. Summary part 5: one line that upstream plain reproduces the mid-submission values and the paper's chunk-30
   numbers (the four drives above).
Then append "closing pass done" with the date to `notes/findings.md`. Nothing else is run.

The study is then closed. What is NOT part of it and stays with the team (do not start any of these unasked):
- the report and any paper text (course rule: written by the team);
- whether the correction and the tables move into `MR_Proj-main/` for the final submission (so far: no);
- deleting `/scratch2/vpr-team/kavish/scale_study/` on Phoenix (all results needed are on the laptop; keep
  `kavish/pixi/`, `kavish/mr_audit_arrays/`, `kavish/repro/`);
- checking every quoted number of other papers against their PDFs before it is cited;
- any follow-up (a second dataset, a lower-noise seam scale estimate for large chunks, another model).

## 5. Order of work and time line (start 19:00 on 15 Nov)
| about | GPU 0 | laptop / Phoenix CPU meanwhile |
|---|---|---|
| 19:00-21:00 | F1, then G1 + G2 | F3 for 60/30 once 09 is copied; write `score_upstream.py`, `check_patch.py`, `fig_traj.py`, `fig_drift.py`, `fig_maps.py`; edit `run_upstream_queue.sh` |
| 21:00-21:30 | G3a | F2, F3 for 30/15, 6b at 30/15, start F5; then 6c |
| 21:30-01:15 | G3b | Q1 and Q2 at 30/15, Q1 at 60/30 |
| 01:15-04:00 | G4 | score upstream 30/15, Q3 |
| 04:00-04:30 | G5a | F2, F3 for 75/30, 6b at 75/30, 6c, transfer rows, F4, `stats.txt`, `fig_ate` |
| 04:30-08:00 | G5b | Q1 at 75/30 |
| 08:00-10:00 | G6 | score upstream 75/30 |
| 10:00-13:30 | G7 | F7, F9 (G6 and G7 numbers are added when they arrive) |
| end | - | F10 |

## 6. The story for the report and the viva
1. Hypothesis: calibrated confidence improves VGGT-Long's pose graph.
2. We calibrated the confidence. Calibration is good, the trajectory does not improve, and oracle weights do not
   either.
3. Why: the graph is a chain with few loops, and the main error is a one-directional bias in the seam scale. No
   variance weight can fix a bias (oracle breakdown, weights table, simulation with the real drives).
4. So we calibrated the mean of the seam scale with the same tools (a fused estimate and an isotonic curve, fitted
   on other drives), with no extra inference.
5. Result: the tables and figures of F8 and F9 with the claim line of F7. If asked "why 30/15": it is VGGT-Long's
   own small-chunk setting, where its scale drift is largest; at its larger chunks the drift and therefore the gain
   are smaller. Supporting studies from before stay as they are (false loops with DCS, MASt3R-SLAM on KITTI, KITTI
   ground-truth heights).
