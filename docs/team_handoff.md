> **Superseded (2026-10-24):** the README now holds the current setup and GPU run instructions. This page is kept for history.

# Team setup and first server task

This guide is for the existing project folder and a teammate starting on another machine. Research concerns already live in `proposal_review.md`; this guide does not duplicate that review. Read `README.md` for the code map and `status.md` for completed versus pending work.

## What becoming a Git repository changes

`MR_Proj` can itself be the repository root. `git init -b main` adds a hidden `.git` directory that stores version history and repository configuration. Existing files stay in their current locations. The Python environment and datasets continue to work from the same folder. No second project folder or fresh local clone is needed.

Git is local version tracking. GitHub is a place to host a copy for collaboration. Initializing Git does not upload files or give teammates access. [Git initialization documentation](https://git-scm.com/docs/git-init)

The local repository was initialized during this setup. The first commit and remote upload are still pending. To make the first commit from PowerShell:

On the original laptop, `.git` was created by the `CodexSandboxOffline` account. If your normal `Dell` PowerShell reports “detected dubious ownership,” run the following in that normal PowerShell first. This records a trust exception for this specific project in your own Git configuration; running it under the sandbox account would affect the wrong account.

```powershell
git config --global --add safe.directory C:/Users/Dell/Desktop/MR_Proj
```

Then run the first-commit commands below. Git refuses repository operations before the ownership exception, so a previously attempted `git add` or `git commit` may not have succeeded. [Git safe.directory documentation](https://git-scm.com/docs/git-config#Documentation/git-config.txt-safedirectory)

```powershell
Set-Location C:\Users\Dell\Desktop\MR_Proj
git status
git add .
git commit -m "Start robotics project with CPU foundations"
```

If Git requests your name/email, set your own values locally for this project, then retry the commit:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR GIT COMMIT EMAIL"
```

`.gitignore` excludes the local environment, downloaded datasets, caches, downloaded source snapshots, separate baseline clones, model weights and `.env` files. Ignored files remain on disk. The proposal, our source, documentation, tests, draft protocol and small example outputs are included.

## Uploading so teammates can clone

Create an empty GitHub repository under your account or team. A private repository is a reasonable starting choice for the team. Leave the remote README, license and gitignore options unchecked because this folder already has content. Replace the URL below with the repository URL you created:

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/MR_Proj.git
git push -u origin main
```

For the original laptop checkout, `origin` has now been configured as `https://github.com/Pranav-S05/MR_Proj.git`. Check with `git remote -v` and skip `remote add` there. A configured remote does not mean any files have been uploaded.

Authenticate with your own GitHub account when prompted. Then invite your friends as collaborators in the repository settings. For course submissions, make sure the TA has the required access. [GitHub's existing-code upload instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

Your laptop folder remains the working copy. Each friend clones the shared remote into their own machine. Push sends committed changes; pull retrieves teammates' committed changes. Each machine has its own virtual environment and datasets.

## Teammate bootstrap on a Linux server

Use Python 3.11 to match the checked CPU environment. After cloning your actual remote:

```bash
git clone https://github.com/YOUR_ACCOUNT/MR_Proj.git
cd MR_Proj
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m mr_cpu.lab
```

Download the KITTI files as described in the README (Data section). The test suite should pass all checks. The lab output remains synthetic plus reference-only evaluation checks. It does not run VGGT.

The local `.venv`, `data/` and `reference_material/` are intentionally absent from the clone. Recreate the environment and fetch data as above. The reference poses do not need to be copied from the laptop.

## Server owner: remaining first-week work

The first assignment is **one reproducible baseline smoke test with verified inputs and retained outputs**. It does not depend on us first settling the final novelty claim.

1. **Inventory the machine.** Record the GPU name and available VRAM, CPU RAM, free storage, OS, driver and CUDA environment, and job/time limits. A server may be CPU-only; access alone does not establish that VGGT inference is feasible. On Linux, `nvidia-smi`, `free -h` and `df -h .` help collect these facts. If `nvidia-smi` is unavailable, clarify the assigned compute before attempting installation.
2. **Set up the baseline separately.** Use a separate GPU environment following the official instructions for [VGGT-Long at our pinned commit](https://github.com/DengKaiCQ/VGGT-Long/tree/c160869d1d99c96bb227f414afb3bc68c29c9a76). Our CPU requirements do not install that system. Keep its clone under ignored `external/` or outside this project, and record its commit and checkpoint hash. Download only the model family actually being tested.
3. **Obtain a small KITTI fitting-sequence sample.** Use sequence 05, with matching RGB images, calibration, frame IDs, reference poses and a few LiDAR frames for validation. Inspect the download/storage options before obtaining full archives. Do not use primary test sequences 00/02 to tune the setup.
4. **Verify sensor and image conventions.** Confirm which input camera is used and correctly transform reference poses and LiDAR into that camera's coordinates. Save an overlay of projected LiDAR on an actual image, including the model's resize/crop handling. Synthetic projection tests do not complete this check.
5. **Try a small reconstruction.** Start with a short clip and a small window, for example 8 frames with 4 shared frames if memory allows; decrease it if needed. Disable loop detection for the first sequential smoke test. Set `Model.delete_temp_files` to `false` so the baseline retains its window predictions. Record the exact configuration. This is a feasibility run, not the final window-size choice or a kilometre-scale result.
6. **Check and retain outputs.** Verify C2W pose exports, finite positive scales/confidences, correct frame ordering, overlap correspondences and expected output dimensions. Record wall time, peak GPU memory and disk bytes. Preserve original local predictions; changing windows or adding loop-centric batches requires additional inference.
7. **Return a compact run record.** Commit a short Markdown report and the small configuration file in a named folder such as `results/server_smoke_05/`. Include software versions, source/checkpoint identifiers, frame range, preprocessing, window/overlap, dtype, success/failure, resource measurements and a pointer to dense outputs on server storage. Keep large arrays, datasets and checkpoints out of Git. Do not report a trajectory score until camera/frame association and metric settings have been verified.

Once the sequential smoke test passes, enable place recognition and loop handling in a separate recorded run. Loop-centric batches have their own memory requirements. The later unit-information parity check and reference seam-label implementation remain separate tasks; our tiny SciPy graph solver is not yet an upstream replacement.

## Keeping laptop and server work separate

The server owner can handle environment setup, real-data conventions, inference and storage. The laptop work can continue with the learning material, geometry/evaluation tools, the operational seam-label definition, uncertainty modelling and literature comparison.

After cloning, a teammate can work on a branch:

```bash
git switch -c server-smoke
# Edit the report/configuration and add only the intended small files.
git add results/server_smoke_05/
git commit -m "Record KITTI 05 baseline smoke test"
git push -u origin server-smoke
```

Open a pull request for review before merging into `main`. This makes the server changes easy to inspect while keeping a common project history. No need to send entire project folders or copy a Windows virtual environment to Linux.
