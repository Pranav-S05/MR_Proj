Runtime and peak GPU memory on the same GPU type (Phoenix RTX A4000, shared machine).
MASt3R-SLAM memory is the nvidia-smi peak of its main process sampled every 30 s; its backend process is not
included (0.8 GB seen on 00, 5.4 GB on 02), so its total is higher. Wall time includes model loading and saving.
| system | sequence | frames | wall time | frames/s | peak GPU memory | GPU |
|---|---|---|---|---|---|---|
| VGGT front end, 32/16 sliding windows (loop windows and CPU back end not included) | 00 | 4541 | 14.1 min | 5.4 | 9.6 GB | NVIDIA RTX A4000 |
| VGGT front end, 32/16 sliding windows (loop windows and CPU back end not included) | 02 | 4661 | 14.0 min | 5.6 | 9.6 GB | NVIDIA RTX A4000 |
| VGGT front end, 32/16 sliding windows (loop windows and CPU back end not included) | 05 | 2761 | 9.2 min | 5.0 | 9.6 GB | NVIDIA RTX A4000 |
| VGGT front end, 32/16 sliding windows (loop windows and CPU back end not included) | 07 | 1101 | 3.6 min | 5.1 | 9.6 GB | NVIDIA RTX A4000 |
| MASt3R-SLAM (uncalibrated) | 00 | 4541 | 22.3 min | 3.4 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (calibrated) | 00 | 4541 | 13.8 min | 5.5 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (uncalibrated) | 02 | 4661 | 21.7 min | 3.6 | not recorded | NVIDIA RTX A4000 |
| MASt3R-SLAM (calibrated) | 02 | 4661 | 14.3 min | 5.4 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (uncalibrated) | 05 | 2761 | 9.3 min | 4.9 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (calibrated) | 05 | 2761 | 10.0 min | 4.6 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (uncalibrated) | 07 | 1101 | 5.1 min | 3.6 | 4.7 GB (main process) | NVIDIA RTX A4000 |
| MASt3R-SLAM (calibrated) | 07 | 1101 | 3.4 min | 5.3 | 4.7 GB (main process) | NVIDIA RTX A4000 |
