# Audio-Visual Speech Separation via Bottleneck Iterative Network
Code repo for the 2025 ICMLWMLA workshop submission 

## Models
 * `profusion_mbt` or `prombt` are alternative/historical names of our proposed `Bottleneck Iterative Network`
 * `avlit`, `iia` (short for `IIA-Net`), `rtfs` (short for `RTFS-Net`) are benchmarks models studied in our work

## Training
 * sbatch scripts in `sbatch` folder follow the name of `<data>_<model>_<gpu-type>.sh`, so `sbatch/lrs3wham_profusion_mbt_a100.sh` trains the proposed `BIN` model on LRS3WHAM data using A100 GPU.
 * `BIN` implementation can be found here:  https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/fc94ca0238e0254d31950b711f0dd6ed41b15b5a/models/progressive_mbt.py#L94
 * Training main pipeline follow this abstract class: https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/260ff8112cfd343a20e092c84301e46fb9dc1dc0/engines/double_separation.py#L47
 * Training pipeline for `BIN` on LRS3 is here: https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/260ff8112cfd343a20e092c84301e46fb9dc1dc0/engines/double_separation.py#L621
 * Training pipeline for `BIN` on NTCD-TIMIT is here: https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/260ff8112cfd343a20e092c84301e46fb9dc1dc0/engines/double_separation.py#L1103
