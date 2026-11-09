# Reproducing the results

This file explains how to rerun every result in `analysis/results/` and `analysis/figures/`. The reproduction study is described in `README.md`, and the full lab notebook (commands, outputs and judgments) is in `notes.md`.

## 1. Setup

Tested on Linux (Ubuntu under WSL2), Python 3.9.25, PyTorch 2.5.1 with CUDA 12.1, and an NVIDIA RTX 4070 SUPER (12 GB). The environment uses [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env          # only right after a fresh uv install (puts uv on PATH)
uv venv ~/.venvs/bsarec --python 3.9
source ~/.venvs/bsarec/bin/activate
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

torch is installed separately because it is the CUDA 12.1 build from the PyTorch index (see the header of `requirements.txt`). The upstream `bsarec_env.yaml` (PyTorch 1.8.1, CUDA 11.1) is the original authors' environment and was not used here.

`run_all.sh` activates `~/.venvs/bsarec` if it exists. To use another environment, set `VENV=/path/to/venv` or activate it yourself first.

## 2. Data

The LastFM dataset ships with the upstream code as `src/data/LastFM.txt`. Each line is one user: the user id followed by the ids of the artists they listened to, in interaction order (the file has no timestamps). The authors' preprocessing scripts are in `src/data/process/`; nothing needs to be downloaded. The split is leave-one-out: the last artist of each user is the test target, the second-to-last the validation target. `analysis/validate_data.py` checks the counts against the paper (1,090 users, 3,646 artists, 52,551 interactions; output in `analysis/results/validation.txt`).

## 3. Running

From the repository root:

```bash
bash run_all.sh --analysis-only   # about 20 s; regenerates every result from the committed logs, checkpoints and predictions
bash run_all.sh                   # trains any run that has not finished, saves predictions, then runs the analysis
```

The training logs, the per-seed predictions (`analysis/predictions/`) and the checkpoints the analysis loads are all committed. So in a fresh clone both commands skip all training and all prediction runs (a prediction run is redone only if its file in `analysis/predictions/` is missing) and only regenerate the results, which should leave `git status` clean. `--analysis-only` also works without a GPU.

A run counts as finished when its log in `src/output/` contains a test result. To retrain a run, delete its log, for example `rm src/output/SASRec_LastFM_s42.log`, and run `bash run_all.sh`. If a run was interrupted, its partial log is deleted and the run starts again from scratch, because the upstream code cannot resume training. All settings come from `analysis/hparams.sh` (lr 0.001, α 0.9, c 3, 1 attention head, seeds 42, 43, 44); everything else uses the upstream defaults.

A single training run, as `run_all.sh` calls it (from the repository root, with the environment active):

```bash
source analysis/hparams.sh && cd src
python main.py --data_name LastFM --model_type BSARec --lr $LR --alpha $ALPHA --c $C --num_attention_heads $HEADS --seed 42 --train_name BSARec_LastFM_s42
```

### Runtime (RTX 4070 SUPER, from the log timestamps)

| Step | Runs | Time |
| --- | --- | --- |
| Main runs: GRU4Rec, SASRec, BSARec × 3 seeds | 9 | 11 min |
| α and c sweep (seed 42) | 10 | 17 min |
| BSARec α = 0 and α = 1 × 3 seeds | 6 | 10 min |
| Depth sweep: SASRec, BSARec α = 0, BSARec at 4 and 8 layers × 3 seeds (patience 30) | 18 | 87 min (4 layers 30, 8 layers 57) |
| FMLPRec, BERT4Rec, Caser, DuoRec × 3 seeds | 12 | 45 min |
| FEARec × 3 seeds | 3 | 2.3 h |
| Predictions (`--do_eval`, eval only) and all analysis | 9 | 43 s |

Full training takes about 5.1 h; FEARec and the depth sweep account for most of it.

## 4. Expected differences from the paper

On a fixed machine and software stack, runs are deterministic. Rerunning a seed reproduced the earlier log and checkpoint exactly. Runs on a different stack can differ: the authors' own log (`src/output/BSARec_LastFM_best.log`, same settings, seed 42) reproduces the paper's numbers exactly, while our seed-42 run with the same settings gives NDCG@10 0.0381 instead of 0.0435. The authors' environment file (`bsarec_env.yaml`) specifies PyTorch 1.8.1 and CUDA 11.1 versus our 2.5.1 and 12.1; this software difference (and probably different GPUs) is the likely cause, since the configuration is identical. We therefore report the mean ± std over 3 seeds, and expect a rerun on other hardware to land within roughly that seed spread rather than on the same numbers.

## 5. Where each result comes from

All paths are under `analysis/`. Report parts refer to the final 6-page report (IEEE format).

| Report part | Result | Files | Script |
| --- | --- | --- | --- |
| Sec. IV Setup | Dataset statistics (the history-length and artist-popularity plots are extra) | `results/validation.txt`, `figures/seq_lengths.png`, `figures/artist_popularity.png` | `validate_data.py` |
| Sec. V, Table I | Main table (mean ± std over 3 seeds) and comparison with the paper's Table 2 | `results/main_table.csv`, `results/summary.txt` | `summarize.py` |
| Sec. V | Per-run test metrics, collected from the logs | `results/results.csv` | `collect_results.py` |
| Sec. V, Table I | Popularity baseline | `results/popularity.json` | `popularity.py` |
| Sec. V, Table I | Paper's Table 2 LastFM rows (entered by hand from the paper) | `results/paper.csv` | none |
| Sec. VI-A, Fig. 1 | α and c sensitivity sweep | `results/ablation.csv`, `results/ablation_summary.txt`, `figures/ablation.png` | `collect_results.py`, `plot_ablation.py` |
| Sec. VI-B, Table II | SASRec vs BSARec at α = 0, α = 1 and the default: loss effect vs Fourier-branch effect | `results/variants_table.csv`, `results/decomposition.txt` | `decomposition.py` |
| Sec. VI-C, Table III | NDCG@10 by history length, taste and test-artist popularity; share of popular artists in the top-10 | `results/groups_s42.txt`, `results/groups_s43.txt`, `results/groups_s44.txt` | `groups.py --seed <seed>` |
| Sec. VI-D | Learned β per layer (paper Fig. 7a) | `results/beta.csv`, `results/beta_summary.txt`, `figures/beta.png` | `beta.py` |
| Sec. VI-D | Oversmoothing per layer (paper Fig. 3) | `results/oversmoothing.txt`, `results/oversmoothing.csv`, `results/oversmoothing_groups.csv`, `figures/oversmoothing.png` | `oversmoothing.py` |
| Sec. VI-D, Table IV | Oversmoothing at 4 and 8 layers, and 2/4/8-layer comparison of the final block | `results/oversmoothing_L4.txt`, `results/oversmoothing_L8.txt` (with `.csv`, `oversmoothing_groups_L*.csv`), `figures/oversmoothing_L4.png`, `figures/oversmoothing_L8.png`, `results/oversmoothing_depth.txt` | `oversmoothing.py --layers <L>`, `oversmoothing_depth.py` |

Every script reads its paths from `analysis/config.py`, so it can also be run on its own from any directory, e.g. `python analysis/summarize.py > analysis/results/summary.txt`. `run_all.sh` shows the exact invocation of each one.

## 6. Change to the upstream code

The only change to the authors' code is a 3-line patch in `src/trainers.py`. During `--do_eval` it saves each user's top-20 predictions to `src/output/{Model}_preds.npy` and `_answers.npy`. Training and scoring are unchanged: re-evaluating a saved checkpoint reproduces its training-time test metrics exactly. `run_all.sh` copies these files to `analysis/predictions/{Model}_s{seed}_*.npy`, because each `--do_eval` overwrites them.
