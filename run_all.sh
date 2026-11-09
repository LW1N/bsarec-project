#!/usr/bin/env bash
# Reproduce every result in analysis/results/ and analysis/figures/.
#
# Usage:
#   bash run_all.sh                  # train missing runs, save predictions, run all analysis
#   bash run_all.sh --analysis-only  # skip training and prediction; regenerate results
#                                    # from the logs (and checkpoints/predictions, if present)
#
# Training is skipped for any run whose log already exists in src/output/, so the
# script is safe to re-run and resumes where it stopped. To retrain everything,
# start from a fresh clone. Settings come from analysis/hparams.sh. See REPRODUCE.md
# for setup and expected runtimes (full training is several hours on one GPU).
#
# Environment: activates $VENV (default ~/.venvs/bsarec) if it exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A="$ROOT/analysis"; R="$A/results"; P="$A/predictions"; OUT="$ROOT/src/output"
VENV="${VENV:-$HOME/.venvs/bsarec}"

ANALYSIS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --analysis-only) ANALYSIS_ONLY=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [[ -f "$VENV/bin/activate" ]]; then set +u; source "$VENV/bin/activate"; set -u; fi
source "$A/hparams.sh"                       # LR ALPHA C HEADS SEEDS
ABL_SEED=42                                  # the alpha/c sweep is single-seed
ALPHAS="0.0 0.1 0.3 0.5 0.7 1.0"             # sweep values besides the default ALPHA
CS="1 5 7 9"                                 # sweep values besides the default C
DEPTHS="4 8"                                 # oversmoothing depth sweep, besides the default 2 layers
DEPTH_PATIENCE=30                            # deeper runs still improve after 10 flat epochs (default patience)
BASELINES="FMLPRec BERT4Rec Caser DuoRec FEARec"  # extra paper baselines (upstream defaults)
GROUP_MODELS="GRU4Rec SASRec BSARec"         # models in the per-user group analysis

step() { echo; echo "=== $*"; }

# Run src/main.py on LastFM unless <name>.log has a test result (a finished run).
# A partial log left by an interrupted run is deleted first: the upstream logger appends.
# Args: <name> <main.py args...>
run_main() {
  local name=$1 log="$OUT/$1.log"; shift
  if grep -qs "Test Score" "$log"; then echo "skip $name (finished)"; return; fi
  if [[ -f $log ]]; then echo "removing partial log of $name"; rm "$log"; fi
  echo "run  $name"
  local err; err=$(mktemp)
  if ! (cd "$ROOT/src" && python main.py --data_name LastFM --lr "$LR" --train_name "$name" "$@" >/dev/null 2>"$err"); then
    echo "FAILED $name; last error lines:" >&2; tail -20 "$err" >&2; rm -f "$err"; exit 1
  fi
  rm -f "$err"
}

bsarec_args() { echo --model_type BSARec --num_attention_heads "$HEADS"; }

if [[ $ANALYSIS_ONLY == 0 ]]; then
  step "Main runs: GRU4Rec, SASRec, BSARec x seeds $SEEDS"
  for seed in $SEEDS; do
    for m in GRU4Rec SASRec; do
      run_main "${m}_LastFM_s$seed" --model_type "$m" --num_attention_heads "$HEADS" --seed "$seed"
    done
    run_main "BSARec_LastFM_s$seed" $(bsarec_args) --alpha "$ALPHA" --c "$C" --seed "$seed"
  done

  step "Ablation: alpha and c sweeps (seed $ABL_SEED)"
  for a in $ALPHAS; do run_main "abl_alpha$a" $(bsarec_args) --alpha "$a" --c "$C" --seed "$ABL_SEED"; done
  for c in $CS; do run_main "abl_c$c" $(bsarec_args) --alpha "$ALPHA" --c "$c" --seed "$ABL_SEED"; done

  step "Variants: BSARec with alpha = 0 and alpha = 1 x seeds"
  for seed in $SEEDS; do
    run_main "BSARecAlpha0_LastFM_s$seed" $(bsarec_args) --alpha 0.0 --c "$C" --seed "$seed"
    run_main "BSARecAlpha1_LastFM_s$seed" $(bsarec_args) --alpha 1.0 --c "$C" --seed "$seed"
  done

  step "Depth sweep: SASRec, BSARec alpha = 0, BSARec at $DEPTHS layers x seeds (patience $DEPTH_PATIENCE)"
  for L in $DEPTHS; do
    for seed in $SEEDS; do
      run_main "SASRecL${L}_LastFM_s$seed" --model_type SASRec --num_attention_heads "$HEADS" --num_hidden_layers "$L" --patience "$DEPTH_PATIENCE" --seed "$seed"
      run_main "BSARecAlpha0L${L}_LastFM_s$seed" $(bsarec_args) --alpha 0.0 --c "$C" --num_hidden_layers "$L" --patience "$DEPTH_PATIENCE" --seed "$seed"
      run_main "BSARecL${L}_LastFM_s$seed" $(bsarec_args) --alpha "$ALPHA" --c "$C" --num_hidden_layers "$L" --patience "$DEPTH_PATIENCE" --seed "$seed"
    done
  done

  step "Baselines: $BASELINES x seeds"
  for m in $BASELINES; do
    for seed in $SEEDS; do
      run_main "${m}_LastFM_s$seed" --model_type "$m" --num_attention_heads "$HEADS" --seed "$seed"
    done
  done

  step "Predictions for the group analysis (--do_eval on saved checkpoints)"
  mkdir -p "$P"
  for seed in $SEEDS; do
    for m in $GROUP_MODELS; do
      if [[ -f "$P/${m}_s${seed}_preds.npy" ]]; then echo "skip $m seed $seed (predictions exist)"; continue; fi
      # clear the previous attempt so a stale {Model}_preds.npy can never be copied
      rm -f "$OUT/predict_${m}_s$seed.log" "$OUT/${m}_preds.npy" "$OUT/${m}_answers.npy"
      extra=(); [[ $m == BSARec ]] && extra=(--alpha "$ALPHA" --c "$C")
      run_main "predict_${m}_s$seed" --model_type "$m" --num_attention_heads "$HEADS" "${extra[@]}" \
               --do_eval --load_model "${m}_LastFM_s$seed"
      # the evaluator patch in src/trainers.py writes {Model}_preds.npy / _answers.npy; keep a per-seed copy
      cp "$OUT/${m}_preds.npy" "$P/${m}_s${seed}_preds.npy"
      cp "$OUT/${m}_answers.npy" "$P/${m}_s${seed}_answers.npy"
    done
  done
fi

have_checkpoints() { local f; for f in "$@"; do [[ -f "$OUT/$f.pt" ]] || return 1; done; }

step "Data validation";        python "$A/validate_data.py" > "$R/validation.txt"
step "Popularity baseline";    python "$A/popularity.py" > /dev/null
step "Collect results";        python "$A/collect_results.py"
step "Main table";             python "$A/summarize.py" > "$R/summary.txt"
step "Ablation figure";        python "$A/plot_ablation.py" > "$R/ablation_summary.txt"
step "Loss vs branch";         python "$A/decomposition.py" > "$R/decomposition.txt"

step "Learned beta"
if have_checkpoints BSARec_LastFM_best $(for s in $SEEDS; do echo BSARec_LastFM_s$s; done); then
  python "$A/beta.py" > "$R/beta_summary.txt"
else echo "skip: BSARec checkpoints (*.pt) not found in src/output (train first)"; fi

step "Oversmoothing"
if have_checkpoints $(for s in $SEEDS; do for m in SASRec BSARecAlpha0 BSARec; do echo ${m}_LastFM_s$s; done; done); then
  python "$A/oversmoothing.py" > "$R/oversmoothing.txt"
else echo "skip: SASRec/BSARec checkpoints (*.pt) not found in src/output (train first)"; fi
for L in $DEPTHS; do
  if have_checkpoints $(for s in $SEEDS; do for m in SASRec BSARecAlpha0 BSARec; do echo ${m}L${L}_LastFM_s$s; done; done); then
    python "$A/oversmoothing.py" --layers "$L" > "$R/oversmoothing_L$L.txt"
  else echo "skip $L layers: depth-sweep checkpoints not found in src/output (train first)"; fi
done
python "$A/oversmoothing_depth.py" > "$R/oversmoothing_depth.txt"   # reads the committed CSVs

step "User groups"
for seed in $SEEDS; do
  if [[ -f "$P/BSARec_s${seed}_preds.npy" ]]; then
    python -W ignore "$A/groups.py" --seed "$seed" > "$R/groups_s$seed.txt"   # -W: known harmless numpy/pandas warnings
  else echo "skip seed $seed: no predictions in analysis/predictions"; fi
done

echo; echo "Done. Check 'git status analysis/' for changes."
