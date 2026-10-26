#!/bin/sh
cd "$(dirname "$0")"
PY=.venv-gpu/Scripts/python.exe
L=experiments/v2
XLSR="--model facebook/wav2vec2-xls-r-300m --tag xlsrft --mode none --lam 0 --noise-aug --save-model"
xlsr() {
  $PY train_wavlm_v2.py $XLSR --seed $1 --freeze-layers 8 --batch-size 8 > $L/gpu2_xlsr_s$1.log 2>&1 || \
  $PY train_wavlm_v2.py $XLSR --seed $1 --freeze-layers 12 --grad-ckpt --batch-size 4 > $L/gpu2_xlsr_s$1_fallback.log 2>&1
}
xlsr 0
for s in 1 2; do
  for h in CREMA-D EMODB RAVDESS SAVEE TESS UA-SER; do
    $PY train_wavlm_v2.py --tag loco --mode none --lam 0 --seed $s --holdout $h > $L/gpu2_loco_${h}_s$s.log 2>&1
  done
done
xlsr 1
xlsr 2
echo QUEUE2_DONE > $L/gpu_queue2_done.txt
