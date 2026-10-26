#!/bin/sh
cd "$(dirname "$0")"
PY=.venv-gpu/Scripts/python.exe
L=experiments/v2
$PY extract_ssl_embeddings.py --model microsoft/wavlm-large --tag wavlm_large > $L/gpu_extract_wavlm_large.log 2>&1
$PY extract_ssl_embeddings.py --model facebook/wav2vec2-xls-r-300m --tag xlsr_300m > $L/gpu_extract_xlsr.log 2>&1
for s in 0 1 2; do
  $PY train_wavlm_v2.py --tag ftnoise --mode none --lam 0 --seed $s --noise-aug --save-model > $L/gpu_ftnoise_s$s.log 2>&1
done
$PY train_wavlm_v2.py --tag ftclean --mode none --lam 0 --seed 0 --save-model > $L/gpu_ftclean_s0.log 2>&1
for h in CREMA-D EMODB RAVDESS SAVEE TESS UA-SER; do
  $PY train_wavlm_v2.py --tag loco --mode none --lam 0 --seed 0 --holdout $h > $L/gpu_loco_$h.log 2>&1
done
echo QUEUE_DONE > $L/gpu_queue_done.txt
