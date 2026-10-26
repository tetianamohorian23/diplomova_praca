# Rozpoznávanie emócií v reči (WavLM) – ako spustiť

Praktická časť diplomovej práce: 7 emócií (angry, disgust, fear, happy, neutral, sad, surprise) na 7 hraných korpusoch.
Hlavný systém: zmrazený WavLM-base-plus → priemer a smerodajná odchýlka po vrstvách → ensemble 5 malých hláv (predvolene so šumovou augmentáciou).
Všetky finálne čísla sú v `experiments/v2/FINAL_RESULTS.md` (v angličtine), grafy v `experiments/v2/figures/`.

## 1. Inštalácia

```powershell
pip install -r requirements.txt
```

Použitý Python 3.14.6; verzia PyTorch pre CPU je uvedená v prvom riadku `requirements.txt`. Pri prvom spustení sa z Hugging Face stiahne
`microsoft/wavlm-base-plus` (treba internet).

## 2. Rýchla kontrola (bez dát, funguje hneď)

```powershell
python evaluate_samples.py
python predict_emotion.py samples/happy_CREMA-D_1078_IWL_HAP_XX.wav --timeline
python gui_emotion.py
```

* `evaluate_samples.py`: spustí model na 28 označených nahrávkach z `samples/` (z testovacej množiny) a vypíše predikciu, pravú emóciu, presnosť a macro-F1
  (očakávajte približne 80 %; je to len kontrola, že celý reťazec funguje, reprezentatívne čísla sú v `FINAL_RESULTS.md`).
* `predict_emotion.py <súbor>`: emócia a pravdepodobnosti jednej nahrávky; `--timeline` po 3-sekundových oknách; `--record 5` nahrá 5 s z mikrofónu;
  `<priečinok> --csv out.csv` spracuje celý priečinok.
* `gui_emotion.py`: malé okno (otvoriť súbor alebo nahrať, stĺpce pravdepodobností).

Používa hlavy z `models/noise_robust/`. Nahrávky v `samples/` pochádzajú z verejných korpusov, sú tu len na vyhodnotenie.

## 3. Reprodukcia experimentov od nuly (v tomto poradí)

Každý krok potrebuje výstupy predchádzajúcich. Všetko sa spúšťa z tohto priečinka. Korpusy v balíku nie sú (kapitola 6).

| # | Príkaz | Výstup / účel |
|---|---|---|
| 1 | `python prepare_combined_dataset.py` | `combined_dataset*.csv`, `audio_16k_mono/` (všetko ako 16 kHz mono) |
| 2 | `python common_splits.py` | `experiments/v2/splits.csv`: jedno rozdelenie train/validation/test pre všetky modely (13 406 nahrávok, test = 3 511) |
| 3 | `python extract_ssl_embeddings.py --model microsoft/wavlm-base-plus --tag wavlm_base_plus` | embeddingy po vrstvách `experiments/v2/emb_wavlm_base_plus.npz` (~50 min na CPU) |
| 4 | `python extract_ssl_embeddings.py --model facebook/wav2vec2-base --tag wav2vec2_base` | to isté pre wav2vec 2.0 (~50 min) |
| 5 | `python train_frozen_heads.py` | hlavy nad zmrazenými embeddingmi (bez pomocnej úlohy / multi-task / adversariálna), 5 seedov |
| 6 | `python train_frozen_heads.py --loco --embeddings wavlm_base_plus --seeds 3 --loco-config none:0,aux:0.1,grl:0.05` | leave-one-corpus-out (kros-korpusové vyhodnotenie) |
| 7 | `python eval_backbones.py` | porovnanie WavLM a wav2vec 2.0 |
| 8 | `python extra_eval.py grouped_cv`, `python extra_eval.py loco_norm`, `python extra_eval.py duplicates`, `python svk_loso.py` | kontroly platnosti: CV podľa hovoriacich, normalizácia po korpusoch, duplicity, slovenské nahrávky |
| 9 | `python train_cnn_ssl_v2.py cache`, potom `pretrain --epochs 20`, `probe`, `finetune --seeds 3` | CNN od nuly vs. samo-supervidované predtrénovanie |
| 10 | `python semi_supervised_wavlm.py`, `python fair_baselines.py`, `python opensmile_baseline.py` | semi-supervidované a klasické modely |
| 11 | `python robustness_noise.py` | odolnosť systému voči šumu |
| 12 | `python export_model.py` a `python export_model.py --norm` | uloží hlavy do `models/` a `models/norm/` |
| 13 | `python extract_noisy_embeddings.py train`, `python extract_noisy_embeddings.py test`, potom `python train_noise_robust.py` | hlavy so šumovou augmentáciou → `models/noise_robust/` |
| 14 | `python analyze_results.py` | `experiments/v2/REPORT.md` |
| 15 | `python final_evaluation.py` | `experiments/v2/FINAL_RESULTS.md` |
| 16 | `python make_figures.py` | `experiments/v2/figures/*.png` |

Kroky 15 a 16 potrebujú `experiments/v2/emb_wavlm_base_plus.npz` z kroku 3 (511 MB, preto nie je v balíku). Malé súbory s výsledkami
(`experiments/v2/frozen/`, `probs.npz`) v balíku sú, takže po kroku 3 sa reporty a grafy dajú znovu vytvoriť bez opätovného trénovania.

### Fine-tuning na GPU (voliteľné)

Potrebné prostredie s CUDA (`.venv-gpu`; behy bežali na RTX 3060, ~23 min na beh). Skripty sú POSIX shell (na Windows cez Git Bash):

```
sh run_gpu_queue.sh
sh run_gpu_queue2.sh
python export_finetuned.py
python evaluate_noise_all.py
```

Jeden beh ručne: `python train_wavlm_v2.py --mode none --lam 0 --seed 0` (so šumovou augmentáciou pridajte `--noise-aug`).

### Ranné experimenty (v1)

`train_models.py`, `train_deep_models.py`, `run_wavlm_ablation.py`, `augment_acoustic_features.py`, `plot_experiments.py`; výsledky v `experiments/*.json`.
Časť z nich vznikla pred zavedením spoločného rozdelenia (`common_splits.py`), preto **všetky finálne čísla pochádzajú z postupu v2** (kroky 1 až 16).

## 4. Kde sú výsledky

| Súbor | Obsah |
|---|---|
| `experiments/v2/FINAL_RESULTS.md` | všetky finálne čísla: test, ablácie, kros-korpus, šum, kalibrácia, fine-tuning |
| `experiments/v2/REPORT.md` | súhrnné tabuľky všetkých experimentov v2 |
| `experiments/v2/figures/`, `experiments/v2/plots/` | grafy |
| `experiments/v2/*.log` | logy behov, z ktorých čísla pochádzajú |

## 5. Kontrolné testy

```powershell
python tests/test_pipeline.py
```

Potrebuje kroky 1 a 2 (korpusy a `audio_16k_mono/`).

## 6. Dáta (korpusy nie sú v balíku)

Každý korpus dajte do vlastného priečinka vedľa skriptov (`.wav` sa hľadajú rekurzívne, `UA-SER` má `dataset.csv` a `clips/`):

| Priečinok | Jazyk | Nahrávok (použitých) | Hovoriacich |
|---|---|---|---|
| `CREMA-D/` | angličtina | 7 442 | 91 |
| `RAVDESS/` | angličtina | 1 248 | 24 |
| `TESS/` | angličtina | 2 800 | 2 |
| `SAVEE/` | angličtina | 480 | 4 |
| `EMODB/` | nemčina | 454 | 10 |
| `UA-SER/` | ukrajinčina | 952 | 400 |
| `SVK-DB/` | slovenčina | 30 | 4 |

Triedy `calm` a `boredom` sa nepoužívajú. Počty sú z `experiments/v2/splits.csv`; niektoré archívy majú duplicitné podpriečinky (napr. `TESS/` má 5 600 súborov
pri 2 800 unikátnych nahrávkach), preto sa do experimentov berie vždy len `splits.csv`.

## 7. Dôležité poznámky

* TESS má iba 2 hovoriace, preto nie je v tréningu a tvorí približne 40 % validačnej a testovacej množiny; `FINAL_RESULTS.md` uvádza aj výsledky bez TESS.
* Kros-korpusové macro-F1 je len okolo 0,3 (hraná reč, nové nahrávacie podmienky); je to uvedené ako obmedzenie práce.
* Natrénované na hranej reči; na spontánnej reči alebo v iných podmienkach je presnosť výrazne nižšia.
* Prostredie výsledkov: Windows 11, Intel Core i9-12900H, 31,7 GB RAM, bez GPU (Python 3.14.6, PyTorch 2.14.1 CPU, transformers 5.18.0); fine-tuning na RTX 3060.
