# Experiment report (v2)
Test set: 3511 clips, speaker-disjoint; same split for every model.

## 1. Frozen SSL features + light head: corpus-task ablation (mean ± std over seeds)
| embedding       | mode   |   lambda |   n |   val_f1 | test_f1       |   acc |   bal_acc |
|:----------------|:-------|---------:|----:|---------:|:--------------|------:|----------:|
| wav2vec2_base   | aux    |    0.050 |   5 |    0.534 | 0.611 ± 0.042 | 0.629 |     0.611 |
| wav2vec2_base   | aux    |    0.100 |   5 |    0.551 | 0.614 ± 0.034 | 0.634 |     0.614 |
| wav2vec2_base   | aux    |    0.200 |   5 |    0.557 | 0.607 ± 0.023 | 0.627 |     0.607 |
| wav2vec2_base   | aux    |    0.500 |   5 |    0.544 | 0.606 ± 0.025 | 0.624 |     0.605 |
| wav2vec2_base   | grl    |    0.050 |   5 |    0.515 | 0.617 ± 0.037 | 0.626 |     0.618 |
| wav2vec2_base   | grl    |    0.100 |   5 |    0.491 | 0.588 ± 0.050 | 0.605 |     0.592 |
| wav2vec2_base   | grl    |    0.200 |   5 |    0.476 | 0.596 ± 0.027 | 0.613 |     0.602 |
| wav2vec2_base   | grl    |    0.500 |   5 |    0.448 | 0.542 ± 0.042 | 0.558 |     0.561 |
| wav2vec2_base   | none   |    0.000 |   5 |    0.523 | 0.597 ± 0.021 | 0.616 |     0.596 |
| wavlm_base_plus | aux    |    0.050 |   5 |    0.548 | 0.661 ± 0.028 | 0.666 |     0.652 |
| wavlm_base_plus | aux    |    0.100 |   5 |    0.540 | 0.648 ± 0.037 | 0.650 |     0.637 |
| wavlm_base_plus | aux    |    0.200 |   5 |    0.540 | 0.647 ± 0.017 | 0.650 |     0.637 |
| wavlm_base_plus | aux    |    0.500 |   5 |    0.541 | 0.660 ± 0.019 | 0.661 |     0.651 |
| wavlm_base_plus | grl    |    0.050 |   5 |    0.518 | 0.661 ± 0.035 | 0.667 |     0.658 |
| wavlm_base_plus | grl    |    0.100 |   5 |    0.527 | 0.649 ± 0.075 | 0.662 |     0.651 |
| wavlm_base_plus | grl    |    0.200 |   5 |    0.483 | 0.562 ± 0.048 | 0.573 |     0.560 |
| wavlm_base_plus | grl    |    0.500 |   5 |    0.466 | 0.529 ± 0.037 | 0.549 |     0.537 |
| wavlm_base_plus | none   |    0.000 |   5 |    0.544 | 0.676 ± 0.034 | 0.675 |     0.676 |

## 2. Leave-one-corpus-out (train on the other corpora, test on the held-out one; macro-F1)
|                                  |   CREMA-D |   EMODB |   RAVDESS |   SAVEE |   TESS |   UA-SER |   average |
|:---------------------------------|----------:|--------:|----------:|--------:|-------:|---------:|----------:|
| ('wavlm_base_plus', 'aux', 0.1)  |     0.274 |   0.346 |     0.157 |   0.329 |  0.322 |    0.240 |     0.278 |
| ('wavlm_base_plus', 'grl', 0.05) |     0.295 |   0.319 |     0.246 |   0.311 |  0.318 |    0.264 |     0.292 |
| ('wavlm_base_plus', 'none', 0.0) |     0.295 |   0.346 |     0.156 |   0.362 |  0.401 |    0.300 |     0.310 |

## 4. Single models vs ensembles (test macro-F1 with 95% bootstrap CI)
| model                                   |   macro_f1 | ci95           |   acc |
|:----------------------------------------|-----------:|:---------------|------:|
| wavlm_base_plus_frozen(mean of 5 seeds) |      0.715 | [0.700, 0.728] | 0.709 |
| wav2vec2_base_frozen(mean of 5 seeds)   |      0.608 | [0.589, 0.624] | 0.629 |
| ENSEMBLE(all members, prob. average)    |      0.690 | [0.673, 0.706] | 0.694 |
- wavlm_base_plus_frozen(mean of 5 seeds) − wav2vec2_base_frozen(mean of 5 seeds): ΔF1 = +0.107 (95% CI +0.091…+0.123)
- wavlm_base_plus_frozen(mean of 5 seeds) − ENSEMBLE(all members, prob. average): ΔF1 = +0.025 (95% CI +0.013…+0.037)

### Per-corpus results of `wavlm_base_plus_frozen(mean of 5 seeds)`
| corpus   | language   |    n |   macro_f1 |   acc |
|:---------|:-----------|-----:|-----------:|------:|
| CREMA-D  | en         | 1470 |      0.788 | 0.784 |
| EMODB    | de         |   78 |      0.880 | 0.872 |
| RAVDESS  | en         |  260 |      0.767 | 0.754 |
| SAVEE    | en         |  120 |      0.703 | 0.725 |
| SVK-DB   | sk         |    2 |      0.000 | 0.000 |
| TESS     | en         | 1400 |      0.611 | 0.614 |
| UA-SER   | uk         |  181 |      0.705 | 0.702 |

### Most frequent confusions (true → predicted)
- neutral → sad: 109
- sad → disgust: 105
- happy → disgust: 98
- neutral → disgust: 85
- fear → sad: 69
- angry → disgust: 67

`surprise` exists only in RAVDESS, SAVEE and TESS; test recall by corpus: RAVDESS: 0.82 (n=40), SAVEE: 1.00 (n=15), TESS: 0.68 (n=200)

## 5. Layer-wise linear probing (frozen, mean pooled)
|   layer |   wav2vec2_base |   wavlm_base_plus |
|--------:|----------------:|------------------:|
|       0 |           0.455 |             0.452 |
|       1 |           0.500 |             0.534 |
|       2 |           0.493 |             0.558 |
|       3 |           0.571 |             0.557 |
|       4 |           0.550 |             0.577 |
|       5 |           0.567 |             0.555 |
|       6 |           0.559 |             0.562 |
|       7 |           0.517 |             0.535 |
|       8 |           0.541 |             0.551 |
|       9 |           0.601 |             0.481 |
|      10 |           0.556 |             0.540 |
|      11 |           0.518 |             0.556 |
|      12 |           0.521 |             0.529 |

## Data caveats
- TESS has only 2 speakers, so the speaker-disjoint split puts none in train and 1400 clips each in validation and test (≈44% of val and 40% of test). Validation-based model selection and test scores are therefore TESS-heavy.
- CREMA-D has no `surprise`; SVK-DB has 30 clips, EMO-DB/UA-SER lack several classes.
