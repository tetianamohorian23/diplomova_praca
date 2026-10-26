# Final results

Test split: 3511 clips, speakers disjoint from train/validation. Macro metrics over 7 emotions.

## 1. All models (test)
| model                                       |   acc |   bal_acc |   macro_P |   macro_R |   macro_F1 |   angry |   disgust |   fear |   happy |   neutral |   sad |   surprise |
|:--------------------------------------------|------:|----------:|----------:|----------:|-----------:|--------:|----------:|-------:|--------:|----------:|------:|-----------:|
| WavLM frozen + head, ensemble of 5 seeds    | 0.709 |     0.710 |     0.744 |     0.710 |      0.715 |   0.840 |     0.639 |  0.716 |   0.741 |     0.681 | 0.647 |      0.739 |
| WavLM frozen + head, seed 0                 | 0.684 |     0.691 |     0.720 |     0.691 |      0.690 |   0.865 |     0.647 |  0.609 |   0.730 |     0.637 | 0.572 |      0.767 |
| wavlm_multitask (old run, 1 seed)           | 0.704 |     0.660 |     0.732 |     0.660 |      0.655 |   0.904 |     0.726 |  0.676 |   0.616 |     0.715 | 0.723 |      0.227 |
| wav2vec2 (old run, 1 seed)                  | 0.684 |     0.646 |     0.699 |     0.646 |      0.645 |   0.789 |     0.615 |  0.743 |   0.615 |     0.749 | 0.742 |      0.262 |
| Wav2Vec2 frozen + head, ensemble of 5 seeds | 0.629 |     0.605 |     0.658 |     0.605 |      0.608 |   0.833 |     0.617 |  0.547 |   0.665 |     0.653 | 0.546 |      0.393 |
| Wav2Vec2 frozen + head, seed 0              | 0.599 |     0.577 |     0.644 |     0.577 |      0.581 |   0.740 |     0.579 |  0.521 |   0.681 |     0.642 | 0.505 |      0.400 |
| openSMILE eGeMAPS + logreg (C=10)           | 0.463 |   nan     |   nan     |   nan     |      0.451 |   0.650 |     0.383 |  0.367 |   0.337 |     0.551 | 0.502 |      0.369 |
| logistic_regression (old run, 1 seed)       | 0.476 |     0.456 |     0.471 |     0.456 |      0.448 |   0.453 |     0.613 |  0.329 |   0.301 |     0.586 | 0.592 |      0.264 |
| cnn_bilstm (old run, 1 seed)                | 0.476 |     0.445 |     0.478 |     0.445 |      0.443 |   0.492 |     0.478 |  0.372 |   0.393 |     0.665 | 0.568 |      0.133 |
| openSMILE eGeMAPS + svm_rbf (C=10)          | 0.482 |   nan     |   nan     |   nan     |      0.437 |   0.605 |     0.446 |  0.395 |   0.490 |     0.431 | 0.555 |      0.138 |
| openSMILE eGeMAPS + random_forest (n=600)   | 0.435 |   nan     |   nan     |   nan     |      0.404 |   0.514 |     0.500 |  0.263 |   0.423 |     0.458 | 0.498 |      0.168 |
| semi_supervised_svm (old run, 1 seed)       | 0.408 |     0.380 |     0.387 |     0.380 |      0.371 |   0.524 |     0.480 |  0.332 |   0.318 |     0.392 | 0.452 |      0.101 |
| cnn_mel (old run, 1 seed)                   | 0.312 |     0.299 |     0.353 |     0.299 |      0.297 |   0.450 |     0.212 |  0.075 |   0.227 |     0.457 | 0.392 |      0.268 |

Best model (WavLM frozen, 5-seed head ensemble): macro-F1 0.715, 95% bootstrap CI [0.700, 0.728].

## 2. Configuration selected on VALIDATION only (no test peeking)
Earlier tables ranked configurations by test F1. Here the configuration is chosen by mean validation macro-F1 over seeds and only then its test score is reported.

| embedding       | config     |   val_f1 |   test_f1 (mean of 5 seeds) |   test_std |
|:----------------|:-----------|---------:|----------------------------:|-----------:|
| wav2vec2_base   | aux λ=0.2  |    0.557 |                       0.607 |      0.023 |
| wavlm_base_plus | aux λ=0.05 |    0.548 |                       0.661 |      0.028 |
| wavlm_large     | none λ=0   |    0.595 |                       0.688 |      0.021 |
| xlsr_300m       | none λ=0   |    0.631 |                       0.701 |      0.021 |

All configurations (mean over 5 seeds):

| embedding       | config     |   val_f1 |   test_f1 |   test_std |
|:----------------|:-----------|---------:|----------:|-----------:|
| wav2vec2_base   | aux λ=0.05 |    0.534 |     0.611 |      0.042 |
| wav2vec2_base   | aux λ=0.1  |    0.551 |     0.614 |      0.034 |
| wav2vec2_base   | aux λ=0.2  |    0.557 |     0.607 |      0.023 |
| wav2vec2_base   | aux λ=0.5  |    0.544 |     0.606 |      0.025 |
| wav2vec2_base   | grl λ=0.05 |    0.515 |     0.617 |      0.037 |
| wav2vec2_base   | grl λ=0.1  |    0.491 |     0.588 |      0.050 |
| wav2vec2_base   | grl λ=0.2  |    0.476 |     0.596 |      0.027 |
| wav2vec2_base   | grl λ=0.5  |    0.448 |     0.542 |      0.042 |
| wav2vec2_base   | none λ=0   |    0.523 |     0.597 |      0.021 |
| wavlm_base_plus | aux λ=0.05 |    0.548 |     0.661 |      0.028 |
| wavlm_base_plus | aux λ=0.1  |    0.540 |     0.648 |      0.037 |
| wavlm_base_plus | aux λ=0.2  |    0.540 |     0.647 |      0.017 |
| wavlm_base_plus | aux λ=0.5  |    0.541 |     0.660 |      0.019 |
| wavlm_base_plus | grl λ=0.05 |    0.518 |     0.661 |      0.035 |
| wavlm_base_plus | grl λ=0.1  |    0.527 |     0.649 |      0.075 |
| wavlm_base_plus | grl λ=0.2  |    0.483 |     0.562 |      0.048 |
| wavlm_base_plus | grl λ=0.5  |    0.466 |     0.529 |      0.037 |
| wavlm_base_plus | none λ=0   |    0.544 |     0.676 |      0.034 |
| wavlm_large     | none λ=0   |    0.595 |     0.688 |      0.021 |
| xlsr_300m       | none λ=0   |    0.631 |     0.701 |      0.021 |

Validation-selected WavLM configuration (`aux λ=0.05`), ensemble of its 5 seeds: macro-F1 0.685. The differences between 'none' and the multi-task variants (about 0.01-0.02) are smaller than the seed-to-seed std (0.02-0.04), so they are not significant; the deployed model uses 'none' (simplest, validation F1 within 0.004 of the best).

## 3. Robustness of the headline number to the TESS imbalance
| view                                             |   macro_F1 |    n |
|:-------------------------------------------------|-----------:|-----:|
| full test (40% TESS)                             |      0.715 | 3511 |
| test without TESS                                |      0.757 | 2111 |
| mean of per-corpus macro-F1 (corpora with n>=20) |      0.742 |    6 |

Per corpus (corpora with at least 20 test clips; SVK-DB has 2 and is evaluated separately below):

| corpus   | language   |   n_test |   macro_F1 |   accuracy |
|:---------|:-----------|---------:|-----------:|-----------:|
| CREMA-D  | en         |     1470 |      0.788 |      0.784 |
| EMODB    | de         |       78 |      0.880 |      0.872 |
| RAVDESS  | en         |      260 |      0.767 |      0.754 |
| SAVEE    | en         |      120 |      0.703 |      0.725 |
| TESS     | en         |     1400 |      0.611 |      0.614 |
| UA-SER   | uk         |      181 |      0.705 |      0.702 |

Note: in the default split TESS (2 speakers) is absent from training, so ~40% of the test set is effectively cross-speaker AND cross-corpus; in speaker-grouped CV (below) TESS reaches F1 0.80 instead of 0.61. Headline numbers of the default split are therefore conservative for TESS.

### Speaker-grouped 5-fold cross-validation over all 13,406 clips
Pooled macro-F1 0.733, accuracy 0.749; per-fold macro-F1 0.743 ± 0.020; without TESS 0.737.

|   fold |        n |   macro_f1 |   accuracy |
|-------:|---------:|-----------:|-----------:|
|  0.000 | 2682.000 |      0.776 |      0.808 |
|  1.000 | 2681.000 |      0.739 |      0.751 |
|  2.000 | 2681.000 |      0.741 |      0.733 |
|  3.000 | 2681.000 |      0.714 |      0.720 |
|  4.000 | 2681.000 |      0.746 |      0.733 |

Per-corpus F1 in CV:

| corpus   |   macro_F1 |
|:---------|-----------:|
| CREMA-D  |      0.738 |
| EMODB    |      0.810 |
| RAVDESS  |      0.756 |
| SAVEE    |      0.645 |
| SVK-DB   |      0.199 |
| TESS     |      0.796 |
| UA-SER   |      0.696 |

## 4. Cross-corpus (leave-one-corpus-out), frozen WavLM, mean of 3 seeds
Train on the other corpora, test on the unseen one. Chance level for 7 classes is about 0.14-0.17.

| config                                      |   CREMA-D |   EMODB |   RAVDESS |   SAVEE |   TESS |   UA-SER |   average |
|:--------------------------------------------|----------:|--------:|----------:|--------:|-------:|---------:|----------:|
| aux λ=0.1                                   |     0.274 |   0.346 |     0.157 |   0.329 |  0.322 |    0.240 |     0.278 |
| grl λ=0.05                                  |     0.295 |   0.319 |     0.246 |   0.311 |  0.318 |    0.264 |     0.292 |
| none λ=0                                    |     0.295 |   0.346 |     0.156 |   0.362 |  0.401 |    0.300 |     0.310 |
| none λ=0 + per-corpus feature normalisation |     0.400 |   0.438 |     0.531 |   0.515 |  0.677 |    0.334 |     0.483 |

### Slovak (SVK-DB, 30 clips, 4 speakers)
| setting   |   n_clips |   macro_F1 | CI95         |   accuracy |
|:----------|----------:|-----------:|:-------------|-----------:|
| zero_shot |        30 |      0.231 | [0.09, 0.35] |      0.267 |
| loso      |        30 |      0.211 | [0.09, 0.30] |      0.267 |

## 5. Calibration and 'uncertain' option (best model)
ECE = 0.155, Brier score = 0.438. The ensemble is under-confident on average (mean confidence 0.55 vs accuracy 0.71).

|   min_confidence |   coverage |   accuracy_on_answered |   macro_F1_on_answered |
|-----------------:|-----------:|-----------------------:|-----------------------:|
|            0.000 |      1.000 |                  0.709 |                  0.715 |
|            0.400 |      0.690 |                  0.808 |                  0.805 |
|            0.500 |      0.508 |                  0.874 |                  0.863 |
|            0.600 |      0.380 |                  0.928 |                  0.912 |
|            0.700 |      0.294 |                  0.954 |                  0.942 |
|            0.800 |      0.191 |                  0.975 |                  0.964 |

## 6. Per-emotion precision / recall / F1 (best model)
| emotion   |   precision |   recall |    f1 |   support |
|:----------|------------:|---------:|------:|----------:|
| angry     |       0.887 |    0.798 | 0.840 |       569 |
| disgust   |       0.503 |    0.876 | 0.639 |       508 |
| fear      |       0.835 |    0.626 | 0.716 |       519 |
| happy     |       0.800 |    0.690 | 0.741 |       574 |
| neutral   |       0.795 |    0.595 | 0.681 |       521 |
| sad       |       0.629 |    0.665 | 0.647 |       565 |
| surprise  |       0.757 |    0.722 | 0.739 |       255 |

## 7. Robustness to additive white noise (stratified test subset, 1000 clips)
| condition   |   macro_F1 |   accuracy |
|:------------|-----------:|-----------:|
| clean       |      0.717 |      0.709 |
| snr_20dB    |      0.499 |      0.495 |
| snr_10dB    |      0.267 |      0.291 |
| snr_5dB     |      0.104 |      0.182 |
| snr_0dB     |      0.038 |      0.144 |

## 7b. Noise-augmented heads (deployed default)
Heads trained on clean train clips plus noisy copies of the TRAIN clips only (white, pink or babble noise, SNR uniform 0-30 dB); noisy copies of validation clips stay in validation and only select the epoch, TESS is never used for training. Clean test: single seeds 0.732, 0.762, 0.732, 0.746, 0.708 (mean 0.736); ensemble of 5 seeds macro-F1 0.769 (95% CI [0.755, 0.782]), accuracy 0.766, without TESS 0.735. Both head sets are compared on the same noisy features of the 1050-clip stratified test subset (noise types seen in training; street noise, music and reverberation were not tested). NOTE: this table uses pre-extracted noisy features; to compare systems use section 7g (noise regenerated identically for all four systems) and do not mix the two:

| condition   |   standard_F1 |   noise_robust_F1 |   gain |
|:------------|--------------:|------------------:|-------:|
| clean       |         0.717 |             0.774 |  0.057 |
| white_20    |         0.506 |             0.674 |  0.168 |
| white_10    |         0.248 |             0.556 |  0.308 |
| white_5     |         0.103 |             0.467 |  0.364 |
| white_0     |         0.036 |             0.384 |  0.348 |
| pink_10     |         0.339 |             0.557 |  0.218 |
| babble_20   |         0.544 |             0.703 |  0.158 |
| babble_10   |         0.291 |             0.529 |  0.238 |
| babble_5    |         0.185 |             0.434 |  0.250 |

Per emotion (robust ensemble, clean test):

| emotion   |   precision |   recall |    f1 |   support |
|:----------|------------:|---------:|------:|----------:|
| angry     |       0.883 |    0.889 | 0.886 |       569 |
| disgust   |       0.606 |    0.858 | 0.710 |       508 |
| fear      |       0.833 |    0.701 | 0.762 |       519 |
| happy     |       0.819 |    0.718 | 0.765 |       574 |
| neutral   |       0.813 |    0.720 | 0.764 |       521 |
| sad       |       0.721 |    0.662 | 0.690 |       565 |
| surprise  |       0.745 |    0.871 | 0.803 |       255 |

### 7b2. Calibration and 'uncertain' option for the DEPLOYED model (noise-augmented ensemble; section 5 above is the standard ensemble)
ECE = 0.209, Brier score = 0.400, mean confidence 0.56 vs accuracy 0.77.

|   min_confidence |   coverage |   accuracy_on_answered |   macro_F1_on_answered |
|-----------------:|-----------:|-----------------------:|-----------------------:|
|            0.000 |      1.000 |                  0.766 |                  0.769 |
|            0.400 |      0.729 |                  0.847 |                  0.841 |
|            0.500 |      0.535 |                  0.904 |                  0.893 |
|            0.600 |      0.395 |                  0.937 |                  0.922 |
|            0.700 |      0.277 |                  0.958 |                  0.943 |
|            0.800 |      0.159 |                  0.978 |                  0.971 |

## 7c. Fine-tuning WavLM-base-plus on the GPU (RTX 3060 Laptop, bf16)
Waveform augmentation (noise, gain, speed), weighted layer sum, layer-wise lr decay, warm-up + cosine, label smoothing, SpecAugment-style masking inside the model; epoch selected on validation macro-F1.

| run            |   best_epoch |   val_F1 |   test_F1 |   test_acc |   F1_without_TESS |   minutes |
|:---------------|-------------:|---------:|----------:|-----------:|------------------:|----------:|
| aux_0.1__seed0 |            7 |    0.568 |     0.703 |      0.707 |             0.773 |    23.448 |
| none_0__seed0  |            5 |    0.564 |     0.763 |      0.758 |             0.782 |    23.597 |
| none_0__seed1  |            5 |    0.572 |     0.704 |      0.714 |             0.760 |    23.431 |
| none_0__seed2  |            6 |    0.569 |     0.635 |      0.677 |             0.760 |    23.449 |

Mean over the 3 'none' runs: 0.701 ± 0.064 (large seed-to-seed spread).

Fine-tuned vs frozen (test, 3511 clips):

| model                                                    |   macro_F1 | CI95           |   accuracy |   F1_without_TESS |
|:---------------------------------------------------------|-----------:|:---------------|-----------:|------------------:|
| fine-tuned, ensemble of the 'none' runs                  |      0.734 | [0.719, 0.749] |      0.738 |             0.786 |
| fine-tuned, ensemble of all runs                         |      0.726 | [0.711, 0.741] |      0.731 |             0.781 |
| frozen WavLM + head, ensemble of 5 seeds                 |      0.715 | [0.700, 0.728] |      0.709 |             0.757 |
| frozen WavLM + noise-augmented head, ensemble of 5 seeds |      0.769 | [0.755, 0.782] |      0.766 |             0.735 |

Paired bootstrap, fine-tuned ('none' ensemble) minus frozen ensemble: delta macro-F1 = +0.020 (95% CI +0.004 .. +0.036).

## 7d. Fine-tuning with noise augmentation (white / pink / babble, 0-30 dB) and a clean control run
| run                           |   best_epoch |   val_F1 |   test_F1 |   test_acc |   F1_without_TESS |   minutes |
|:------------------------------|-------------:|---------:|----------:|-----------:|------------------:|----------:|
| ftnoise__none_0__seed0__noise |            3 |    0.534 |     0.654 |      0.645 |             0.716 |    21.308 |
| ftnoise__none_0__seed1__noise |            5 |    0.595 |     0.734 |      0.735 |             0.759 |    25.443 |
| ftnoise__none_0__seed2__noise |            4 |    0.595 |     0.684 |      0.704 |             0.737 |    20.800 |
| ftclean__none_0__seed0        |            5 |    0.564 |     0.763 |      0.758 |             0.782 |    23.368 |

Ensemble of the noise-augmented fine-tuned runs: macro-F1 0.733 (95% CI [0.718, 0.747]), accuracy 0.736, without TESS 0.765.

## 7d2. Fine-tuning the multilingual XLS-R-300M (lowest 8 layers frozen, noise augmentation, bf16, 6 GB GPU)
| run                          |   best_epoch |   val_F1 |   test_F1 |   test_acc |   F1_without_TESS |   minutes |
|:-----------------------------|-------------:|---------:|----------:|-----------:|------------------:|----------:|
| xlsrft__none_0__seed0__noise |            4 |    0.555 |     0.618 |      0.637 |             0.537 |    66.909 |
| xlsrft__none_0__seed1__noise |            4 |    0.556 |     0.700 |      0.707 |             0.633 |   112.226 |
| xlsrft__none_0__seed2__noise |            8 |    0.525 |     0.698 |      0.706 |             0.594 |    99.137 |

Ensemble of 3 runs: macro-F1 0.689 (95% CI [0.675, 0.703]), without TESS 0.605. Seed spread is large (first run 0.62, the others 0.70). XLS-R generalises well to the unseen TESS speakers but is clearly weaker than fine-tuned WavLM on the seen corpora (without-TESS F1 about 0.60 vs 0.77), and the frozen XLS-R features with a light head (7e) are as good as the fine-tuned model.

## 7e. Backbone comparison (frozen features, same plain head)
Default split: mean ± std over 5 seeds. Cross-corpus: mean over the 6 held-out corpora (3 seeds each), std across corpora.

| backbone                  | default F1    | cross-corpus F1   |   CREMA-D |   EMODB |   RAVDESS |   SAVEE |   TESS |   UA-SER |
|:--------------------------|:--------------|:------------------|----------:|--------:|----------:|--------:|-------:|---------:|
| Wav2Vec2-base             | 0.597 ± 0.021 | 0.279 ± 0.052     |     0.262 |   0.334 |     0.222 |   0.351 |  0.235 |    0.271 |
| WavLM-base-plus           | 0.676 ± 0.034 | 0.310 ± 0.085     |     0.295 |   0.346 |     0.156 |   0.362 |  0.401 |    0.300 |
| WavLM-large               | 0.688 ± 0.021 | 0.372 ± 0.055     |     0.388 |   0.377 |     0.311 |   0.447 |  0.407 |    0.305 |
| XLS-R-300M (multilingual) | 0.701 ± 0.021 | 0.403 ± 0.031     |     0.385 |   0.452 |     0.402 |   0.410 |  0.413 |    0.358 |

## 7f. Cross-corpus: fine-tuned model vs frozen features
Leave-one-corpus-out, macro-F1 on the unseen corpus. The fine-tuned model uses clean training and is averaged over the available seeds (seed spread of fine-tuning is large, see 7c); per-corpus normalisation needs a batch of recordings from the target source.

| held-out corpus   |   fine-tuned WavLM (3 seed(s), mean) |   frozen WavLM, plain head |   frozen + per-corpus normalisation |
|:------------------|-------------------------------------:|---------------------------:|------------------------------------:|
| CREMA-D           |                                0.391 |                      0.295 |                               0.400 |
| EMODB             |                                0.388 |                      0.346 |                               0.438 |
| RAVDESS           |                                0.545 |                      0.156 |                               0.531 |
| SAVEE             |                                0.509 |                      0.362 |                               0.515 |
| TESS              |                                0.431 |                      0.401 |                               0.677 |
| UA-SER            |                                0.318 |                      0.300 |                               0.334 |
| average           |                                0.430 |                      0.310 |                               0.483 |

## 7g. All systems on identical corrupted audio (1,050-clip test subset, macro-F1)
Conditions marked 'NO' were never used in training of any system. Brown noise has >99% of its energy below 100 Hz (outside the speech band).

| condition   | seen in training   |   frozen, clean training |   frozen, noise-augmented |   fine-tuned, clean training |   fine-tuned, noise-augmented |
|:------------|:-------------------|-------------------------:|--------------------------:|-----------------------------:|------------------------------:|
| clean       | yes                |                    0.717 |                     0.774 |                        0.750 |                         0.713 |
| white_20    | yes                |                    0.499 |                     0.669 |                        0.652 |                         0.685 |
| white_10    | yes                |                    0.257 |                     0.540 |                        0.609 |                         0.661 |
| white_5     | yes                |                    0.101 |                     0.495 |                        0.543 |                         0.647 |
| white_0     | yes                |                    0.036 |                     0.394 |                        0.397 |                         0.590 |
| pink_10     | yes                |                    0.331 |                     0.563 |                        0.620 |                         0.670 |
| babble_20   | yes                |                    0.562 |                     0.695 |                        0.589 |                         0.697 |
| babble_10   | yes                |                    0.273 |                     0.525 |                        0.367 |                         0.637 |
| babble_5    | yes                |                    0.171 |                     0.439 |                        0.254 |                         0.553 |
| reverb_0.3  | NO                 |                    0.515 |                     0.516 |                        0.447 |                         0.478 |
| reverb_0.8  | NO                 |                    0.266 |                     0.302 |                        0.353 |                         0.395 |
| telephone   | NO                 |                    0.424 |                     0.529 |                        0.551 |                         0.537 |
| lowpass     | NO                 |                    0.610 |                     0.733 |                        0.697 |                         0.671 |
| brown_10    | NO                 |                    0.674 |                     0.745 |                        0.726 |                         0.700 |
| hum_10      | NO                 |                    0.623 |                     0.675 |                        0.705 |                         0.669 |
| clip        | NO                 |                    0.705 |                     0.768 |                        0.750 |                         0.712 |

## 8. Self-supervised pre-training of the CNN encoder (same architecture, only initialisation differs)
|   fraction | init    |   mean |   std |   count |
|-----------:|:--------|-------:|------:|--------:|
|      0.100 | scratch |  0.096 | 0.011 |       2 |
|      0.100 | ssl     |  0.208 | 0.031 |       2 |
|      1.000 | scratch |  0.340 | 0.031 |       3 |
|      1.000 | ssl     |  0.411 | 0.007 |       2 |

Linear probe on the frozen encoder: scratch 0.384, contrastive-pretrained 0.428.

### Semi-supervised (self-training) on WavLM embeddings vs supervised, same labelled subsets
|   labeled_fraction |   supervised mean |   supervised std |   self_training mean |   self_training std |
|-------------------:|------------------:|-----------------:|---------------------:|--------------------:|
|              0.050 |             0.430 |            0.017 |                0.437 |               0.017 |
|              0.100 |             0.504 |            0.020 |                0.498 |               0.013 |
|              0.200 |             0.543 |            0.025 |                0.551 |               0.020 |
|              0.500 |             0.572 |            0.026 |                0.573 |               0.028 |

### Same comparison on acoustic features (LinearSVC)
|   labeled_fraction |   supervised_f1 mean |   supervised_f1 std |   self_training_f1 mean |   self_training_f1 std |
|-------------------:|---------------------:|--------------------:|------------------------:|-----------------------:|
|              0.200 |                0.318 |               0.012 |                   0.324 |                  0.020 |
|              0.500 |                0.373 |               0.016 |                   0.364 |                  0.021 |
|              1.000 |                0.383 |               0.000 |                   0.383 |                  0.000 |

## 9. Data-leakage checks
```
{
 "exact_duplicate_rows": 10,
 "exact_duplicate_groups_crossing_splits": 0,
 "near_duplicates_test_vs_train(cos>0.995)": 0,
 "near_dup_same_label_rate": null,
 "max_cosine_test_to_train_percentiles": {
  "50": 0.6697821021080017,
  "90": 0.8438881635665894,
  "99": 0.9420391917228699,
  "100": 0.9811538457870483
 }
}
```

## 10. Representation analysis
`plots/tsne.png` (checked visually): WavLM embeddings form clearly separated islands per corpus (CREMA-D, RAVDESS, SAVEE, EMO-DB, UA-SER, TESS), while emotions are mixed inside each island except in the small, very acted corpora; the model must learn emotion across strong corpus offsets, which explains the cross-corpus drop. `plots/layer_weights.png`: learned weights over WavLM layers.
