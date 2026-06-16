# Audio Compression Module

This module is responsible for reducing the length of the audio embedding sequence before it is passed into the GPT-2 fusion model.

In the full LAM4SER pipeline, it sits between the audio encoder and the language-model fusion block:

```text
Audio encoder → Audio compression → Audio/GPT-2 fusion → Emotion classifier
```

## Why compression is needed

The audio encoder produces a sequence of frame-level embeddings. For example, a short utterance can produce many audio tokens, where each token is a high-dimensional vector. Passing all of these audio tokens directly into the fusion model would make training slower and more memory-heavy.

The goal of this module is therefore simple:

> keep the most useful emotional information while reducing the number of audio tokens.

In our experiments, compressed audio representations are used as the audio input to the GPT-2 based fusion model.

## Implemented compression strategies

The project supports several audio compression strategies:

| Compressor   | Type          | Idea                                                     |
| ------------ | ------------- | -------------------------------------------------------- |
| `mean`       | non-learnable | averages the audio sequence over time                    |
| `max`        | non-learnable | keeps the strongest activation values over time          |
| `attention`  | learnable     | learns attention-based compression of the audio sequence |
| `gated`      | learnable     | learns a gating mechanism over the audio representation  |
| `multiscale` | learnable     | combines information from multiple temporal scales       |

The compressor is created using:

```python
from models.compression.compressor import build_compressor

compressor = build_compressor(
    "mean",
    target_len=50,
    hidden_dim=1024,
)
```

The expected tensor format is:

```text
Input:  [batch_size, audio_length, audio_dim]
Output: [batch_size, compressed_length, audio_dim]
```

For example:

```text
Input:  [16, T, 1024]
Output: [16, 50, 1024]
```

depending on the selected compressor.

## AIBO speech emotion recognition experiments

The compression module was updated and tested as part of the AIBO speech emotion recognition experiments.

The main changes were:

* Added support for comparing several compressors under the same training setup.
* Fixed training so that compressor parameters are included in the optimizer.
* Saved and restored the best compressor state together with the model checkpoint.
* Selected the best checkpoint using validation weighted F1 instead of validation loss.
* Evaluated compressors on a held-out AIBO cross-domain split.
* Added weighted F1 and macro F1 reporting, since AIBO is highly imbalanced.

This was important because earlier runs only trained the GPT-2 fusion parameters correctly, while the compressor parameters were not always handled as part of the full trainable system.

## AIBO compressor comparison

The compressor comparison was run on the AIBO dataset using a cross-domain setup:

```text
Train/validation: Mont
Test:             Ohm
```

The same hyperparameters were used for all compressors:

```text
Encoder:          wav2vec2-large-emotion
Prompt:           base
Target audio len: 50
Batch size:       16
Learning rate:    5e-6
LoRA rank:        8
LoRA LR:          5e-5
Dropout:          0.4
Adapter dim:      32
Epochs:           30
Early stopping:   patience 8
```

The final results were:

| Rank | Compressor   | Best Epoch | Val Weighted F1 | Val Macro F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
| ---: | ------------ | ---------: | --------------: | -----------: | ------------: | ---------------: | ------------: |
|    1 | `mean`       |         24 |          0.8769 |       0.4268 |        0.8348 |           0.8257 |        0.4832 |
|    2 | `multiscale` |         15 |          0.8882 |       0.4575 |        0.8353 |           0.8178 |        0.4486 |
|    3 | `max`        |          3 |          0.8682 |       0.3390 |        0.7959 |           0.7518 |        0.2797 |
|    4 | `attention`  |          3 |          0.8589 |       0.3028 |        0.8033 |           0.7456 |        0.2507 |
|    5 | `gated`      |          3 |          0.8619 |       0.3042 |        0.8059 |           0.7361 |        0.2227 |

## Interpretation

The best compressor in the Mont-to-Ohm setting was `mean`.

This was a useful finding because `multiscale` achieved the best validation score, but `mean` performed better on the held-out Ohm test set. This suggests that the more expressive compressors may fit the Mont validation data better, while the simpler mean pooling compressor generalizes better under domain shift.

In other words, for this setup:

```text
mean > multiscale > max > attention > gated
```

The result also shows that a simple non-learnable baseline can be very competitive. Mean pooling likely works well because it gives a stable global summary of the utterance and avoids overfitting to domain-specific temporal patterns.

For the current AIBO Mont-to-Ohm experiments, `mean` is therefore the recommended default compressor.


## Compared to Mont → Ohm
| Direction  | Winner     | Test Acc | Test W-F1 | Test M-F1 |
| ---------- | ---------- | -------: | --------: | --------: |
| Mont → Ohm | mean       |   0.8348 |    0.8257 |    0.4832 |
| Ohm → Mont | multiscale |   0.7854 |    0.8188 |    0.3797 |


## Current recommended default

```python
compressor = build_compressor(
    "mean",
    target_len=50,
    hidden_dim=audio_dim,
)
```



## Notes

The AIBO dataset is highly imbalanced. The neutral class dominates the dataset, while the rarest class has very few samples. Because of this, weighted F1 and macro F1 should both be reported.

Weighted F1 reflects performance on the natural imbalanced distribution. Macro F1 gives a better view of how well the model handles minority emotion classes.

In the current results, `mean` wins on both weighted F1 and macro F1, which makes it the strongest compressor choice for this setup.
