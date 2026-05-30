# lam4ser-project

Large Audio Models for Speech Emotion Recognition.

This repository contains the group implementation for the ASL project.

## Current focus

- preprocessing and audio embedding extraction
- audio token compression and projection
- audio-LLM fusion with cross-attention/adapters
- training and evaluation on SER datasets

---

## Dataset

**EMoDB** -- German emotional speech corpus, 816 samples, 7 emotion classes (anger, boredom, disgust, fear, happiness, neutral, sadness).

We use a speaker-independent split so the model never sees a speaker during training that it will be tested on.

- Train: speakers 11, 12, 13, 14, 15, 16 -- 493 samples
- Val:   speakers 09, 10 -- 161 samples
- Test:  speakers 03, 08 -- 162 samples

Speaker IDs are extracted from the first two characters of the filename, which is how EMoDB encodes them.

---

## Modules

### `data/`

`dataset.py` handles loading the pre-extracted embeddings from disk and the speaker-independent split logic. The dataset returns three things per sample: the fixed text prompt as input_ids, the audio embedding, and the label.

Every sample gets the same text input ("Classify the emotion of this speech:").

### `models/audio_encoder/`

`preprocessing.py` runs offline (not during training). It loads the raw EMoDB audio files, passes each one through a chosen encoder, and saves the last hidden states to disk as a .pt file. The encoder is selected via `--encoder`:

```
python models/audio_encoder/preprocessing.py --encoder wavlm-large
python models/audio_encoder/preprocessing.py --encoder wav2vec2-large-emotion
python models/audio_encoder/preprocessing.py --encoder wav2vec2-base   # default
python models/audio_encoder/preprocessing.py --encoder hubert-large
```

Output goes to `embeddings/{encoder-name}_embeddings.pt`. Supported encoders and their output dimensions:

| Key | Model | Dim |
|---|---|---|
| `wav2vec2-base` | facebook/wav2vec2-base-960h | 768 |
| `wav2vec2-large-emotion` | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | 1024 |
| `wavlm-large` | microsoft/wavlm-large | 1024 |
| `hubert-large` | facebook/hubert-large-ls960-ft | 1024 |

- Audio is padded or truncated to 8 seconds max before encoding
- Output embedding per clip: [T_audio, audio_dim], where T_audio varies by clip length
- Embeddings are saved together with file paths (needed for speaker splitting) and label mappings
- To add a new encoder, add one entry to the `ENCODERS` dict in preprocessing.py

### `models/compression/`

`compressor.py` collapses variable-length audio sequences to a fixed length of 50 tokens using temporal mean pooling. This is needed because GPT-2 expects fixed-size inputs.

Output shape: [B, 50, audio_dim] (audio_dim depends on the encoder, e.g. 768 or 1024)

### `models/fusion/`

We inject audio information into GPT-2 at multiple layers using cross-attention adapters.

`cross_attention.py` -- CrossAttentionAdapter takes text hidden states as query and audio hidden states as key/value, runs multi-head cross-attention, passes the output through a small bottleneck MLP (hidden_dim -> adapter_dim -> hidden_dim), and adds it back to the text hidden states with a residual connection + LayerNorm.

`fusion_block.py` -- thin wrapper around CrossAttentionAdapter, handles the case where audio_dim != text_dim with a linear projection.

### `models/`

`audio_gpt2.py` -- the full model. GPT-2 is loaded and fully frozen (124M params, none trained). We attach one CrossAttentionAdapter after every third GPT-2 transformer block. Specifically at layers 2, 5, 8, 11.

After the final GPT-2 layer, we take the last token's hidden state and pass it through a small classifier (LayerNorm -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, 7)).

`AudioGPT2` accepts an `audio_dim` argument (default 768) which the training script reads automatically from the loaded embeddings file. If `audio_dim != 768`, each fusion block adds a Linear(audio_dim, 768) projection before the cross-attention.

Trainable parameters: ~10M with 768-dim encoder, ~13.2M with 1024-dim encoder (the extra 3.1M are the four projection layers)

### `training/`

`train_base_model.py` -- training loop. Some specific choices worth noting:

- **Optimizer**: AdamW with lr=1e-5 and weight_decay=1e-2. The low learning rate is because 493 training samples is very small dataset. A higher lr (tried 1e-4) causes the model to memorize the training set in a few epochs.
- **LR schedule**: linear warmup for 10% of steps then linear decay to 0. Warmup helps with the cross-attention weights early in training.
- **Loss**: cross-entropy with class weights computed from the training split using sklearn's compute_class_weight. Safeguard against class imbalance within the speaker split.
- **Grad clipping**: norm clipped to 1.0
- **Batch size**: 8
- **Checkpointing**: saves the best model by validation loss

Best configuration so far (after 4 training runs): 100 epochs, 4 fusion blocks, adapter_dim=64, dropout=0.3. See training_notes.txt for the full history.

### `evaluation/`

`evaluate.py` -- computes accuracy, weighted F1, and confusion matrix on the test set using the saved best checkpoint.

---

## Results so far

Best test result (run 4, epoch 33 checkpoint, WavLM-large encoder): **88.3% accuracy, 88.2% weighted F1**

The encoder was the dominant factor. Swapping wav2vec2-base for WavLM-large took AudioGPT2 from 49.4% to 88.3% with no other changes. Note that EMoDB is acted speech in a clean studio. Numbers will be lower on naturalistic datasets. See `training_notes.txt` for the full run history.
