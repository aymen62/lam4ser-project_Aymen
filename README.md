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

We use a speaker-independent split so the model never sees a speaker during training that it will be tested on. This is harder than a random split but more realistic.

- Train: speakers 11, 12, 13, 14, 15, 16 -- 493 samples
- Val:   speakers 09, 10 -- 161 samples
- Test:  speakers 03, 08 -- 162 samples

Speaker IDs are extracted from the first two characters of the filename, which is how EMoDB encodes them.

---

## Modules

### `data/`

`dataset.py` handles loading the pre-extracted embeddings from disk and the speaker-independent split logic. The dataset returns three things per sample: the fixed text prompt as input_ids, the audio embedding, and the label.

One thing worth noting: every sample gets the same text input ("Classify the emotion of this speech:"). The text branch of GPT-2 is just a fixed context -- all the discriminative signal has to come through the audio fusion.

### `models/audio_encoder/`

`preprocessing.py` runs offline (not during training). It loads the raw EMoDB audio files, passes each one through **wav2vec2-base-960h** (Facebook, pretrained on 960h of LibriSpeech), and saves the last hidden states to disk as a .pt file.

- Audio is padded or truncated to 8 seconds max before going into wav2vec2
- Output embedding per clip: [T_audio, 768], where T_audio varies by clip length
- Embeddings are saved together with file paths (needed for speaker splitting) and label mappings

Running this once upfront is much faster than extracting features every training epoch. The saved file is around 500MB so it is not committed to the repo -- run preprocessing.py first if you clone this.

### `models/compression/`

`compressor.py` collapses variable-length audio sequences to a fixed length of 50 tokens using temporal mean pooling. This is needed because GPT-2 expects fixed-size inputs.

Groups of consecutive frames get averaged together. For example if T_audio=399 and target_len=50, every group of 7 frames becomes one token (399 // 50 = 7, with the remainder trimmed). Not the most sophisticated approach but it works as a baseline -- alternatives like learned compression or strided convolutions could be explored later.

Output shape: [B, 50, 768]

### `models/fusion/`

This is the main contribution. Instead of just appending audio tokens to the text sequence, we inject audio information into GPT-2 at multiple layers using cross-attention adapters.

`cross_attention.py` -- CrossAttentionAdapter takes text hidden states as query and audio hidden states as key/value, runs multi-head cross-attention, passes the output through a small bottleneck MLP (hidden_dim -> adapter_dim -> hidden_dim), and adds it back to the text hidden states with a residual connection + LayerNorm. adapter_dim=64 keeps the parameter count manageable.

`fusion_block.py` -- thin wrapper around CrossAttentionAdapter, handles the case where audio_dim != text_dim with a linear projection (not needed here since both are 768).

### `models/`

`audio_gpt2.py` -- the full model. GPT-2 is loaded and fully frozen (124M params, none trained). We attach one CrossAttentionAdapter after every third GPT-2 transformer block -- specifically at layers 2, 5, 8, 11. This gives 4 fusion points covering early, middle and late representations without using 12 separate adapters (which was too many parameters for this dataset size).

After the final GPT-2 layer, we take the last token's hidden state and pass it through a small classifier (LayerNorm -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, 7)).

The last-token pooling choice follows the standard GPT-2 classification approach -- the last token has attended to all previous positions via causal self-attention, so it should carry aggregated information from the whole sequence.

Total trainable parameters: ~10M (all in the 4 fusion adapters + classifier)

### `training/`

`train_base_model.py` -- training loop with a few specific choices worth documenting:

- **Optimizer**: AdamW with lr=1e-5 and weight_decay=1e-2. The low learning rate is important here because 493 training samples is a very small dataset -- higher lr (tried 1e-4) causes the model to memorize the training set in a few epochs.
- **LR schedule**: linear warmup for 10% of steps then linear decay to 0. Warmup helps with the cross-attention weights early in training.
- **Loss**: cross-entropy with class weights computed from the training split using sklearn's compute_class_weight. Safeguard against class imbalance within the speaker split.
- **Grad clipping**: norm clipped to 1.0
- **Batch size**: 8
- **Checkpointing**: saves the best model by validation loss

Best configuration so far (after 3 training runs): 100 epochs, 4 fusion blocks, adapter_dim=64, dropout=0.3. See training_notes.txt for the full history.

### `evaluation/`

`evaluate.py` -- computes accuracy, weighted F1, and confusion matrix on the test set using the saved best checkpoint.

---

## Results so far

Best test result (run 3, epoch 81 checkpoint): **49.4% accuracy, 46.8% weighted F1**

For context, speaker-independent EMoDB baselines in the literature are roughly 65-75% for SVM-based methods and 75-85% for dedicated deep learning SER models. We are below that but this is a novel architecture and only the first week of tuning.

See `training_notes.txt` for detailed notes on all three training runs.
