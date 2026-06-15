"""Extract encoder embeddings from AIBO and save as a .pt file.

Usage:
    python models/audio_encoder/preprocessing_aibo.py --encoder wav2vec2-large-emotion
"""
import argparse
import os
from dataclasses import dataclass
from typing import Type

import audiofile
import numpy as np
import torch
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)

SAMPLING_RATE = 16000
MAX_DURATION_SEC = 8.0
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLING_RATE)
EMBEDDINGS_DIR = "embeddings"

AIBO_WAV_DIR    = "/data/chi-gpu1/asl_alm_ss26/data/wav"
AIBO_LABEL_FILE = "/data/chi-gpu1/asl_alm_ss26/data/labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt"
MIN_CONFIDENCE  = 0.6  # filter out low confidence samples


@dataclass
class EncoderSpec:
    model_id: str
    hidden_dim: int
    processor_cls: Type
    model_cls: Type


ENCODERS: dict = {
    "wav2vec2-base": EncoderSpec(
        model_id="facebook/wav2vec2-base-960h",
        hidden_dim=768,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wav2vec2-large-emotion": EncoderSpec(
        model_id="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        hidden_dim=1024,
        processor_cls=Wav2Vec2Processor,
        model_cls=Wav2Vec2Model,
    ),
    "wavlm-large": EncoderSpec(
        model_id="microsoft/wavlm-large",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=WavLMModel,
    ),
    "hubert-large": EncoderSpec(
        model_id="facebook/hubert-large-ls960-ft",
        hidden_dim=1024,
        processor_cls=Wav2Vec2FeatureExtractor,
        model_cls=HubertModel,
    ),
}


def load_aibo_labels(label_file: str, min_confidence: float = 0.6):
    """Load AIBO 5-class labels, filter by confidence score."""
    samples = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            clip_id, label, confidence = parts[0], parts[1], float(parts[2])
            if confidence >= min_confidence:
                samples[clip_id] = label

    print(f"Loaded {len(samples)} samples with confidence >= {min_confidence}")
    return samples


def extract(encoder_name: str, output_path: str | None = None) -> str:
    spec = ENCODERS[encoder_name]

    if output_path is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        output_path = os.path.join(EMBEDDINGS_DIR, f"aibo_{encoder_name}_embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoder : {encoder_name} ({spec.model_id})")
    print(f"Device  : {device}")
    print(f"Output  : {output_path}")

    print("\nLoading AIBO labels...")
    samples = load_aibo_labels(AIBO_LABEL_FILE, MIN_CONFIDENCE)

    emotion_classes = sorted(set(samples.values()))
    label2idx = {label: idx for idx, label in enumerate(emotion_classes)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"Emotion classes: {emotion_classes}")
    print(f"Label mapping:   {label2idx}")

    print(f"\nLoading {spec.model_id}...")
    processor = spec.processor_cls.from_pretrained(spec.model_id)
    model = spec.model_cls.from_pretrained(spec.model_id).to(device).eval()

    embeddings, labels, file_paths = [], [], []
    skipped = 0

    for i, (clip_id, label_str) in enumerate(samples.items()):
        wav_path = os.path.join(AIBO_WAV_DIR, f"{clip_id}.wav")

        if not os.path.exists(wav_path):
            skipped += 1
            continue

        signal, _ = audiofile.read(wav_path, always_2d=False)

        if len(signal) > MAX_SAMPLES:
            signal = signal[:MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, MAX_SAMPLES - len(signal)), mode="constant")

        inputs = processor(
            signal,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            hidden = model(input_values).last_hidden_state.squeeze(0).cpu()

        embeddings.append(hidden)
        labels.append(label2idx[label_str])
        file_paths.append(wav_path)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(samples)}")

    T_audio = embeddings[0].shape[0]
    print(f"\nDone. T_audio={T_audio}, hidden_dim={spec.hidden_dim}, samples={len(embeddings)}, skipped={skipped}")

    torch.save(
        {
            "embeddings":  embeddings,
            "labels":      labels,
            "file_paths":  file_paths,
            "label2idx":   label2idx,
            "idx2label":   idx2label,
            "T_audio":     T_audio,
            "hidden_dim":  spec.hidden_dim,
            "encoder":     encoder_name,
            "model_id":    spec.model_id,
        },
        output_path,
    )
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio encoder embeddings from AIBO.")
    parser.add_argument(
        "--encoder",
        choices=list(ENCODERS),
        default="wav2vec2-large-emotion",
        help=f"Encoder to use. Options: {', '.join(ENCODERS)}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path (default: embeddings/aibo_<encoder>_embeddings.pt)",
    )
    args = parser.parse_args()
    extract(args.encoder, args.output)
