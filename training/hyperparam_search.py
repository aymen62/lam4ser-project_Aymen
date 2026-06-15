import os
import sys
import csv
import json
import time
import gc
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from data.dataset import EmoDBFusionDataset
from models.compression.compressor import build_compressor
from models.audio_gpt2 import AudioGPT2


# ============================================================
# AIBO hyperparameter search config
# ============================================================

EMBEDDINGS_PATH = "embeddings/aibo_wav2vec2-large-emotion_embeddings.pt"

PROMPT_TYPE = "base"
MAX_PROMPT_LENGTH = 32

# Current experiment:
#   Train/val on Mont
#   Test held out as Ohm
#
# IMPORTANT:
#   This script does NOT evaluate on test.
#   It only uses train/val to choose hyperparameters.
TEST_PREFIX = "Ohm"
VAL_FRACTION = 0.15

EPOCHS = 30
PATIENCE = 8
BATCH_SIZE = 16
TARGET_AUDIO_LEN = 50
COMPRESSOR = "multiscale"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "checkpoints/hparam_search"
RESULTS_CSV = os.path.join(RESULTS_DIR, "aibo_mont_to_ohm_hparam_results.csv")
RESULTS_JSON = os.path.join(RESULTS_DIR, "aibo_mont_to_ohm_hparam_results.json")


# ============================================================
# Grid
# ============================================================

GRID = [
    # Adapter dim comparison with stable LoRA-8 setup
    {"lr": 5e-6, "lora_rank": 8,  "lora_lr": 5e-5, "dropout": 0.4, "adapter_dim": 32},
    {"lr": 5e-6, "lora_rank": 8,  "lora_lr": 5e-5, "dropout": 0.4, "adapter_dim": 64},
    {"lr": 5e-6, "lora_rank": 8,  "lora_lr": 5e-5, "dropout": 0.4, "adapter_dim": 128},

    # Dropout check
    {"lr": 5e-6, "lora_rank": 8,  "lora_lr": 5e-5, "dropout": 0.5, "adapter_dim": 32},
    {"lr": 5e-6, "lora_rank": 8,  "lora_lr": 5e-5, "dropout": 0.5, "adapter_dim": 64},

    # Lower LR check
    {"lr": 3e-6, "lora_rank": 8,  "lora_lr": 3e-5, "dropout": 0.4, "adapter_dim": 32},
    {"lr": 3e-6, "lora_rank": 8,  "lora_lr": 3e-5, "dropout": 0.4, "adapter_dim": 64},

    # LoRA rank 16 check
    {"lr": 5e-6, "lora_rank": 16, "lora_lr": 5e-5, "dropout": 0.4, "adapter_dim": 32},
    {"lr": 5e-6, "lora_rank": 16, "lora_lr": 5e-5, "dropout": 0.4, "adapter_dim": 64},
]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aibo_mont_ohm_split(
    dataset,
    test_prefix: str = "Ohm",
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """
    AIBO split:
      - Test set = all samples whose basename starts with test_prefix, e.g. Ohm
      - Train/Val = all remaining samples, e.g. Mont, randomly split

    This script does not use the test set for hyperparameter selection.
    It only prints the size so we know what is held out.
    """
    if dataset.file_paths is None:
        raise ValueError("AIBO Mont/Ohm split requires file paths in the embeddings file.")

    train_val_indices = []
    test_indices = []

    for i, path in enumerate(dataset.file_paths):
        basename = os.path.basename(path)

        if basename.startswith(test_prefix):
            test_indices.append(i)
        else:
            train_val_indices.append(i)

    if not train_val_indices:
        raise ValueError("AIBO train/val split is empty.")
    if not test_indices:
        raise ValueError("AIBO test split is empty.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_val_indices), generator=generator).tolist()

    val_size = int(val_fraction * len(train_val_indices))
    val_positions = set(perm[:val_size])

    val_indices = [
        idx for pos, idx in enumerate(train_val_indices)
        if pos in val_positions
    ]

    train_indices = [
        idx for pos, idx in enumerate(train_val_indices)
        if pos not in val_positions
    ]

    print("AIBO Mont→Ohm split summary:")
    print(f"  Train/Val source: not {test_prefix}")
    print(f"  Test source:      {test_prefix}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")
    print()

    return train_indices, val_indices, test_indices


def print_label_distribution(dataset, indices, name: str):
    labels = [dataset[i]["label"].item() for i in indices]
    counts = Counter(labels)

    print(f"{name} label distribution:")
    for idx in range(len(dataset.idx2label)):
        label_name = dataset.idx2label[idx]
        count = counts.get(idx, 0)
        print(f"  {label_name}: {count}")
    print()


def build_optimizer(model, compressor, params):
    """
    Use separate LR for LoRA parameters and the rest.
    Compressor parameters are trained with the main LR.
    """
    lora_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.endswith((".A", ".B"))
    ]

    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.endswith((".A", ".B"))
    ] + [
        p for p in compressor.parameters() if p.requires_grad
    ]

    optimizer_groups = []

    if other_params:
        optimizer_groups.append(
            {"params": other_params, "lr": params["lr"]}
        )

    if lora_params:
        optimizer_groups.append(
            {"params": lora_params, "lr": params["lora_lr"]}
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=1e-2,
    )

    trainable_params = other_params + lora_params

    return optimizer, trainable_params


def evaluate(model, compressor, loader, criterion):
    model.eval()
    compressor.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return avg_loss, acc, weighted_f1, macro_f1


def run_one(params, run_id, train_loader, val_loader, dataset, class_weights):
    set_seed(SEED + run_id)

    audio_dim = dataset.embeddings[0].shape[-1]
    num_classes = len(dataset.label2idx)

    compressor = build_compressor(
        COMPRESSOR,
        target_len=TARGET_AUDIO_LEN,
        hidden_dim=audio_dim,
    ).to(DEVICE)

    model = AudioGPT2(
        num_classes=num_classes,
        audio_dim=audio_dim,
        adapter_dim=params["adapter_dim"],
        dropout=params["dropout"],
        lora_rank=params["lora_rank"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    )

    optimizer, trainable_params = build_optimizer(model, compressor, params)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_val_macro_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        compressor.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        val_loss, val_acc, val_f1, val_macro_f1 = evaluate(
            model,
            compressor,
            val_loader,
            criterion,
        )

        improved = val_f1 > best_val_f1

        if improved:
            best_val_f1 = val_f1
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch {epoch:2d}/{EPOCHS} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_weighted_f1: {val_f1:.4f} | "
            f"val_macro_f1: {val_macro_f1:.4f}"
            + ("  ✓" if improved else ""),
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"  Early stopping: no val_weighted_f1 improvement for {PATIENCE} epochs.",
                flush=True,
            )
            break

    elapsed_minutes = (time.time() - start_time) / 60.0

    result = {
        "run_id": run_id,
        "lr": params["lr"],
        "lora_rank": params["lora_rank"],
        "lora_lr": params["lora_lr"],
        "dropout": params["dropout"],
        "adapter_dim": params["adapter_dim"],
        "compressor": COMPRESSOR,
        "target_audio_len": TARGET_AUDIO_LEN,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_weighted_f1": best_val_f1,
        "best_val_macro_f1": best_val_macro_f1,
        "elapsed_minutes": elapsed_minutes,
    }

    del model
    del compressor
    del optimizer
    del scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return result


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ranked = sorted(
        results,
        key=lambda r: (r["best_val_weighted_f1"], r["best_val_macro_f1"]),
        reverse=True,
    )

    fieldnames = [
        "rank",
        "run_id",
        "lr",
        "lora_rank",
        "lora_lr",
        "dropout",
        "adapter_dim",
        "compressor",
        "target_audio_len",
        "batch_size",
        "epochs_requested",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "best_val_weighted_f1",
        "best_val_macro_f1",
        "elapsed_minutes",
    ]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, result in enumerate(ranked, 1):
            row = {"rank": rank, **result}
            writer.writerow(row)

    with open(RESULTS_JSON, "w") as f:
        json.dump(ranked, f, indent=2)

    return ranked


def main():
    print("=" * 70)
    print("AIBO HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Device:          {DEVICE}")
    print(f"Embeddings:      {EMBEDDINGS_PATH}")
    print(f"Prompt type:     {PROMPT_TYPE}")
    print(f"Compressor:      {COMPRESSOR}")
    print(f"Target len:      {TARGET_AUDIO_LEN}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Patience:        {PATIENCE}")
    print(f"Grid size:       {len(GRID)}")
    print()

    set_seed(SEED)

    print("Loading dataset...")
    dataset = EmoDBFusionDataset(
        EMBEDDINGS_PATH,
        prompt_type=PROMPT_TYPE,
        max_length=MAX_PROMPT_LENGTH,
    )

    train_idx, val_idx, test_idx = aibo_mont_ohm_split(
        dataset,
        test_prefix=TEST_PREFIX,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    print_label_distribution(dataset, train_idx, "Train")
    print_label_distribution(dataset, val_idx, "Val")
    print_label_distribution(dataset, test_idx, "Held-out test")

    train_labels = [dataset[i]["label"].item() for i in train_idx]

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(len(dataset.label2idx)),
        y=train_labels,
    )

    print("Class weights computed from train split:")
    for idx, weight in enumerate(class_weights):
        print(f"  {dataset.idx2label[idx]}: {weight:.4f}")
    print()

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []

    for i, params in enumerate(GRID, 1):
        print("\n" + "=" * 70)
        print(
            f"Run {i}/{len(GRID)} | "
            f"lr={params['lr']} | "
            f"lora_rank={params['lora_rank']} | "
            f"lora_lr={params['lora_lr']} | "
            f"dropout={params['dropout']} | "
            f"adapter_dim={params['adapter_dim']}"
        )
        print("=" * 70)

        result = run_one(
            params=params,
            run_id=i,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset=dataset,
            class_weights=class_weights,
        )

        results.append(result)

        ranked_so_far = save_results(results)
        best_so_far = ranked_so_far[0]

        print("\nBest so far:")
        print(
            f"  run_id={best_so_far['run_id']} | "
            f"weighted_f1={best_so_far['best_val_weighted_f1']:.4f} | "
            f"macro_f1={best_so_far['best_val_macro_f1']:.4f} | "
            f"acc={best_so_far['best_val_acc']:.4f} | "
            f"epoch={best_so_far['best_epoch']}"
        )

    ranked = save_results(results)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("Ranked by validation weighted F1, then macro F1")
    print("=" * 70)

    header = (
        f"{'Rank':<5} "
        f"{'Run':<5} "
        f"{'lr':<10} "
        f"{'LoRA':<6} "
        f"{'LoRA LR':<10} "
        f"{'Dropout':<8} "
        f"{'Adapter':<8} "
        f"{'Epoch':<7} "
        f"{'Val Acc':<9} "
        f"{'W-F1':<9} "
        f"{'M-F1':<9}"
    )

    print(header)
    print("-" * len(header))

    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5} "
            f"{r['run_id']:<5} "
            f"{r['lr']:<10} "
            f"{r['lora_rank']:<6} "
            f"{r['lora_lr']:<10} "
            f"{r['dropout']:<8} "
            f"{r['adapter_dim']:<8} "
            f"{r['best_epoch']:<7} "
            f"{r['best_val_acc']:<9.4f} "
            f"{r['best_val_weighted_f1']:<9.4f} "
            f"{r['best_val_macro_f1']:<9.4f}"
        )

    best = ranked[0]

    print("\nBest config:")
    print(
        f"  lr={best['lr']}, "
        f"lora_rank={best['lora_rank']}, "
        f"lora_lr={best['lora_lr']}, "
        f"dropout={best['dropout']}, "
        f"adapter_dim={best['adapter_dim']}"
    )

    print(f"\nSaved CSV:  {RESULTS_CSV}")
    print(f"Saved JSON: {RESULTS_JSON}")

    print("\nImportant:")
    print("  This script does not evaluate on Ohm test.")
    print("  After choosing the best config, run train_base_model.py once")
    print("  with that config and evaluate the held-out Ohm test set.")


if __name__ == "__main__":
    main()
