import os
import sys
import csv
import wave
import time
import copy
import gc
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from data.dataset import EmoDBFusionDataset
from models.compression.compressor import build_compressor
from models.audio_gpt2 import AudioGPT2


CONFIG = {
    "embeddings_path": "embeddings/aibo_wav2vec2-large-emotion_embeddings.pt",
    "prompt_type": "base",
    "max_prompt_length": 32,

    # Best config from hyperparameter search
    "batch_size": 16,
    "lr": 5e-6,
    "lora_rank": 8,
    "lora_lr": 5e-5,
    "epochs": 30,
    "patience": 8,
    "adapter_dim": 32,
    "dropout": 0.4,
    "target_len": 50,

    # Official AIBO Ohm→Mont setting
    "test_prefix": "Mont",
    "val_fraction": 0.15,

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "checkpoint_dir": "checkpoints/compressor_comparison_official_ohm_to_mont",
    "results_csv": "checkpoints/compressor_comparison_official_ohm_to_mont/aibo_compressor_results.csv",
    "output_plot": "checkpoints/compressor_comparison_official_ohm_to_mont/aibo_compressor_loss_curves.png",
}

# Compressors evaluated in the AIBO comparison.
COMPRESSOR_NAMES = ["mean", "max", "attention", "gated", "multiscale"]
# COMPRESSOR_NAMES = ["mean", "max", "attention", "conv1d", "gated", "multiscale"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aibo_official_ohm_mont_split(dataset):
    """
    Official-style AIBO split.

    Train: Ohm school, all speakers except the last two
    Val:   Ohm school, last two speakers
    Test:  Mont school
    """
    if dataset.file_paths is None:
        raise ValueError("AIBO split requires file paths in the embeddings file.")

    ohm_indices = []
    mont_indices = []
    ohm_speaker_to_indices = {}

    for i, path in enumerate(dataset.file_paths):
        basename = os.path.basename(path)
        file_id = os.path.splitext(basename)[0]

        parts = file_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Could not parse AIBO file id: {file_id}")

        school = parts[0]
        speaker = parts[1]

        if school == "Ohm":
            ohm_indices.append(i)

            if speaker not in ohm_speaker_to_indices:
                ohm_speaker_to_indices[speaker] = []

            ohm_speaker_to_indices[speaker].append(i)

        elif school == "Mont":
            mont_indices.append(i)

    speakers = sorted(ohm_speaker_to_indices.keys())

    if len(speakers) < 3:
        raise ValueError(
            f"Need at least 3 Ohm speakers for official split, got {len(speakers)}."
        )

    val_speakers = set(speakers[-2:])
    train_speakers = set(speakers[:-2])

    train_indices = []
    val_indices = []

    for speaker in train_speakers:
        train_indices.extend(ohm_speaker_to_indices[speaker])

    for speaker in val_speakers:
        val_indices.extend(ohm_speaker_to_indices[speaker])

    test_indices = mont_indices

    print("Official-style AIBO split summary:")
    print("  Train: Ohm speakers except last two")
    print("  Val:   last two Ohm speakers")
    print("  Test:  Mont")
    print(f"  Train speakers: {sorted(train_speakers)}")
    print(f"  Val speakers:   {sorted(val_speakers)}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")
    print()

    if not train_indices:
        raise ValueError("Train split is empty.")
    if not val_indices:
        raise ValueError("Val split is empty.")
    if not test_indices:
        raise ValueError("Test split is empty.")

    return train_indices, val_indices, test_indices


def print_label_distribution(dataset, indices, name):
    from collections import Counter

    labels = [dataset[i]["label"].item() for i in indices]
    counts = Counter(labels)

    print(f"{name} label distribution:")
    for idx in range(len(dataset.idx2label)):
        print(f"  {dataset.idx2label[idx]}: {counts.get(idx, 0)}")
    print()


def build_optimizer(model, compressor):
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

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": CONFIG["lr"]})
    if lora_params:
        groups.append({"params": lora_params, "lr": CONFIG["lora_lr"]})

    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    trainable = other_params + lora_params

    return optimizer, trainable


def evaluate(model, compressor, loader, criterion):
    device = CONFIG["device"]

    model.eval()
    compressor.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, compressor(audio))
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

    uar = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "acc": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "uar": uar,
        "confusion_matrix": cm,
    }


def clone_state_dict_to_cpu(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def load_state_dict_from_cpu(module, state_dict):
    module.load_state_dict({
        k: v.to(CONFIG["device"])
        for k, v in state_dict.items()
    })


def get_audio_duration_seconds(path, embedding=None):
    try:
        with wave.open(path, "rb") as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        pass

    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / float(info.sample_rate)
    except Exception:
        pass

    if embedding is not None:
        return float(embedding.shape[0]) / 50.0

    return -1.0


def duration_bin(seconds):
    if seconds < 2.0:
        return "0-2s"
    if seconds < 4.0:
        return "2-4s"
    if seconds < 6.0:
        return "4-6s"
    if seconds < 8.0:
        return "6-8s"
    return ">8s"


def metrics_from_predictions(labels, preds):
    if len(labels) == 0:
        return {
            "samples": 0,
            "acc": 0.0,
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "uar": 0.0,
        }

    return {
        "samples": len(labels),
        "acc": sum(p == y for p, y in zip(preds, labels)) / len(labels),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "uar": recall_score(labels, preds, average="macro", zero_division=0),
    }


def evaluate_by_duration(model, compressor, dataset, indices):
    device = CONFIG["device"]

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    durations = []
    for idx in indices:
        path = dataset.file_paths[idx] if dataset.file_paths is not None else ""
        embedding = dataset.embeddings[idx]
        durations.append(get_audio_duration_seconds(path, embedding))

    model.eval()
    compressor.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, compressor(audio))
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    bins = ["0-2s", "2-4s", "4-6s", "6-8s", ">8s"]
    grouped = {
        b: {"labels": [], "preds": [], "durations": []}
        for b in bins
    }

    for y, pred, dur in zip(all_labels, all_preds, durations):
        b = duration_bin(dur)
        grouped[b]["labels"].append(y)
        grouped[b]["preds"].append(pred)
        grouped[b]["durations"].append(dur)

    rows = []
    for b in bins:
        labels = grouped[b]["labels"]
        preds = grouped[b]["preds"]
        ds = grouped[b]["durations"]
        m = metrics_from_predictions(labels, preds)

        rows.append({
            "duration_bin": b,
            "samples": m["samples"],
            "avg_duration": sum(ds) / len(ds) if ds else 0.0,
            "acc": m["acc"],
            "weighted_f1": m["weighted_f1"],
            "macro_f1": m["macro_f1"],
            "uar": m["uar"],
        })

    return rows


def save_duration_results(compressor_name, rows):
    path = os.path.join(
        CONFIG["checkpoint_dir"],
        f"{compressor_name}_duration_results.csv",
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "compressor",
                "duration_bin",
                "samples",
                "avg_duration",
                "acc",
                "weighted_f1",
                "macro_f1",
                "uar",
            ],
        )
        writer.writeheader()

        for row in rows:
            out = dict(row)
            out["compressor"] = compressor_name
            writer.writerow(out)

    print(f"  [{compressor_name}] Saved duration results to: {path}", flush=True)


def print_duration_results(compressor_name, rows):
    print(f"  [{compressor_name}] Test performance by duration:")
    print("    Bin    Samples  AvgDur  Acc     W-F1    M-F1    UAR")
    print("    -------------------------------------------------------")

    for r in rows:
        print(
            f"    {r['duration_bin']:<6} "
            f"{r['samples']:<7} "
            f"{r['avg_duration']:<7.2f} "
            f"{r['acc']:<7.4f} "
            f"{r['weighted_f1']:<7.4f} "
            f"{r['macro_f1']:<7.4f} "
            f"{r['uar']:<7.4f}"
        )

    print()

def run_one(compressor_name, loaders, dataset, class_weights):
    set_seed(CONFIG["seed"])

    device = CONFIG["device"]
    epochs = CONFIG["epochs"]
    patience = CONFIG["patience"]

    train_loader, val_loader, test_loader = loaders

    audio_dim = dataset.embeddings[0].shape[-1]
    num_classes = len(dataset.label2idx)

    compressor = build_compressor(
        compressor_name,
        target_len=CONFIG["target_len"],
        hidden_dim=audio_dim,
    ).to(device)

    model = AudioGPT2(
        num_classes=num_classes,
        audio_dim=audio_dim,
        adapter_dim=CONFIG["adapter_dim"],
        dropout=CONFIG["dropout"],
        lora_rank=CONFIG["lora_rank"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )

    optimizer, trainable = build_optimizer(model, compressor)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses = []
    val_losses = []

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_weighted_f1 = -1.0
    best_val_macro_f1 = -1.0
    best_val_uar = -1.0

    best_model_state = None
    best_compressor_state = None

    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        compressor.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            audio_compressed = compressor(audio)
            logits = model(input_ids, audio_compressed)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        val_metrics = evaluate(model, compressor, val_loader, criterion)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_weighted_f1 = val_metrics["weighted_f1"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_uar = val_metrics["uar"]

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        improved = val_weighted_f1 > best_val_weighted_f1

        if improved:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_weighted_f1 = val_weighted_f1
            best_val_macro_f1 = val_macro_f1
            best_val_uar = val_uar
            best_model_state = clone_state_dict_to_cpu(model)
            best_compressor_state = clone_state_dict_to_cpu(compressor)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  [{compressor_name}] epoch {epoch:2d}/{epochs} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_w_f1: {val_weighted_f1:.4f} | "
            f"val_m_f1: {val_macro_f1:.4f} | val_uar: {val_uar:.4f}"
            + ("  ✓" if improved else ""),
            flush=True,
        )

        if epochs_without_improvement >= patience:
            print(
                f"  [{compressor_name}] Early stopping: no val weighted F1 improvement "
                f"for {patience} epochs.",
                flush=True,
            )
            break

    if best_model_state is None or best_compressor_state is None:
        raise RuntimeError(f"No best checkpoint stored for compressor {compressor_name}")

    load_state_dict_from_cpu(model, best_model_state)
    load_state_dict_from_cpu(compressor, best_compressor_state)

    test_metrics = evaluate(model, compressor, test_loader, criterion)

    elapsed_minutes = (time.time() - start_time) / 60.0

    print(f"\n  [{compressor_name}] Best validation checkpoint:")
    print(f"    epoch:        {best_epoch}")
    print(f"    val_loss:     {best_val_loss:.4f}")
    print(f"    val_acc:      {best_val_acc:.4f}")
    print(f"    val_w_f1:     {best_val_weighted_f1:.4f}")
    print(f"    val_m_f1:     {best_val_macro_f1:.4f}")
    print(f"    val_uar:      {best_val_uar:.4f}")

    print(f"  [{compressor_name}] Mont test results:")
    print(f"    test_loss:    {test_metrics['loss']:.4f}")
    print(f"    test_acc:     {test_metrics['acc']:.4f}")
    print(f"    test_w_f1:    {test_metrics['weighted_f1']:.4f}")
    print(f"    test_m_f1:    {test_metrics['macro_f1']:.4f}")
    print(f"    test_uar:     {test_metrics['uar']:.4f}")

    duration_rows = evaluate_by_duration(
        model,
        compressor,
        dataset,
        CONFIG["duration_test_indices"],
    )
    print_duration_results(compressor_name, duration_rows)
    save_duration_results(compressor_name, duration_rows)
    print(f"    confusion_matrix:")
    print(test_metrics["confusion_matrix"])
    print()

    result = {
        "name": compressor_name,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "val_weighted_f1": best_val_weighted_f1,
        "val_macro_f1": best_val_macro_f1,
        "val_uar": best_val_uar,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_uar": test_metrics["uar"],
        "elapsed_minutes": elapsed_minutes,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    del model
    del compressor
    del optimizer
    del scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return result


def save_results_csv(results):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    ranked = sorted(
        results,
        key=lambda r: (r["test_weighted_f1"], r["test_macro_f1"]),
        reverse=True,
    )

    fieldnames = [
        "rank",
        "compressor",
        "best_epoch",
        "val_loss",
        "val_acc",
        "val_weighted_f1",
        "val_macro_f1",
        "val_uar",
        "test_loss",
        "test_acc",
        "test_weighted_f1",
        "test_macro_f1",
        "test_uar",
        "elapsed_minutes",
    ]

    with open(CONFIG["results_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, r in enumerate(ranked, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "compressor": r["name"],
                    "best_epoch": r["best_epoch"],
                    "val_loss": r["val_loss"],
                    "val_acc": r["val_acc"],
                    "val_weighted_f1": r["val_weighted_f1"],
                    "val_macro_f1": r["val_macro_f1"],
                    "val_uar": r["val_uar"],
                    "test_loss": r["test_loss"],
                    "test_acc": r["test_acc"],
                    "test_weighted_f1": r["test_weighted_f1"],
                    "test_macro_f1": r["test_macro_f1"],
                    "test_uar": r["test_uar"],
                    "elapsed_minutes": r["elapsed_minutes"],
                }
            )

    return ranked


def plot_losses(results):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    plt.figure(figsize=(10, 5))
    for r in results:
        epochs_range = range(1, len(r["train_losses"]) + 1)
        plt.plot(epochs_range, r["train_losses"], label=f"{r['name']} train")

    plt.title("Train Loss by Compressor")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    train_plot = os.path.join(CONFIG["checkpoint_dir"], "aibo_compressor_train_loss.png")
    plt.savefig(train_plot)
    plt.close()

    plt.figure(figsize=(10, 5))
    for r in results:
        epochs_range = range(1, len(r["val_losses"]) + 1)
        plt.plot(epochs_range, r["val_losses"], label=f"{r['name']} val")

    plt.title("Validation Loss by Compressor")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    val_plot = os.path.join(CONFIG["checkpoint_dir"], "aibo_compressor_val_loss.png")
    plt.savefig(val_plot)
    plt.close()

    print(f"Saved train loss plot to: {train_plot}")
    print(f"Saved val loss plot to:   {val_plot}")


def main():
    print("=" * 70)
    print("AIBO COMPRESSOR COMPARISON: OFFICIAL OHM TO MONT SPLIT")
    print("=" * 70)
    print(f"Device:       {CONFIG['device']}")
    print(f"Embeddings:   {CONFIG['embeddings_path']}")
    print(f"Compressors:  {COMPRESSOR_NAMES}")
    print(f"LR:           {CONFIG['lr']}")
    print(f"LoRA rank:    {CONFIG['lora_rank']}")
    print(f"LoRA LR:      {CONFIG['lora_lr']}")
    print(f"Dropout:      {CONFIG['dropout']}")
    print(f"Adapter dim:  {CONFIG['adapter_dim']}")
    print(f"Batch size:   {CONFIG['batch_size']}")
    print(f"Epochs:       {CONFIG['epochs']}")
    print(f"Patience:     {CONFIG['patience']}")
    print()

    set_seed(CONFIG["seed"])

    print("Loading dataset...")
    dataset = EmoDBFusionDataset(
        CONFIG["embeddings_path"],
        prompt_type=CONFIG["prompt_type"],
        max_length=CONFIG["max_prompt_length"],
    )

    train_idx, val_idx, test_idx = aibo_official_ohm_mont_split(dataset)
    CONFIG["duration_test_indices"] = test_idx

    print_label_distribution(dataset, train_idx, "Train")
    print_label_distribution(dataset, val_idx, "Val")
    print_label_distribution(dataset, test_idx, "Test")

    generator = torch.Generator().manual_seed(CONFIG["seed"])

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    train_labels = [dataset[i]["label"].item() for i in train_idx]
    num_classes = len(dataset.label2idx)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(num_classes),
        y=train_labels,
    )

    print("Class weights:")
    for idx, weight in enumerate(class_weights):
        print(f"  {dataset.idx2label[idx]}: {weight:.4f}")
    print()

    results = []

    for name in COMPRESSOR_NAMES:
        print("\n" + "=" * 70)
        print(f"Compressor: {name.upper()}")
        print("=" * 70)

        result = run_one(
            name,
            loaders=(train_loader, val_loader, test_loader),
            dataset=dataset,
            class_weights=class_weights,
        )

        results.append(result)

        ranked_so_far = save_results_csv(results)
        print("Current ranking by Mont test weighted F1:")
        for rank, r in enumerate(ranked_so_far, 1):
            print(
                f"  {rank}. {r['name']:<12} "
                f"test_w_f1={r['test_weighted_f1']:.4f} | "
                f"test_m_f1={r['test_macro_f1']:.4f} | "
                f"test_acc={r['test_acc']:.4f}"
            )

    ranked = save_results_csv(results)
    plot_losses(results)

    print("\n" + "=" * 70)
    print("FINAL COMPRESSOR COMPARISON")
    print("Ranked by Mont test weighted F1, then test macro F1")
    print("=" * 70)

    header = (
        f"{'Rank':<5} "
        f"{'Compressor':<12} "
        f"{'Epoch':<7} "
        f"{'Val W-F1':<10} "
        f"{'Val M-F1':<10} "
        f"{'Test Acc':<10} "
        f"{'Test W-F1':<11} "
        f"{'Test M-F1':<11}"
    )

    print(header)
    print("-" * len(header))

    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5} "
            f"{r['name']:<12} "
            f"{r['best_epoch']:<7} "
            f"{r['val_weighted_f1']:<10.4f} "
            f"{r['val_macro_f1']:<10.4f} "
            f"{r['test_acc']:<10.4f} "
            f"{r['test_weighted_f1']:<11.4f} "
            f"{r['test_macro_f1']:<11.4f}"
        )

    print(f"\nWinner by Mont test weighted F1: {ranked[0]['name'].upper()}")
    print(f"Saved CSV to: {CONFIG['results_csv']}")


if __name__ == "__main__":
    main()
