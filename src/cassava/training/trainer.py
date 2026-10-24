"""Training loop: weighted loss / balanced sampling, cosine LR, early stopping on a validation metric.

Model selection uses the VALIDATION split only. The test split is never touched here.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from cassava.evaluation.metrics import classification_report
from cassava.utils import get_logger

log = get_logger()


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1 over classes (n / (K * count))."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, len(labels) / (n_classes * np.maximum(counts, 1)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32) if (counts > 0).any() else w.astype(np.float32)


def make_train_loader(ds, labels: np.ndarray, n_classes: int, cfg: dict, seed: int) -> DataLoader:
    tc, nw = cfg["train"], cfg["data"]["num_workers"]
    g = torch.Generator().manual_seed(seed)
    if tc["balance"] == "sampler":
        w = class_weights(labels, n_classes)[labels]
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(labels), replacement=True, generator=g)
        return DataLoader(ds, batch_size=tc["batch_size"], sampler=sampler, num_workers=nw, drop_last=len(ds) > tc["batch_size"])
    return DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, generator=g, num_workers=nw, drop_last=len(ds) > tc["batch_size"])


def make_eval_loader(ds, cfg: dict) -> DataLoader:
    return DataLoader(ds, batch_size=max(32, cfg["train"]["batch_size"]), shuffle=False, num_workers=cfg["data"]["num_workers"])


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probabilities [N, K], labels [N])."""
    model.eval()
    probs, labels = [], []
    for x, f, y in loader:
        logits = model(x.to(device), f.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _monitor_value(name: str, report: dict, loss: float) -> float:
    return -loss if name == "loss" else float(report[name])


def fit(model, train_ds, val_ds, train_labels, class_names, cfg, device, out_dir: Path, seed: int) -> dict:
    """Train; return {'history': DataFrame, 'best_state': state_dict, 'best_epoch': int, 'train_time_s': float}."""
    tc, k = cfg["train"], len(class_names)
    model.to(device)
    train_loader = make_train_loader(train_ds, train_labels, k, cfg, seed)
    val_loader = make_eval_loader(val_ds, cfg)
    weights = torch.from_numpy(class_weights(train_labels, k)).to(device) if tc["balance"] == "weighted_loss" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tc["label_smoothing"])

    backbone_lr = tc["lr"] * cfg["model"]["transfer"]["backbone_lr_mult"]
    is_transfer = cfg["model"]["kind"] == "transfer" or (cfg["model"]["kind"] == "fusion" and cfg["model"]["fusion"]["backbone"] != "cnn")
    if is_transfer:
        body_params = list(model.body.parameters()) if hasattr(model, "body") else list(model.backbone.body.parameters())
        ids = {id(p) for p in body_params}
        rest = [p for p in model.parameters() if id(p) not in ids]
        groups = [{"params": body_params, "lr": backbone_lr}, {"params": rest, "lr": tc["lr"]}]
    else:
        groups = [{"params": list(model.parameters()), "lr": tc["lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    if tc["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tc["epochs"]))
    elif tc["scheduler"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    else:
        sched = None

    freeze_epochs = cfg["model"]["transfer"]["freeze_epochs"] if is_transfer else 0
    best, best_epoch, best_state, bad = -np.inf, -1, None, 0
    rows, t_start = [], time.perf_counter()
    for epoch in range(tc["epochs"]):
        t0 = time.perf_counter()
        if is_transfer and hasattr(model, "set_backbone_frozen"):
            model.set_backbone_frozen(epoch < freeze_epochs)
        model.train()
        train_ds.set_epoch(epoch)
        run_loss, n_seen = 0.0, 0
        for x, f, y in train_loader:
            x, f, y = x.to(device), f.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x, f), y)
            loss.backward()
            if tc["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            run_loss += float(loss.item()) * len(y)
            n_seen += len(y)
        probs, yv = predict_loader(model, val_loader, device)
        rep = classification_report(yv, probs.argmax(1), class_names)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(yv)).item())
        score = _monitor_value(tc["monitor"], rep, val_loss)
        if sched is not None:
            sched.step(score) if tc["scheduler"] == "plateau" else sched.step()
        rows.append({
            "epoch": epoch, "train_loss": run_loss / max(1, n_seen), "val_loss": val_loss,
            "val_accuracy": rep["accuracy"], "val_macro_f1": rep["macro_f1"],
            "lr": opt.param_groups[-1]["lr"], "epoch_time_s": time.perf_counter() - t0,
        })
        log.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f (%.1fs)", epoch + 1, tc["epochs"],
                 rows[-1]["train_loss"], val_loss, rep["accuracy"], rep["macro_f1"], rows[-1]["epoch_time_s"])
        in_freeze = epoch < freeze_epochs
        if score > best:
            best, best_epoch, best_state, bad = score, epoch, copy.deepcopy(model.state_dict()), 0
        elif not in_freeze:
            bad += 1
            if tc["early_stopping_patience"] and bad >= tc["early_stopping_patience"]:
                log.info("early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch + 1)
                break
    if best_state is None:  # pragma: no cover - only if epochs == 0
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
    hist = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_dir / "history.csv", index=False)
    return {"history": hist, "best_state": best_state, "best_epoch": best_epoch, "train_time_s": time.perf_counter() - t_start}
