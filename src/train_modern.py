# -*- coding: utf-8 -*-
"""Architecture-adaptation protocol: modern backbones from scratch with AdamW.

Purpose: test whether the reported from-scratch failure of modern architectures
persists under a comparable tuning budget, or is an artefact of forcing SGD.
"""
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_loaders(root, res, bs, workers=8):
    tr = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    va = transforms.Compose([
        transforms.Resize(int(round(res * 256 / 224))),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dtr = datasets.ImageFolder(os.path.join(root, "train"), tr)
    dva = datasets.ImageFolder(os.path.join(root, "val"), va)
    return (DataLoader(dtr, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False),
            DataLoader(dva, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True),
            len(dtr), len(dva))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--disable_cudnn", action="store_true")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    a = ap.parse_args()

    if a.disable_cudnn:
        torch.backends.cudnn.enabled = False
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    tr_loader, va_loader, n_tr, n_va = build_loaders(a.data_dir, a.resolution, a.bs, a.workers)
    model = timm.create_model(a.model, pretrained=False, num_classes=8).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    if a.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                              weight_decay=a.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 100], gamma=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
    crit = nn.CrossEntropyLoss()

    print(f"model={a.model} opt={a.optimizer} params={n_par/1e6:.1f}M lr={a.lr} wd={a.wd} "
          f"train={n_tr} val={n_va} res={a.resolution} cudnn={not a.disable_cudnn}", flush=True)

    best = 0.0; best_ep = -1; t0 = time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        run_loss = 0.0; nb = 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item(); nb += 1
        sched.step()
        acc = evaluate(model, va_loader, device) * 100
        hist.append({"epoch": ep, "loss": run_loss / max(nb, 1), "val_acc": acc})
        if acc > best:
            best, best_ep = acc, ep
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "best_acc": best / 100, "arch": a.model}, Path(a.output_dir) / "best.pth")
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch [{ep:3d}/{a.epochs}] loss={run_loss/max(nb,1):.4f} val={acc:.2f} "
                  f"best={best:.2f}@{best_ep} {(time.time()-t0)/60:.1f}min", flush=True)

    rec = {"arch": a.model, "optimizer": a.optimizer, "params_M": round(n_par / 1e6, 2), "lr": a.lr, "wd": a.wd,
           "seed": a.seed, "resolution": a.resolution, "epochs": a.epochs,
           "best_acc": round(best, 2), "best_epoch": best_ep,
           "final_acc": round(hist[-1]["val_acc"], 2) if hist else None,
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(rec, open(Path(a.output_dir) / "result.json", "w"), indent=1)
    json.dump(hist, open(Path(a.output_dir) / "history.json", "w"), indent=1)
    print("BEST", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
