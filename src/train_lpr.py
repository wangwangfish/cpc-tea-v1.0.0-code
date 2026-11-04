# -*- coding: utf-8 -*-
"""Four-cell experiment: does the gain come from the CPC operator or the readout?

configs
  r18          ResNet-18, GAP + linear          (baseline)
  r18_lpr      ResNet-18 + local prototype head
  cpc          CPC-Net (stages 3-4), GAP + linear
  cpc_lpr      CPC-Net + local prototype head
  cpc_cos      CPC-Net, GAP + cosine classifier
"""
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cpc_resnet import CPCResNet, BasicBlock          # noqa: E402
from models.local_proto_head import (                        # noqa: E402
    LocalPrototypeHead, CosineClassifier, WrapWithHead)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class C4Rotate:
    """Exact 90-degree rotation of a square tensor crop (no interpolation artifacts)."""

    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, t):
        if torch.rand(1).item() >= self.p:
            return t
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(t, k, dims=(-2, -1))

    def __repr__(self):
        return f"C4Rotate(p={self.p})"


def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def _cutmix(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)
    rh, rw = int(H * (1.0 - lam) ** 0.5), int(W * (1.0 - lam) ** 0.5)
    cy, cx = int(np.random.randint(H)), int(np.random.randint(W))
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(H * W))
    return x, y, y[idx], lam


def loaders(root, res, bs, workers, aug_rot=False, aug="basic", cls_balance="none"):
    ops = [transforms.RandomResizedCrop(res, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
           transforms.RandomHorizontalFlip(0.5)]
    if aug == "ra":
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug == "ra_wide":
        ops.append(transforms.RandAugment(num_ops=3, magnitude=12))
    elif aug == "trivial":
        ops.append(transforms.TrivialAugmentWide())
    ops.append(transforms.ToTensor())
    if aug_rot:
        ops.append(C4Rotate(0.75))
    ops.append(transforms.Normalize(MEAN, STD))
    tr = transforms.Compose(ops)
    va = transforms.Compose([
        transforms.Resize(int(round(res * 256 / 224))),
        transforms.CenterCrop(res),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), tr)
    if cls_balance in ("sampler", "both"):
        from torch.utils.data import WeightedRandomSampler
        targets = [s_[1] for s_ in train_ds.samples]
        cnt = np.bincount(targets, minlength=len(train_ds.classes)).astype(np.float64)
        w = 1.0 / np.maximum(cnt, 1.0)[targets]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        print("cls_balance=sampler: per-class counts %s" % cnt.astype(int).tolist(), flush=True)
        tr_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                               num_workers=workers, pin_memory=True)
    else:
        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=workers, pin_memory=True)
    return (tr_loader,
            DataLoader(datasets.ImageFolder(os.path.join(root, "val"), va),
                       batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return c / max(t, 1)


def dual_flags(spec, use_cpc):
    if not use_cpc:
        return [False, False, False, False]
    v = [s.strip() not in ("0", "", "false", "False") for s in spec.split(",")]
    if len(v) != 4:
        raise ValueError("dual_stages needs 4 flags, got %r" % spec)
    return v


def stage_flags(spec, use_cpc):
    """'0,1,1,1' -> [False, True, True, True]. Ignored for r18 configs."""
    if not use_cpc:
        return [False, False, False, False]
    v = [s.strip() not in ("0", "", "false", "False") for s in spec.split(",")]
    if len(v) != 4:
        raise ValueError("cpc_stages needs 4 comma-separated flags, got %r" % spec)
    return v


def build(cfg, n_cls=8, cpc_pos="both", cpc_stages="0,0,1,1", dual_stages="0,0,0,0",
          width_mult=1.0):
    use_cpc = cfg.startswith("cpc")
    backbone = CPCResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_cls,
                         cpc_stages=stage_flags(cpc_stages, use_cpc),
                         cpc_pos=cpc_pos,
                         dual_stages=dual_flags(dual_stages, use_cpc),
                         width_mult=width_mult)
    if cfg.endswith("_lpr"):
        return WrapWithHead(backbone, LocalPrototypeHead(512, n_cls), lam=1.0), "lpr"
    if cfg.endswith("_cos"):
        wrapper = WrapWithHead(backbone, nn.Identity(), lam=0.0)
        wrapper.head = nn.Identity()
        wrapper.cosine = CosineClassifier(512, n_cls)
        return wrapper, "cos"
    return backbone, "linear"


class CosWrap(nn.Module):
    def __init__(self, backbone, cos):
        super().__init__()
        self.backbone, self.cos = backbone, cos
        self._f = {}
        backbone.layer4.register_forward_hook(lambda m, i, o: self._f.__setitem__("x", o))

    def forward(self, x):
        self.backbone(x)
        return self.cos(self._f["x"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init_imagenet", action="store_true",
                    help="initialise the backbone from torchvision ImageNet weights; the classifier stays fresh")
    ap.add_argument("--disable_cudnn", action="store_true")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="1.0 = pure cosine (published CPC); <1 retains input magnitude")
    ap.add_argument("--aug", default="basic",
                    choices=["basic", "ra", "ra_wide", "trivial"],
                    help="basic = RandCrop + HFlip (historical); ra = +RandAugment(2,9)")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="Beta(a,a) mixing coefficient; 0 disables")
    ap.add_argument("--cutmix", type=float, default=0.0,
                    help="Beta(a,a) cut-and-paste coefficient; mixed with mixup 50/50")
    ap.add_argument("--aug_rot", action="store_true",
                    help="add exact C4 (90 deg) rotation augmentation")
    ap.add_argument("--c4", default="none", choices=["none", "max", "mean"],
                    help="orbit-pooled CPC prototypes over C4")
    ap.add_argument("--c4_order", type=int, default=4, choices=[2, 4])
    ap.add_argument("--op", default="v1", choices=["v1", "dc", "lcn", "soft", "mild"],
                    help="v1 = published CPC; dc/lcn = common-mode removal; "
                         "soft = energy floor; mild = directional suppression")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="soft mode: delta = kappa * mean patch norm")
    ap.add_argument("--lambda_dir", type=float, default=0.0,
                    help="mild mode: suppression strength on the analytic DC direction")
    ap.add_argument("--proto_no_wd", action="store_true",
                    help="exclude the prototype parameter from weight decay")
    ap.add_argument("--cls_balance", default="none",
                    choices=["none", "weight", "sampler", "both"],
                    help="counter class imbalance: effective-number loss weights, "
                         "weighted sampling, or both")
    ap.add_argument("--cb_beta", type=float, default=0.999,
                    help="effective-number beta for class weighting")
    ap.add_argument("--width_mult", type=float, default=1.0,
                    help="channel multiplier for the backbone; 1.0 is ResNet-18 as published")
    ap.add_argument("--head", default="gap",
                    choices=["gap", "gem", "ms", "ms_gem", "pph", "pph_gem", "lin0", "linp", "cos0", "cosp"],
                    help="gap = historical GAP+FC; gem = generalised-mean pooling; "
                         "ms = stage3+stage4 fusion; ms_gem = both")
    ap.add_argument("--conv_type", default="cpc", choices=["cpc", "wn"],
                    help="cpc = cosine prototype convolution (published operator); "
                         "wn = weight-normalised Conv2d at the same eight positions. "
                         "The second is the mechanism control, not a proposed method.")
    ap.add_argument("--dual_agg", default="none", choices=["none", "max", "soft", "mean"],
                    help="dual-prototype competition: max / soft / mean; none = single prototype")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="soft competition strength (large -> max, small -> mean)")
    ap.add_argument("--dual_init", default="mirror", choices=["mirror", "independent"],
                    help="mirror = p+-delta (symmetric); independent = second direction drawn fresh")
    ap.add_argument("--dual_delta", type=float, default=0.1,
                    help="relative perturbation splitting the prototype into two")
    ap.add_argument("--dual_stages", default="0,0,0,0",
                    help="which stages use dual-prototype CPC, e.g. 0,0,0,1 = stage 4 only")
    ap.add_argument("--cpc_stages", default="0,0,1,1",
                    help="which of the 4 stages use CPC, e.g. 0,0,1,1 = published "
                         "(8 layers), 0,1,1,1 = 12 layers, 1,1,1,1 = 16 layers")
    ap.add_argument("--cpc_pos", default="both", choices=["both", "conv1", "conv2"],
                    help="which conv inside each basic block becomes CPC: "
                         "both = published, conv1 = variant B, conv2 = variant A")
    a = ap.parse_args()

    if a.disable_cudnn:
        torch.backends.cudnn.enabled = False
    set_seed(a.seed)
    device = torch.device("cuda")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    if a.alpha < 0.999:
        import models.cpc_resnet as _cr
        from models.cpc_conv_v2 import CPCConv2dV2
        CPCConv2dV2.ALPHA = a.alpha
        _cr.CPCConv2d = CPCConv2dV2
        print(f"alpha={a.alpha}: CPCConv2d patched -> CPCConv2dV2", flush=True)

    if a.op != "v1":
        import models.cpc_resnet as _cr
        from models.cpc_conv_v3 import CPCConv2dV3
        CPCConv2dV3.MODE = a.op
        CPCConv2dV3.KAPPA = a.kappa
        CPCConv2dV3.LAMBDA_DIR = a.lambda_dir
        _cr.CPCConv2d = CPCConv2dV3
        print(f"cpc -> CPCConv2dV3 (mode={a.op}, kappa={a.kappa}, "
              f"lambda={a.lambda_dir})", flush=True)

    if a.c4 != "none":
        import models.cpc_resnet as _cr
        from models.cpc_conv_c4 import CPCConv2dC4
        CPCConv2dC4.MODE = a.c4
        CPCConv2dC4.ORDER = a.c4_order
        _cr.CPCConv2d = CPCConv2dC4
        print(f"cpc -> CPCConv2dC4 (order=C{a.c4_order}, mode={a.c4})", flush=True)

    if a.dual_agg != "none":
        from models.cpc_conv_dual import CPCConv2dDual
        CPCConv2dDual.AGG = a.dual_agg
        CPCConv2dDual.BETA = a.beta
        CPCConv2dDual.DELTA = a.dual_delta
        CPCConv2dDual.INIT = a.dual_init
        print("cpc -> CPCConv2dDual (agg=%s beta=%.2f delta=%.3f init=%s, dual_stages=%s)"
              % (a.dual_agg, a.beta, a.dual_delta, a.dual_init, a.dual_stages), flush=True)

    tr, va = loaders(a.data_dir, a.resolution, a.bs, a.workers,
                     aug_rot=a.aug_rot, aug=a.aug, cls_balance=a.cls_balance)
    if a.config.endswith("_cos"):
        use_cpc = a.config.startswith("cpc")
        base = CPCResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=8,
                         cpc_stages=stage_flags(a.cpc_stages, use_cpc),
                         cpc_pos=a.cpc_pos,
                         dual_stages=dual_flags(a.dual_stages, use_cpc))
        model = CosWrap(base, CosineClassifier(512, 8)).to(device)
        head = "cos"
    else:
        model, head = build(a.config, cpc_pos=a.cpc_pos, cpc_stages=a.cpc_stages,
                            width_mult=a.width_mult,
                            dual_stages=a.dual_stages)
        model = model.to(device)

    if a.init_imagenet:
        import torchvision.models as _tvm
        _w = _tvm.resnet18(weights=_tvm.ResNet18_Weights.IMAGENET1K_V1).state_dict()
        _own = model.state_dict()
        _got = {k: v for k, v in _w.items() if k in _own and _own[k].shape == v.shape}
        _own.update(_got)
        model.load_state_dict(_own)
        print(f"init_imagenet: loaded {len(_got)}/{len(_own)} tensors", flush=True)
    if a.op == "soft" and a.kappa > 0:
        from models.cpc_conv_v3 import CPCConv2dV3
        mods = [m for m in model.modules() if isinstance(m, CPCConv2dV3)]
        acc = [0.0] * len(mods)
        cnt = [0] * len(mods)
        handles = []
        for i, m in enumerate(mods):
            def hook(mm, inp, i=i):
                with torch.no_grad():
                    acc[i] += float(mm.mean_patch_norm(inp[0]))
                    cnt[i] += 1
            handles.append(m.register_forward_pre_hook(hook))
        model.eval()
        with torch.no_grad():
            for k, (xb, _) in enumerate(tr):
                model(xb.to(device))
                if k >= 3:
                    break
        for h in handles:
            h.remove()
        for i, m in enumerate(mods):
            r = acc[i] / max(cnt[i], 1)
            with torch.no_grad():
                m.delta.fill_(a.kappa * r)
        print("calibrated delta:", [(round(acc[i] / max(cnt[i], 1), 2),
                                     round(float(mods[i].delta), 3))
                                    for i in range(len(mods))], flush=True)

    if a.head != "gap":
        from models.head_variants import HeadVariant
        if isinstance(model, nn.Module) and not hasattr(model, "bb"):
            if hasattr(model, "fc"):
                model.fc = nn.Identity()
            _hn = 8
            _f = getattr(model, "fc", None)
            if isinstance(_f, nn.Linear):
                _hn = _f.out_features
            if a.head in ("lin0", "linp", "cos0", "cosp"):
                from models.head_variants import QuadHead
                _cc = datasets.ImageFolder(os.path.join(a.data_dir, "train")).targets
                _cnt = np.bincount(_cc, minlength=_hn).tolist()
                _uc = a.head.startswith("cos")
                _up = a.head.endswith("p")
                print("head=%s: use_cos=%s use_prior=%s" % (a.head, _uc, _up), flush=True)
                model = QuadHead(model, _cnt, use_cos=_uc, use_prior=_up).to(device)
            elif a.head.startswith("pph"):
                from models.head_variants import PriorPrototypeHead
                _cc = datasets.ImageFolder(os.path.join(a.data_dir, "train")).targets
                _cnt = np.bincount(_cc, minlength=_hn).tolist()
                print("head=pph: class counts %s -> prior bias %s"
                      % (_cnt, np.round(np.log(np.array(_cnt)/sum(_cnt)), 3).tolist()), flush=True)
                model = PriorPrototypeHead(model, _cnt, mode=a.head).to(device)
            else:
                model = HeadVariant(model, _hn, mode=a.head).to(device)
            print("head=%s: pooling replaced (params %.3fM)"
                  % (a.head, sum(p_.numel() for p_ in model.parameters()) / 1e6), flush=True)

    if a.conv_type == "wn":
        # Wrap the same eight convolutions with weight normalisation: w = g * v/||v||.
        # g is one learnable scalar per output channel, v is free, so the parameter count
        # matches a plain Conv2d to within ~2k scalars overall.
        import torch.nn.utils as _nnu
        _n = 0
        for _ln in ("layer3", "layer4"):
            _st = getattr(model, _ln, None)
            if _st is None:
                continue
            for _blk in _st:
                for _cn in ("conv1", "conv2"):
                    _c = getattr(_blk, _cn, None)
                    if isinstance(_c, nn.Conv2d):
                        setattr(_blk, _cn, _nnu.weight_norm(_c))
                        _n += 1
        print("conv_type=wn: weight-normalised %d convolutions in stages 3-4" % _n, flush=True)
        if _n != 8:
            raise SystemExit("expected 8 convolutions, wrapped %d" % _n)

    # --- structural self-check: how many CPC layers, and exactly where ---
    import models.cpc_resnet as _cr2
    _cpc_layers = [n_ for n_, m_ in model.named_modules()
                   if isinstance(m_, _cr2.CPCConv2d)]
    _pos = []
    for _ln in ("layer3", "layer4"):
        _st = getattr(model, _ln, None)
        if _st is not None:
            for _bi, _blk in enumerate(_st):
                for _cn in ("conv1", "conv2"):
                    _pos.append("%s.%d.%s=%s" % (_ln, _bi, _cn,
                                                 type(getattr(_blk, _cn)).__name__))
    print("cpc_stages=%s cpc_pos=%s n_cpc_layers=%d"
          % (a.cpc_stages, a.cpc_pos, len(_cpc_layers)), flush=True)
    _dual_mods = [m for m in model.modules() if type(m).__name__ == "CPCConv2dDual"]
    print("n_dual_layers=%d" % len(_dual_mods), flush=True)
    print("per-stage cpc counts: " + " ".join(
        "layer%d=%d" % (i + 1, sum(1 for n_ in _cpc_layers if n_.startswith("layer%d." % (i + 1))))
        for i in range(4)), flush=True)
    print("stage3-4: " + " ".join(_pos), flush=True)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if a.proto_no_wd:
        decay, no_decay = [], []
        for n_, p_ in model.named_parameters():
            (no_decay if n_.endswith("prototype") else decay).append(p_)
        groups = [{"params": decay, "weight_decay": a.wd},
                  {"params": no_decay, "weight_decay": 0.0}]
        print(f"proto_no_wd: {len(no_decay)} tensors without weight decay", flush=True)
    else:
        groups = None

    if a.optimizer == "sgd":
        opt = torch.optim.SGD(groups or model.parameters(), lr=a.lr, momentum=0.9,
                              weight_decay=a.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 100], gamma=0.1)
    else:
        opt = torch.optim.AdamW(groups or model.parameters(), lr=a.lr,
                                weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
    if a.cls_balance in ("weight", "both"):
        _tr_ds = datasets.ImageFolder(os.path.join(a.data_dir, "train"))
        _cnt = np.bincount([s_[1] for s_ in _tr_ds.samples],
                           minlength=len(_tr_ds.classes)).astype(np.float64)
        _eff = (1.0 - np.power(a.cb_beta, _cnt)) / (1.0 - a.cb_beta)
        _w = 1.0 / np.maximum(_eff, 1e-8)
        _w = _w / _w.sum() * len(_w)
        print("cls_balance=weight: counts %s -> weights %s"
              % (_cnt.astype(int).tolist(), np.round(_w, 3).tolist()), flush=True)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(_w, dtype=torch.float32).to(device))
    else:
        crit = nn.CrossEntropyLoss()
    print(f"config={a.config} head={head} params={n_par/1e6:.3f}M seed={a.seed}", flush=True)

    best, best_ep, last = 0.0, -1, 0.0
    hist = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); rl = 0.0; nb = 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if a.mixup > 0 or a.cutmix > 0:
                if a.cutmix > 0 and (a.mixup <= 0 or random.random() < 0.5):
                    xm, ya, yb, lam = _cutmix(x, y, a.cutmix)
                elif a.mixup > 0:
                    xm, ya, yb, lam = _mixup(x, y, a.mixup)
                else:
                    xm, ya, yb, lam = x, y, y, 1.0
                loss = lam * crit(model(xm), ya) + (1.0 - lam) * crit(model(xm), yb)
            else:
                loss = crit(model(x), y)
            loss.backward(); opt.step()
            rl += loss.item(); nb += 1
        sched.step()
        acc = evaluate(model, va, device) * 100
        last = acc
        hist.append({"epoch": ep, "loss": rl / max(nb, 1), "val_acc": acc})
        if acc > best:
            best, best_ep = acc, ep
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "best_acc": best / 100, "config": a.config},
                       Path(a.output_dir) / "best.pth")
        if ep % 25 == 0 or ep == 1:
            print(f"Epoch [{ep:3d}/{a.epochs}] loss={rl/max(nb,1):.4f} val={acc:.2f} "
                  f"best={best:.2f}@{best_ep} {(time.time()-t0)/60:.1f}min", flush=True)

    _wr = []
    if _dual_mods:
        for _m in _dual_mods:
            _m.reset_stats()
        model.eval()
        with torch.no_grad():
            for _xb, _ in va:
                model(_xb.to(device))
        _wr = [round(_m.win_rate(), 4) for _m in _dual_mods]
        print("dual_win_rate p0_wins:", _wr, flush=True)

    rec = {"config": a.config, "head": head, "optimizer": a.optimizer, "alpha": a.alpha,
           "aug_rot": a.aug_rot, "c4": a.c4, "c4_order": a.c4_order, "op": a.op,
           "kappa": a.kappa, "lambda_dir": a.lambda_dir, "proto_no_wd": a.proto_no_wd,
           "cpc_pos": a.cpc_pos, "cpc_stages": a.cpc_stages,
           "dual_agg": a.dual_agg, "beta": a.beta, "dual_delta": a.dual_delta,
           "dual_stages": a.dual_stages, "dual_init": a.dual_init,
           "conv_type": a.conv_type, "head_variant": a.head,
           "cls_balance": a.cls_balance, "cb_beta": a.cb_beta,
           "aug": a.aug, "mixup": a.mixup, "cutmix": a.cutmix,
           "n_dual_layers": len(_dual_mods),
           "dual_win_rate": _wr,
           "n_cpc_layers": len(_cpc_layers),
           "params_M": round(n_par / 1e6, 3),
           "seed": a.seed, "epochs": a.epochs, "resolution": a.resolution,
           "best_acc": round(best, 2), "best_epoch": best_ep,
           "last_acc": round(last, 2), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(rec, open(Path(a.output_dir) / "result.json", "w"), indent=1)
    json.dump(hist, open(Path(a.output_dir) / "history.json", "w"), indent=1)
    print("BEST", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
