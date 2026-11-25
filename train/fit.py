"""The training loop: the encoder's top layers (or only its Dense layer) plus
one prototype per label, trained against hierarchy-aware soft targets."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import PrototypeHead, apply_suffix, encode, features_to, pooled_features, set_trainable

PROGRESS_SECONDS = 30   # a progress line this often during an epoch


@dataclass
class FitConfig:
    unfreeze_layers: int = 4        # 0: only the Dense layer and the prototypes train (fast; CPU is fine)
    epochs: Optional[int] = None    # default: 40 when unfreeze_layers is 0, else 4
    batch_size: int = 32            # with unfreeze_layers 0 the batch is at least 256
    max_seq_length: int = 256
    lr_encoder: float = 2e-5
    lr_dense: float = 3e-4
    lr_protos: float = 5e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    scale: float = 20.0             # cosine logit scale
    ancestor_eps: float = 0.1       # target mass moved from a label to its ancestors
    anchor_weight: float = 0.1      # pull toward stock LaBSE, to keep the space multilingual
    pseudo_weight: float = 0.5      # loss weight of a pseudo-label, times its confidence
    balance: float = 0.0            # 1: every label weighs the same in all, whatever its size; 0: by its rows
    node_cap: int = 0               # the coredrill uses at most this many rows per node (0: no cap)
    radius_pull: float = 0.0        # the coredrill: how far each node's radius moves toward the median (0 to 1)
    centers_per: int = 0            # the coredrill: one center per this many rows of a node (0: one center)
    centers_max: int = 1            # the coredrill: centers per node, at most
    val_frac: float = 0.1
    patience: int = 3
    seed: int = 42


@dataclass
class TrainSet:
    texts: List[str]
    y: np.ndarray          # label index per row
    weight: np.ndarray     # loss weight per row
    f0: np.ndarray         # stock LaBSE embedding per row (the anchor target)
    labeled: np.ndarray    # bool: labeled by a person; validation draws only from these


def stratified_split(y: np.ndarray, labeled: np.ndarray, frac: float, seed: int):
    """(train rows, validation rows): about frac of each label's labeled rows."""
    rng = np.random.default_rng(seed)
    val: List[int] = []
    for c in np.unique(y[labeled]):
        idx = np.where(labeled & (y == c))[0]
        k = int(round(len(idx) * frac))
        if k and len(idx) - k >= 1:
            rng.shuffle(idx)
            val.extend(idx[:k].tolist())
    va = np.array(sorted(val), dtype=np.int64)
    return np.setdiff1d(np.arange(len(y)), va), va


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimizes validation log loss, so a probability from
    the head means roughly what it says when self-training thresholds it."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 26):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        nll = float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def fit(ts: TrainSet, st, init_protos: np.ndarray, targets: np.ndarray, cfg: FitConfig,
        device: torch.device, log=print, tap=None) -> dict:
    """tap (live.LiveTap) only reads what the steps compute; None trains the same."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    st.to(device)
    trainable = set_trainable(st, cfg.unfreeze_layers)
    cached = cfg.unfreeze_layers == 0
    epochs = cfg.epochs or (40 if cached else 4)
    tr, va = stratified_split(ts.y, ts.labeled, cfg.val_frac, cfg.seed)

    head = PrototypeHead(torch.from_numpy(init_protos), cfg.scale).to(device)
    T = torch.from_numpy(targets).to(device)
    y = torch.from_numpy(ts.y).to(device)
    w = torch.from_numpy(ts.weight.astype(np.float32)).to(device)
    f0 = torch.from_numpy(ts.f0.astype(np.float32)).to(device)

    pooled = None
    if cached:
        t0 = time.perf_counter()
        pooled = pooled_features(st, ts.texts, device).to(device)
        log(f"  cached pooled features for {len(ts.texts):,} texts ({time.perf_counter() - t0:.0f}s)")

    named = [(n, p) for n, p in st.named_parameters() if p.requires_grad]
    enc_params = [p for n, p in named if n.startswith("0.")]
    dense_params = [p for n, p in named if not n.startswith("0.")]
    groups = [{"params": dense_params, "lr": cfg.lr_dense, "weight_decay": cfg.weight_decay},
              {"params": [head.protos], "lr": cfg.lr_protos, "weight_decay": 0.0}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay})
    opt = torch.optim.AdamW(groups)
    all_params = [p for g in groups for p in g["params"]]

    bs = max(cfg.batch_size, 256) if cached else cfg.batch_size
    total = math.ceil(len(tr) / bs) * epochs
    warm = max(1, int(total * cfg.warmup_frac))

    def lr_factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = device.type == "cuda" and not cached
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def embed_batch(rows: np.ndarray) -> torch.Tensor:
        if cached:
            return apply_suffix(st, pooled[torch.as_tensor(rows, device=device)])
        feats = features_to(st.preprocess([ts.texts[i] for i in rows]), device)
        if tap is not None and tap.armed:
            tap.feats = {k: feats[k] for k in ("input_ids", "attention_mask")}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            emb = st(feats)["sentence_embedding"]
        return F.normalize(emb.float(), dim=1)

    @torch.no_grad()
    def embed_rows(rows: np.ndarray) -> torch.Tensor:
        if cached:
            return torch.cat([apply_suffix(st, pooled[torch.as_tensor(rows[s:s + 2048], device=device)])
                              for s in range(0, len(rows), 2048)])
        return torch.from_numpy(encode(st, [ts.texts[i] for i in rows], device)).to(device)

    def val_logits() -> torch.Tensor:
        return head(embed_rows(va))

    def snapshot():
        return ({n: p.detach().cpu().clone() for n, p in named}, head.protos.detach().cpu().clone())

    def restore(snap) -> None:
        params, protos = snap
        current = dict(named)
        with torch.no_grad():
            for n, v in params.items():
                current[n].copy_(v.to(current[n].device))
            head.protos.copy_(protos.to(device))

    log(f"  training: {len(tr):,} rows ({int(ts.labeled[tr].sum()):,} labeled), {len(va):,} validation, "
        f"{len(trainable)} trainable tensors ({sum(p.numel() for p in all_params) / 1e6:.1f}M params "
        f"incl. prototypes), {epochs} epochs, batch {bs}, {'cached' if cached else 'fine-tune'} mode")
    history, best, bad = [], (-1.0, None, 0), 0
    n_steps = math.ceil(len(tr) / bs)
    if tap is not None:
        tap.start(st, head, named, ts, epochs, n_steps)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        last_report = t0
        st.train()
        perm = tr[torch.randperm(len(tr)).numpy()]
        loss_sum = 0.0
        for step, s in enumerate(range(0, len(perm), bs), 1):
            now = time.perf_counter()
            if now - last_report >= PROGRESS_SECONDS and step > 1:
                rate = (step - 1) / (now - t0)
                log(f"  epoch {epoch}/{epochs}  step {step - 1}/{n_steps} ({100 * (step - 1) / n_steps:.0f}%)  "
                    f"loss {loss_sum / s:.4f}  {rate:.1f} steps/s  epoch ETA {(n_steps - step + 1) / rate / 60:.1f} min")
                last_report = now
            rows = perm[s:s + bs]
            ridx = torch.as_tensor(rows, device=device)
            armed = tap is not None and tap.arm()
            emb = embed_batch(rows)
            logits = head(emb)
            logp = F.log_softmax(logits, dim=1)
            ce = -(T[y[ridx]] * logp).sum(dim=1)
            wb = w[ridx]
            loss = (ce * wb).sum() / wb.sum()
            ce_loss = loss
            if cfg.anchor_weight > 0:
                loss = loss + cfg.anchor_weight * (1.0 - (emb * f0[ridx]).sum(dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if armed:
                tap.before_step(emb, logits, rows, y[ridx], loss.detach(), ce_loss.detach(),
                                (loss - ce_loss).detach(), grad_norm, [g["lr"] for g in opt.param_groups])
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_value = float(loss)
            loss_sum += loss_value * len(rows)
            if armed:
                tap.after_step(epoch, step)
            if tap is not None:
                tap.step(loss_value)
        acc = float((val_logits().argmax(dim=1) == y[torch.as_tensor(va, device=device)]).float().mean()) if len(va) else 0.0
        history.append({"epoch": epoch, "loss": round(loss_sum / len(perm), 5), "val_acc": round(acc, 4),
                        "seconds": round(time.perf_counter() - t0, 1)})
        if tap is not None:
            tap.epoch(history)
        log(f"  epoch {epoch}/{epochs}: loss {history[-1]['loss']:.4f}  val acc {acc:.4f}  "
            f"({history[-1]['seconds']:.0f}s)")
        if acc > best[0]:
            best, bad = (acc, snapshot(), epoch), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  stopping early: no validation gain for {cfg.patience} epochs")
                break
    if tap is not None:
        tap.close()
    restore(best[1])
    opt.zero_grad(set_to_none=True)   # gradients and optimizer state are not needed past here
    del opt, sched, scaler
    with torch.no_grad():
        temperature = fit_temperature(val_logits().cpu().numpy(), ts.y[va]) if len(va) else 1.0
    log(f"  best epoch {best[2]} (val acc {best[0]:.4f}); temperature {temperature:.2f}")
    return {"head": head, "trainable": trainable, "temperature": temperature, "history": history,
            "best_epoch": best[2], "val_acc": best[0], "epochs": epochs, "pooled": pooled,
            "cached": cached}


@torch.no_grad()
def embed_trainset(st, ts: TrainSet, result: dict, device) -> np.ndarray:
    """Embeddings of every training row with the trained encoder."""
    if result["cached"]:
        pooled = result["pooled"]
        return torch.cat([apply_suffix(st, pooled[s:s + 2048]) for s in range(0, len(pooled), 2048)]).cpu().numpy()
    return encode(st, ts.texts, device)
