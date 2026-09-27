"""End-to-end seq2seq data, training, evaluation, and checkpoint helpers."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Seq2SeqJsonlDataset(Dataset):
    def __init__(self, paths, tokenizer, *, max_source_length: int, max_target_length: int):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        for raw in paths:
            path = Path(raw)
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item.get("source"), str) or not isinstance(item.get("target"), str):
                        raise ValueError(f"{path}:{line_no} requires string source and target")
                    self.rows.append((item["source"], item["target"]))
        if not self.rows:
            raise ValueError("seq2seq dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, target = self.rows[index]
        src = self.tokenizer.encode(source, add_bos=True, add_eos=True)[: self.max_source_length]
        tgt = self.tokenizer.encode(target, add_bos=True, add_eos=True)[: self.max_target_length]
        return {"source_ids": src, "target_ids": tgt}


def _special_id(tokenizer, name: str, fallback: int = 0) -> int:
    value = tokenizer.special_tokens.get(name)
    return int(fallback if value is None else value)


def make_seq2seq_collate(tokenizer):
    pad_id = _special_id(tokenizer, "<|pad|>", 0)

    def collate(rows):
        b = len(rows)
        src_w = max(len(r["source_ids"]) for r in rows)
        tgt_w = max(len(r["target_ids"]) for r in rows)
        src = torch.full((b, src_w), pad_id, dtype=torch.long)
        tgt = torch.full((b, tgt_w), pad_id, dtype=torch.long)
        sm = torch.zeros((b, src_w), dtype=torch.bool)
        tm = torch.zeros((b, tgt_w), dtype=torch.bool)
        for i, row in enumerate(rows):
            s, t = row["source_ids"], row["target_ids"]
            src[i, : len(s)] = torch.tensor(s)
            tgt[i, : len(t)] = torch.tensor(t)
            sm[i, : len(s)] = True
            tm[i, : len(t)] = True
        return {"source_ids": src, "target_ids": tgt, "source_mask": sm, "target_mask": tm}
    return collate


def build_seq2seq_loader(paths, tokenizer, config, *, shuffle: bool):
    dataset = Seq2SeqJsonlDataset(
        paths,
        tokenizer,
        max_source_length=int(config.get("max_source_length", config.get("max_sequence_length", 512))),
        max_target_length=int(config.get("max_target_length", config.get("max_sequence_length", 512))),
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=make_seq2seq_collate(tokenizer),
    )


class Seq2SeqTrainer:
    def __init__(self, model, optimizer, *, scheduler=None, device="cpu", gradient_clip_norm=1.0, tracker=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.gradient_clip_norm = gradient_clip_norm
        self.tracker = tracker
        self.global_step = 0
        self.best_validation_loss = float("inf")

    def _loss(self, batch):
        src = batch["source_ids"].to(self.device)
        tgt = batch["target_ids"].to(self.device)
        sm = batch["source_mask"].to(self.device)
        tm = batch["target_mask"].to(self.device)
        if tgt.shape[1] < 2:
            raise ValueError("target sequence must contain at least two tokens")
        decoder_in = tgt[:, :-1]
        labels = tgt[:, 1:]
        decoder_mask = tm[:, :-1]
        label_mask = tm[:, 1:]
        logits = self.model(src, decoder_in, source_mask=sm, target_mask=decoder_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
        denom = label_mask.sum().clamp_min(1)
        return (loss.view_as(labels) * label_mask).sum() / denom, int(label_mask.sum())

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, tokens = self._loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite seq2seq loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        metrics = {"train/loss": float(loss.detach()), "train/tokens": float(tokens), "train/grad_norm": float(grad)}
        if self.tracker is not None:
            self.tracker.log(metrics, step=self.global_step)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total, tokens = 0.0, 0
        for batch in loader:
            loss, n = self._loss(batch)
            total += float(loss) * n
            tokens += n
        if not tokens:
            raise ValueError("seq2seq validation loader is empty")
        result = {"loss": total / tokens, "tokens": tokens}
        if self.tracker is not None:
            self.tracker.log({"validation/loss": result["loss"]}, step=self.global_step)
        return result

    def fit(self, train_loader, *, epochs: int, validation_loader=None, checkpoint_callback=None, best_checkpoint_callback=None):
        history = []
        for epoch in range(int(epochs)):
            for batch in train_loader:
                self.train_step(batch)
            row = {"epoch": epoch + 1, "step": self.global_step}
            if validation_loader is not None:
                metrics = self.evaluate(validation_loader)
                row.update({f"validation_{k}": v for k, v in metrics.items()})
                if metrics["loss"] < self.best_validation_loss:
                    self.best_validation_loss = metrics["loss"]
                    if best_checkpoint_callback:
                        best_checkpoint_callback(self, epoch)
            if checkpoint_callback:
                checkpoint_callback(self, epoch)
            history.append(row)
        return history


def save_seq2seq_checkpoint(path, model, optimizer=None, *, step=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "step": int(step), "metadata": metadata or {}}, path)


def load_seq2seq_checkpoint(path, model, optimizer=None, *, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
