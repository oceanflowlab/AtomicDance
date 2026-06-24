"""Train and debug the atomic planner and motion completion stages."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.atomic import AtomicMotionLibrary, plan_boundaries
from dataset.atomic_dataset import AtomicSequenceDataset, collate_atomic_sequences
from model.atomic_completion import AtomicCompletionDecoder, AtomicCompletionDiffusion
from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_dataset(root, split, limit=None):
    dataset = AtomicSequenceDataset(root, split=split)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def make_loader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_atomic_sequences,
        drop_last=False,
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def planner_model(args):
    model = AtomicPlannerTransformer(
        num_atomic_classes=args.num_classes,
        music_dim=args.music_dim,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    )
    return UniformD3PM(model, num_steps=args.diffusion_steps)


def completion_model(args):
    model = AtomicCompletionDecoder(
        motion_dim=args.motion_dim,
        seq_len=args.seq_len,
        music_dim=args.music_dim,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    return AtomicCompletionDiffusion(
        model,
        num_steps=args.diffusion_steps,
        transition_weight=args.transition_weight,
        cond_drop_prob=args.cond_drop_prob,
        guidance_weight=args.guidance_weight,
    )


def build_library(dataset):
    motions = []
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        motions.append(sample["motion"])
        labels.append(sample["labels"])
    library = AtomicMotionLibrary.from_sequences(motions, labels)
    missing = sorted(set(range(1, 101)) - set(library.motions))
    if missing:
        print("Library does not contain classes: {}".format(missing))
    return library


def completion_conditions(labels, library, motion_dim, noise_ratio, device):
    drafts = []
    masks = []
    for plan in labels.cpu():
        draft, mask = library.build_draft(plan, motion_dim)
        drafts.append(draft)
        masks.append(mask * noise_ratio)
    return (
        torch.stack(drafts).to(device, non_blocking=True),
        torch.stack(masks).to(device, non_blocking=True),
        plan_boundaries(labels),
    )


@torch.no_grad()
def evaluate_planner(model, loader, device):
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    output = model.training_step(batch["labels"], batch["music"], batch["padding_mask"])
    valid = ~batch["padding_mask"]
    denoising_accuracy = (
        (output.logits.argmax(dim=-1) == output.target_labels) & valid
    ).sum().float() / valid.sum().clamp_min(1)
    sample = model.sample(batch["music"], batch["padding_mask"], deterministic=False)
    sample_correct = (sample == batch["labels"]) & valid
    nonzero = valid & (batch["labels"] != 0)
    sample_accuracy = sample_correct.sum().float() / valid.sum().clamp_min(1)
    nonzero_accuracy = sample_correct[nonzero].float().mean() if nonzero.any() else torch.tensor(0.0, device=device)
    segment_counts = ((sample[:, 1:] != sample[:, :-1]) & valid[:, 1:]).sum(dim=1) + 1
    return {
        "loss": float(output.loss),
        "denoising_accuracy": float(denoising_accuracy),
        "sample_accuracy": float(sample_accuracy),
        "sample_nonzero_accuracy": float(nonzero_accuracy),
        "sample_transition_fraction": float(((sample == 0) & valid).sum().float() / valid.sum().clamp_min(1)),
        "sample_mean_segments": float(segment_counts.float().mean()),
        "sample_shape": list(sample.shape),
    }


@torch.no_grad()
def evaluate_completion(model, loader, library, args, device):
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    draft, mask, boundaries = completion_conditions(
        batch["labels"], library, args.motion_dim, args.draft_noise_ratio, device
    )
    losses = model.training_step(batch["motion"], batch["music"], draft, mask, boundaries)
    sample = model.sample(batch["music"][:1], draft[:1], mask[:1], guidance_weight=1.0)
    return {
        "loss": float(losses.total),
        "denoising": float(losses.denoising),
        "transition": float(losses.transition),
        "sample_shape": list(sample.shape),
        "sample_finite": bool(torch.isfinite(sample).all()),
    }


def save_checkpoint(path, model, optimizer, args, step, epoch, metrics=None):
    torch.save(
        {
            "stage": args.stage,
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics or {},
        },
        str(path),
    )


def train(args):
    if args.log_every_epochs < 1 or args.save_every_epochs < 1:
        raise ValueError("epoch logging and saving intervals must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_dataset = make_dataset(args.data_root, "train", args.limit)
    test_dataset = make_dataset(args.data_root, "test", args.validation_limit)
    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True)
    test_loader = make_loader(test_dataset, args.batch_size, args.workers, False)

    if args.stage == "planner":
        model = planner_model(args).to(device)
        library = None
    else:
        model = completion_model(args).to(device)
        library = build_library(train_dataset)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        start_epoch = int(checkpoint.get("epoch", 0))

    print("device={} stage={} parameters={} samples={}".format(
        device, args.stage, sum(parameter.numel() for parameter in model.parameters()), len(train_dataset)
    ))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = start_step
    completed_epochs = start_epoch
    stop = args.max_steps is not None and step >= args.max_steps
    remaining_updates = max(0, (args.epochs - start_epoch) * len(train_loader))
    if args.max_steps is not None:
        remaining_updates = min(remaining_updates, max(0, args.max_steps - step))
    progress = tqdm(total=remaining_updates, desc="{} training".format(args.stage), unit="step")
    log_loss = 0.0
    log_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "planner":
                output = model.training_step(
                    batch["labels"], batch["music"], batch["padding_mask"]
                )
                loss = output.loss
            else:
                draft, mask, boundaries = completion_conditions(
                    batch["labels"], library, args.motion_dim, args.draft_noise_ratio, device
                )
                output = model.training_step(
                    batch["motion"], batch["music"], draft, mask, boundaries
                )
                loss = output.total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            loss_value = float(loss.detach())
            epoch_loss += loss_value
            epoch_steps += 1
            progress.update(1)
            progress.set_postfix(epoch=epoch + 1, loss="{:.4f}".format(loss_value))
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        if epoch_steps == len(train_loader):
            completed_epochs = epoch + 1
            epoch_average = epoch_loss / epoch_steps
            log_loss += epoch_average
            log_epochs += 1
            if completed_epochs % args.log_every_epochs == 0:
                progress.write(
                    "epoch={} mean_loss={:.6f}".format(
                        completed_epochs, log_loss / max(log_epochs, 1)
                    )
                )
                log_loss = 0.0
                log_epochs = 0
            if completed_epochs % args.save_every_epochs == 0:
                checkpoint_path = output_dir / "{}_epoch{}_step{}.pt".format(
                    args.stage, completed_epochs, step
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args,
                    step,
                    completed_epochs,
                    {"train_loss": epoch_average},
                )
                progress.write("checkpoint={}".format(checkpoint_path))
        if stop:
            break
    progress.close()

    metrics = (
        evaluate_planner(model, test_loader, device)
        if args.stage == "planner"
        else evaluate_completion(model, train_loader, library, args, device)
    )
    checkpoint_path = output_dir / "{}_step{}.pt".format(args.stage, step)
    save_checkpoint(
        checkpoint_path, model, optimizer, args, step, completed_epochs, metrics
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "epoch": completed_epochs,
        "step": step,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("planner", "completion"), required=True)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument("--output-dir", default="runs/atomic_debug")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--save-every-epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--motion-dim", type=int, default=151)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--cond-drop-prob", type=float, default=0.25)
    parser.add_argument("--guidance-weight", type=float, default=2.0)
    parser.add_argument("--draft-noise-ratio", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
