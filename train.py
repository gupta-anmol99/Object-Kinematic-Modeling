import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, Subset

from models.SequencePointCloudDataset import SequencePointCloudDataset
from models.PokeNet import PokeNet
from models.HungarianMatcher import HungarianJointMatcher
from models.GTAdapter import parse_gt_batch
from models.loss import JointSetLoss
from models.metrics import JointMetrics
from evaluate import run_eval


def build_datasets(args, data_kwargs):
    if args.test_categories:
        # unseen-category protocol: train on some categories, validate on held-out ones
        train_ds = SequencePointCloudDataset(args.data_root, categories=args.train_categories, **data_kwargs)
        val_ds = SequencePointCloudDataset(args.data_root, categories=args.test_categories,
                                           deterministic=True, **data_kwargs)
        return train_ds, val_ds
    full = SequencePointCloudDataset(args.data_root, categories=args.train_categories, **data_kwargs)
    full_eval = SequencePointCloudDataset(args.data_root, categories=args.train_categories,
                                          deterministic=True, **data_kwargs)
    idx = list(range(len(full)))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(round(args.val_frac * len(idx))))
    return Subset(full, idx[n_val:]), Subset(full_eval, idx[:n_val])


def main():
    ap = argparse.ArgumentParser(description="Train PokeNet")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="checkpoint/pokenet")
    ap.add_argument("--train_categories", nargs="*", default=None)
    ap.add_argument("--test_categories", nargs="*", default=None,
                    help="held-out categories used for validation (unseen-category split)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_points", type=int, default=8192)
    ap.add_argument("--sequence_length", type=int, default=12)
    ap.add_argument("--num_queries", type=int, default=6)
    ap.add_argument("--w_state", type=float, default=1.0)
    ap.add_argument("--eos_coef", type=float, default=0.5)
    ap.add_argument("--w_conf_match", type=float, default=1.0,
                    help="confidence term in the Hungarian cost (0 = paper's cost: type+axis+anchor)")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    data_kwargs = dict(num_points=args.num_points, sequence_length=args.sequence_length)
    model_kwargs = dict(num_queries=args.num_queries)
    loss_kwargs = dict(w_state=args.w_state, eos_coef=args.eos_coef)
    matcher_kwargs = dict(w_conf=args.w_conf_match)

    train_ds, val_ds = build_datasets(args, data_kwargs)
    print(f"train {len(train_ds)}  val {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = PokeNet(**model_kwargs).to(device)
    matcher = HungarianJointMatcher(**matcher_kwargs)
    loss_fn = JointSetLoss(**loss_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.wandb:
        import wandb
        wandb.init(project="pokenet", config=vars(args))

    def save(name, epoch, val_losses=None, val_metrics=None):
        torch.save({"model": model.state_dict(), "model_kwargs": model_kwargs,
                    "data_kwargs": data_kwargs, "loss_kwargs": loss_kwargs,
                    "matcher_kwargs": matcher_kwargs,
                    "epoch": epoch, "args": vars(args),
                    "val_losses": val_losses, "val_metrics": val_metrics},
                   os.path.join(args.out_dir, name))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, n = {}, 0
        for seq, labels in train_loader:
            seq = seq.to(device)
            preds = model(seq)
            gt = parse_gt_batch(labels, device=device)
            loss, stats = loss_fn(preds, gt, matcher(preds, gt))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        scheduler.step()
        train_stats = {k: v / max(n, 1) for k, v in sums.items()}
        print(f"[{epoch:03d}] train  {JointMetrics.format(train_stats)}", flush=True)
        log = {f"train/{k}": v for k, v in train_stats.items()}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_losses, val_metrics = run_eval(model, val_loader, loss_fn, matcher, device, mu=args.mu)
            print(f"[{epoch:03d}] val    {JointMetrics.format(val_losses)}")
            print(f"[{epoch:03d}] metric {JointMetrics.format(val_metrics)}", flush=True)
            log.update({f"val/{k}": v for k, v in val_losses.items()})
            log.update({f"metric/{k}": v for k, v in val_metrics.items()})
            if val_losses["total"] < best:
                best = val_losses["total"]
                save("best.pt", epoch, val_losses, val_metrics)
        save("last.pt", epoch)
        if args.wandb:
            wandb.log(log, step=epoch)


if __name__ == "__main__":
    main()
