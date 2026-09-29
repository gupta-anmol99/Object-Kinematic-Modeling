import argparse
import json
import torch
from torch.utils.data import DataLoader

from models.SequencePointCloudDataset import SequencePointCloudDataset
from models.PokeNet import PokeNet
from models.HungarianMatcher import HungarianJointMatcher
from models.GTAdapter import parse_gt_batch
from models.loss import JointSetLoss
from models.metrics import JointMetrics


@torch.no_grad()
def run_eval(model, loader, loss_fn, matcher, device, mu=0.5):
    model.eval()
    metrics = JointMetrics(mu=mu)
    loss_sums, n = {}, 0
    for seq, labels in loader:
        seq = seq.to(device)
        preds = model(seq)
        gt = parse_gt_batch(labels, device=device)
        assignments = matcher(preds, gt)
        _, stats = loss_fn(preds, gt, assignments)
        for k, v in stats.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v * seq.shape[0]
        n += seq.shape[0]
        metrics.update(preds, gt, assignments, labels["center"].to(device), labels["scale"].to(device))
    losses = {k: v / max(n, 1) for k, v in loss_sums.items()}
    return losses, metrics.compute()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PokeNet(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description="Evaluate a PokeNet checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--categories", nargs="*", default=None, help="e.g. unseen test categories")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mu", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--out", default=None, help="optional json file for the metrics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    data_kwargs = ckpt["data_kwargs"]
    ds = SequencePointCloudDataset(args.data_root, categories=args.categories,
                                   deterministic=True, **data_kwargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    losses, metrics = run_eval(model, loader, JointSetLoss(**ckpt["loss_kwargs"]),
                               HungarianJointMatcher(**ckpt.get("matcher_kwargs", {})), device, mu=args.mu)
    print(f"{len(ds)} sequences")
    print("loss   :", JointMetrics.format(losses))
    print("metrics:", JointMetrics.format(metrics))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"losses": losses, "metrics": metrics, "n": len(ds)}, f, indent=2)


if __name__ == "__main__":
    main()
