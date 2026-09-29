"""
Run a trained PokeNet on one demonstration (a folder of point_cloud_*.pcd frames, no labels
needed) and print / save the predicted kinematic model in camera coordinates.

    python infer.py --ckpt checkpoint/run1/best.pt --sequence path/to/demo_folder [--out joints.json]
"""
import argparse
import json
import math
import torch

from models.SequencePointCloudDataset import load_sequence
from evaluate import load_model


def main():
    ap = argparse.ArgumentParser(description="PokeNet inference on one point-cloud sequence")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sequence", required=True, help="folder with point_cloud_*.pcd frames")
    ap.add_argument("--mu", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--out", default=None, help="optional json output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    model.eval()
    dk = ckpt["data_kwargs"]

    seq, center, scale = load_sequence(args.sequence, num_points=dk["num_points"], seed=0)
    joints = model.predict(seq.unsqueeze(0).to(device),
                           torch.from_numpy(center).unsqueeze(0).to(device),
                           torch.tensor([scale], device=device), mu=args.mu)[0]

    result = []
    for i, j in enumerate(joints):
        unit = "deg" if j["type"] == "revolute" else "m"
        state = [math.degrees(v) for v in j["state"].tolist()] if unit == "deg" else j["state"].tolist()
        result.append({
            "order": i,
            "type": j["type"],
            "confidence": round(j["confidence"], 4),
            "axis": [round(v, 4) for v in j["axis"].tolist()],
            "anchor_m": [round(v, 4) for v in j["anchor"].tolist()],
            f"state_per_frame_{unit}": [round(v, 4) for v in state],
        })
        print(f"joint {i}: {j['type']:9s} conf {j['confidence']:.2f}  axis {result[-1]['axis']}  "
              f"anchor {result[-1]['anchor_m']} m  final state {state[-1]:.3f} {unit}")
    if not result:
        print(f"no joint above confidence {args.mu}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"sequence": args.sequence, "joints": result}, f, indent=2)


if __name__ == "__main__":
    main()
