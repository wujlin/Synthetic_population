import json
import os
from typing import Dict, List, Sequence


def compute_saliency_and_locality(
    data_root: str,
    model_path: str,
    bands_km: Sequence[float],
) -> None:
    """Compute saliency-based locality curves for the trained CNN.

    This implementation is intentionally simple: it computes gradients
    of the output w.r.t. inputs and aggregates them radially on each grid.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PyTorch/NumPy required for nn-saliency: {exc}")

    from .nn import _iter_datasets  # reuse helper

    bands = list(sorted(bands_km))
    if not bands:
        bands = [0, 2, 4, 6, 8, 10]

    # Rebuild model architecture consistent with nn.py
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    state_cpu = torch.load(model_path, map_location="cpu")
    if "out.weight" in state_cpu:
        out_channels = state_cpu["out.weight"].shape[0]
    else:
        out_channels = 1
        print("[nn-saliency] Warning: unable to infer out_channels from checkpoint; defaulting to 1.")

    class SimpleUNet(nn.Module):
        def __init__(self, in_channels: int = 3, out_channels: int = out_channels):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.enc2 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Conv2d(32, out_channels, 1)

        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x3 = self.dec1(x2)
            if x3.shape[-2:] != x1.shape[-2:]:
                x3 = torch.nn.functional.interpolate(x3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            x = x3 + x1
            return self.out(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=3, out_channels=out_channels).to(device)
    state = {k: v.to(device) if hasattr(v, "to") else v for k, v in state_cpu.items()}
    model.load_state_dict(state)
    model.eval()

    channel_names = ["rhoH", "rhoW", "In"]
    output_names = ["drhoH", "drhoW"] if out_channels >= 2 else ["drhoH"]
    all_curves: Dict[str, Dict[str, List[List[float]]]] = {
        out: {ch: [] for ch in channel_names} for out in output_names
    }

    print(f"[nn-saliency] Loading grids from {data_root}...")
    files = list(_iter_datasets(data_root))
    if not files:
        raise RuntimeError(f"[nn-saliency] No grid files in {data_root}")
    print(f"[nn-saliency] Found {len(files)} grid files.")

    for grid_path in files:
        if grid_path.endswith(".npz"):
            arr = np.load(grid_path)
            X = arr["X"].astype("float32")
            mask = arr.get("mask")
            grid_res_km = float(arr.get("grid_res_km", 1.0))
        else:
            with open(grid_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            X = np.asarray(payload["X"], dtype=np.float32)
            mask = payload.get("mask")
            grid_res_km = float(payload.get("grid_res_km", 1.0))
        if mask is None:
            mask = np.ones((len(X[0]), len(X[0][0])), dtype=np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)

        x = torch.from_numpy(X[None, ...]).to(device)
        x.requires_grad_(True)

        y = model(x)
        C, H, W = X.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * grid_res_km

        for out_idx, out_name in enumerate(output_names):
            grads = torch.autograd.grad(
                y[:, out_idx, :, :].sum(), x, retain_graph=True
            )[0]
            grad_np = grads.detach().cpu().numpy()[0]
            for c_idx, cname in enumerate(channel_names):
                g = np.abs(grad_np[c_idx]) * mask
                curve: List[float] = []
                for b_idx in range(len(bands) - 1):
                    r_min, r_max = bands[b_idx], bands[b_idx + 1]
                    sel = (r >= r_min) & (r < r_max) & (mask > 0)
                    if sel.sum() == 0:
                        curve.append(0.0)
                    else:
                        curve.append(float(g[sel].mean()))
                all_curves[out_name][cname].append(curve)

    # Aggregate across counties
    summary: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for out_name, channel_curves in all_curves.items():
        summary[out_name] = {}
        for cname, curves in channel_curves.items():
            if not curves:
                continue
            arr = np.asarray(curves)
            mean_curve = arr.mean(axis=0).tolist()
            std_curve = arr.std(axis=0).tolist()
            summary[out_name][cname] = {
                "bands_km": bands,
                "mean": mean_curve,
                "std": std_curve,
            }

    # Non-reciprocity index
    nri = None
    if "drhoH" in summary and "dIn" in summary:
        try:
            k_H_from_In = summary["drhoH"]["In"]["mean"][0]
            k_In_from_H = summary["dIn"]["rhoH"]["mean"][0]
            denom = k_H_from_In + k_In_from_H
            if denom != 0:
                nri = (k_H_from_In - k_In_from_H) / denom
        except Exception:
            nri = None

    out_dir = os.path.join("project", "results", "node")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "saliency_locality_curves.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[nn-saliency] Saved locality curves to {out_json}")
    if nri is not None:
        nri_path = os.path.join(out_dir, "nonreciprocity_index.json")
        with open(nri_path, "w", encoding="utf-8") as f:
            json.dump({"NRI": nri}, f, indent=2)
        print(f"[nn-saliency] Non-reciprocity index saved to {nri_path}")
