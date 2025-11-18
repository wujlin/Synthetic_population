import json
import os
from typing import Iterable, List, Sequence

import numpy as np


def _iter_datasets(data_root: str) -> Iterable[str]:
    for fname in os.listdir(data_root):
        if fname.endswith(".npz") or fname.endswith(".json"):
            yield os.path.join(data_root, fname)


def train_cnn(
    data_root: str,
    inputs: Sequence[str],
    target: str,
    split: str = "leave-county-out",
    epochs: int = 200,
) -> None:
    """Train a lightweight CNN/U-Net for self-prediction of drhoH.

    This is implemented as a PyTorch training loop but kept minimal.
    Dependencies (torch, numpy) are imported lazily so the module
    can be imported without them present.
    """
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for nn-train. Install on this interpreter with e.g.\n"
            "  python3 -m pip install --upgrade pip\n"
            "  python3 -m pip install torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu\n"
            f"Original import error: {exc}"
        )

    class SimpleUNet(nn.Module):
        def __init__(self, in_channels: int = 3, out_channels: int = 1):
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
            # simple skip connection (crop if needed)
            if x3.shape[-2:] != x1.shape[-2:]:
                x3 = torch.nn.functional.interpolate(x3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            x = x3 + x1
            return self.out(x)

    print(f"[nn-train] Loading datasets from {data_root}...")
    paths = list(_iter_datasets(data_root))
    if not paths:
        raise RuntimeError(f"[nn-train] No grid files found under {data_root}")
    print(f"[nn-train] Found {len(paths)} grid files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=3, out_channels=3).to(device)

    data = []
    for p in paths:
        if p.endswith(".npz"):
            arr = np.load(p)
            X = arr["X"].astype("float32")
            Y = arr["Y"].astype("float32")
            data.append((X, Y))
        else:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            X = np.asarray(payload["X"], dtype=np.float32)
            Y = np.asarray(payload["Y"], dtype=np.float32)
            data.append((X, Y))

    # Simple split: last county as test, others as train
    train_data = data[:-1]
    test_data = data[-1:]

    def _shape(arr):
        if hasattr(arr, "shape"):
            return int(arr.shape[1]), int(arr.shape[2])
        return len(arr[0]), len(arr[0][0])

    def _pad_tensor(arr, channels, max_h, max_w):
        tensor = torch.zeros((channels, max_h, max_w), dtype=torch.float32)
        src = torch.as_tensor(arr, dtype=torch.float32)
        h = min(src.shape[1], max_h)
        w = min(src.shape[2], max_w)
        tensor[:, :h, :w] = src[:, :h, :w]
        return tensor

    max_h = 0
    max_w = 0
    shapes = []
    for X, Y in data:
        h, w = _shape(X)
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        shapes.append((h, w))

    def _to_batch(ds, shapes_list):
        Xs = []
        Ys = []
        for (X, Y), (h, w) in zip(ds, shapes_list):
            Xs.append(_pad_tensor(X, len(X), max_h, max_w))
            Ys.append(_pad_tensor(Y, len(Y), max_h, max_w))
        return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0)

    X_train, Y_train = _to_batch(train_data, shapes[:-1])
    X_test, Y_test = _to_batch(test_data, shapes[-1:])
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print(f"[nn-train] Training on {len(train_data)} counties, testing on {len(test_data)}.")
    model.train()
    for epoch in range(max(1, int(epochs))):
        optim.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optim.step()

    target_names = ["drhoH", "drhoW", "dIn"]
    model.eval()
    R2 = {}
    with torch.no_grad():
        pred_test = model(X_test)
        for idx, name in enumerate(target_names):
            y_true = Y_test[:, idx : idx + 1].reshape(-1)
            y_pred = pred_test[:, idx : idx + 1].reshape(-1)
            y_mean = y_true.mean()
            ss_tot = ((y_true - y_mean) ** 2).sum()
            ss_res = ((y_true - y_pred) ** 2).sum()
            if ss_tot.item() > 0:
                R2[name] = float(1.0 - ss_res.item() / ss_tot.item())
            else:
                R2[name] = float("nan")

    metrics = {
        "R2_oos": R2,
        "epochs": int(epochs),
        "n_train_counties": len(train_data),
        "n_test_counties": len(test_data),
    }
    results_dir = os.path.join("project", "results", "node")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "nn_selfpred_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    model_path = os.path.join("project", "results", "node", "model_cnn.pt")
    torch.save(model.state_dict(), model_path)
    config = {"in_channels": 3, "out_channels": model.out.out_channels}
    with open(os.path.join(results_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[nn-train] Completed. Metrics -> {metrics_path}, model -> {model_path}")
