"""
DeepKriging on Synthetic IRF_k Gaussian fields on the sphere

This script:
  1) Generates a synthetic Intrinsic Random Function of order k (IRF_k) on S^2
     using a truncated spherical-harmonic spectral expansion with coefficients
     c_l = r^l for l >= k (i.e., covariance C(h) = sum_{l=k}^inf (2l+1)/(4*pi) r^l P_l(cos h)).
  2) Splits locations into train/test (default 90/10).
  3) Builds DeepKriging features: trig(positional) + RBFs on great-circle distance to M centers.
  4) Trains an MLP (PyTorch) to predict field values at new locations.
  5) Reports RMSE on the test split. Optionally runs multiple seeds and averages.

Notes
-----
- Spherical harmonics are computed with scipy.special.sph_harm (complex-valued, orthonormal); we
  sample complex coefficients with conjugate symmetry to obtain a real-valued field.
- The spectral truncation at l = L_max controls smoothness/detail in the simulated field.
- This is a *sphere-aware* setup: all distances are geodesic (great-circle). RBF features are
  exp(-(d/ell)^2), where d is central angle in radians and ell is a global length-scale.

Usage
-----
python deepkriging_irf_synth.py \
  --n_points 1500 --kappa 2 --r 0.75 --lmax 64 \
  --rbf_centers 128 --epochs 200 --patience 20 --hidden 256 256 256 \
  --seeds 1 --device auto

Dependencies
------------
- numpy, scipy, scikit-learn, torch, tqdm (optional)

"""
from __future__ import annotations
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy.random import default_rng
from scipy import special as sp
# Prefer new SciPy API to avoid deprecation warnings (SciPy>=1.15)
sph_harm_fn = getattr(sp, 'sph_harm_y', None) or sp.sph_harm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import csv
from datetime import datetime
from scipy import stats as st

try:
    from tqdm import trange
except Exception:
    # fallback if tqdm not installed
    def trange(n, **kwargs):
        return range(n)

# ----------------------------
# Geometry utilities (sphere)
# ----------------------------

def sample_uniform_sphere(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample n points on the sphere. Returns lat, lon in radians.
    lat in [-pi/2, pi/2], lon in [0, 2pi).
    """
    u = rng.uniform(-1.0, 1.0, size=n)  # sin(lat)
    lat = np.arcsin(u)
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return lat, lon


def sph_to_cart(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) -> Cartesian xyz (unit sphere)."""
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def colatitude(lat: np.ndarray) -> np.ndarray:
    """Convert latitude (radians) to colatitude theta in [0, pi]."""
    return np.pi / 2.0 - lat


def great_circle_angle(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Central angle (radians) via haversine (vectorized/broadcastable)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


# ----------------------------------------
# Spectral simulation of IRF_k on the sphere
# ----------------------------------------

def simulate_irf_spectral(
    lat: np.ndarray,
    lon: np.ndarray,
    kappa: int = 2,
    r: float = 0.75,
    lmax: int = 64,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a real-valued Gaussian random field on S^2 that is IRF_k with
    truncated spectral coefficients c_l = r^l for l >= kappa, 0 otherwise.

    f(x) = sum_{l=kappa}^{lmax} sum_{m=-l}^l a_{l m} Y_{l m}(x)
    with E[a_{lm}] = 0, Var(a_{lm}) = c_l = r^l, coefficients chosen to yield a real field.

    Returns array f of shape (n,).
    """
    assert 0.0 < r < 1.0, "r must be in (0,1)"
    assert lmax >= kappa >= 0
    rng = default_rng(seed)

    n = lat.shape[0]
    theta = colatitude(lat)  # [0, pi]
    phi = lon % (2.0 * np.pi)  # [0, 2pi)

    f = np.zeros(n, dtype=np.float64)

    for l in range(kappa, lmax + 1):
        c_l = r ** l  # spectral variance at degree l
        # m = 0 term (real-valued)
        a_l0 = rng.normal(0.0, np.sqrt(c_l))
        Y_l0 = sph_harm_fn(0, l, phi, theta).real  # (n,)
        f += a_l0 * Y_l0
        # m >= 1 terms (complex harmonics; enforce real field via conjugate symmetry)
        if l > 0:
            for m in range(1, l + 1):
                # Sample complex coefficient: real & imag ~ N(0, c_l/2)
                re = rng.normal(0.0, np.sqrt(c_l / 2.0))
                im = rng.normal(0.0, np.sqrt(c_l / 2.0))
                a_lm = re + 1j * im
                Y_lm = sph_harm_fn(m, l, phi, theta)  # complex (n,)
                # Contribution of m and -m together equals 2 * Re[a_lm * Y_lm]
                f += 2.0 * np.real(a_lm * Y_lm)

    # Optional: rescale to unit variance across points (helps training stability)
    f = (f - f.mean()) / (f.std() + 1e-9)
    return f


# ----------------------------
# DeepKriging feature builder
# ----------------------------

def build_rbf_centers(lat: np.ndarray, lon: np.ndarray, n_centers: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans on (x,y,z) to get approximately uniform centers; returns lat_c, lon_c in radians."""
    xyz = sph_to_cart(lat, lon)
    km = KMeans(n_clusters=n_centers, n_init=10, random_state=seed)
    km.fit(xyz)
    centers = km.cluster_centers_
    # Project centers back to unit sphere
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    lat_c = np.arcsin(z)
    lon_c = (np.arctan2(y, x) + 2.0 * np.pi) % (2.0 * np.pi)
    return lat_c, lon_c


def rbf_features(lat: np.ndarray, lon: np.ndarray, lat_c: np.ndarray, lon_c: np.ndarray, ell: float) -> np.ndarray:
    """Compute exp(-(d/ell)^2) between points and centers, where d is great-circle angle (radians)."""
    # Broadcast to (n, m)
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat_c[None, :]
    lon2 = lon_c[None, :]
    d = great_circle_angle(lat1, lon1, lat2, lon2)  # (n, m)
    return np.exp(- (d / ell) ** 2)


def trig_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Basic positional encodings on the sphere (periodic in lon)."""
    return np.stack([
        np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    ], axis=1)


# ---------------
# Torch dataset
# ---------------
class SpatialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------
# DeepKriging MLP
# ----------------
class DeepKrigingMLP(nn.Module):
    def __init__(self, d_in: int, hidden: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------
# Train / Eval
# -------------
@dataclass
class Config:
    n_points: int = 1500
    kappa: int = 2
    r: float = 0.75
    lmax: int = 64

    test_frac: float = 0.1
    rbf_centers: int = 128
    rbf_ell: float | None = None  # if None, auto-tune from data

    hidden: Tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1
    lr: float = 2e-3
    batch_size: int = 512
    epochs: int = 200
    patience: int = 20

    seeds: int = 1
    base_seed: int = 123
    device: str = "auto"  # "cpu" | "cuda" | "auto"

    csv: str | None = None
    run_name: str | None = None


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_std_ci(vals: List[float], alpha: float = 0.05):
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    mean = float(np.mean(vals)) if n > 0 else float('nan')
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / math.sqrt(n)
        tcrit = float(st.t.ppf(1 - alpha/2.0, df=n-1))
        half = tcrit * se
        return mean, std, (mean - half, mean + half)
    else:
        return mean, std, (float('nan'), float('nan'))


def append_csv(path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerows(rows)



def choose_device(pref: str) -> torch.device:
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_seed(cfg: Config, seed: int) -> Tuple[float, float]:
    # Reproducibility
    rng = default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Sample locations and simulate field
    lat, lon = sample_uniform_sphere(cfg.n_points, rng)
    y = simulate_irf_spectral(lat, lon, kappa=cfg.kappa, r=cfg.r, lmax=cfg.lmax, seed=seed + 77)

    # 2) Train/test split
    idx = np.arange(cfg.n_points)
    rng.shuffle(idx)
    n_test = int(round(cfg.test_frac * cfg.n_points))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    lat_tr, lon_tr, y_tr = lat[train_idx], lon[train_idx], y[train_idx]
    lat_te, lon_te, y_te = lat[test_idx], lon[test_idx], y[test_idx]

    # 3) RBF centers & features
    lat_c, lon_c = build_rbf_centers(lat_tr, lon_tr, cfg.rbf_centers, seed=seed)

    # Characteristic length ell: median nearest-center distance on train if not provided
    if cfg.rbf_ell is None:
        d_train = great_circle_angle(lat_tr[:, None], lon_tr[:, None], lat_c[None, :], lon_c[None, :])
        nearest = np.partition(d_train, kth=0, axis=1)[:, 0]  # min per row
        ell = float(np.median(nearest) + 1e-6)
    else:
        ell = float(cfg.rbf_ell)

    Phi_tr = rbf_features(lat_tr, lon_tr, lat_c, lon_c, ell)
    Phi_te = rbf_features(lat_te, lon_te, lat_c, lon_c, ell)

    Trig_tr = trig_features(lat_tr, lon_tr)
    Trig_te = trig_features(lat_te, lon_te)

    X_tr = np.concatenate([Trig_tr, Phi_tr], axis=1)
    X_te = np.concatenate([Trig_te, Phi_te], axis=1)

    # 4) Standardize target using training stats only
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-9)
    y_tr_n = (y_tr - y_mean) / y_std

    # 5) Torch setup
    dev = choose_device(cfg.device)

    train_ds = SpatialDataset(X_tr.astype(np.float32), y_tr_n.astype(np.float32))
    test_ds = SpatialDataset(X_te.astype(np.float32), ((y_te - y_mean) / y_std).astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = DeepKrigingMLP(d_in=X_tr.shape[1], hidden=list(cfg.hidden), dropout=cfg.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_rmse = float("inf")
    best_state = None
    no_imp = 0

    # 6) Training loop with early stopping
    for epoch in trange(cfg.epochs, desc="training", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Evaluate RMSE on test (in original scale)
        model.eval()
        with torch.no_grad():
            preds = []
            ys = []
            for xb, yb in test_loader:
                xb = xb.to(dev)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                ys.append(yb.cpu().numpy())
        yhat_n = np.concatenate(preds, axis=0).squeeze(-1)
        y_true_n = np.concatenate(ys, axis=0).squeeze(-1)
        # back to original scale
        yhat = yhat_n * y_std + y_mean
        ytrue = y_true_n * y_std + y_mean
        cur_rmse = rmse(yhat, ytrue)

        if cur_rmse + 1e-8 < best_rmse:
            best_rmse = cur_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= cfg.patience:
            break

    # Load best and do final eval
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        ys = []
        for xb, yb in test_loader:
            xb = xb.to(dev)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            ys.append(yb.cpu().numpy())
    yhat_n = np.concatenate(preds, axis=0).squeeze(-1)
    y_true_n = np.concatenate(ys, axis=0).squeeze(-1)
    yhat = yhat_n * y_std + y_mean
    ytrue = y_true_n * y_std + y_mean
    final_rmse = rmse(yhat, ytrue)

    return final_rmse, ell


def main():
    p = argparse.ArgumentParser(description="DeepKriging on Synthetic IRF_k on the sphere")
    p.add_argument("--n_points", type=int, default=1500)
    p.add_argument("--kappa", type=int, default=2)
    p.add_argument("--r", type=float, default=0.75)
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--test_frac", type=float, default=0.10)

    p.add_argument("--rbf_centers", type=int, default=128)
    p.add_argument("--rbf_ell", type=float, default=None, help="RBF lengthscale in radians; if omitted, auto")

    p.add_argument("--hidden", type=int, nargs="*", default=[256, 256, 256])
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--seeds", type=int, default=1, help="number of replicates (different seeds)")
    p.add_argument("--base_seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--csv", type=str, default=None, help="Path to CSV to append results (per-seed and summary).")
    p.add_argument("--run_name", type=str, default=None, help="Optional tag/name for this run (stored in CSV).")

    args = p.parse_args()

    cfg = Config(
        n_points=args.n_points,
        kappa=args.kappa,
        r=args.r,
        lmax=args.lmax,
        test_frac=args.test_frac,
        rbf_centers=args.rbf_centers,
        rbf_ell=args.rbf_ell,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seeds=args.seeds,
        base_seed=args.base_seed,
        device=args.device,
        csv=args.csv,
        run_name=args.run_name,
    )

    rmses = []
    ells = []
    for i in range(cfg.seeds):
        seed = cfg.base_seed + i
        rmse_i, ell_i = run_one_seed(cfg, seed)
        print(f"Seed {seed} -> RMSE: {rmse_i:.4f} | ell={ell_i:.6f} rad")
        rmses.append(rmse_i)
        ells.append(ell_i)

    mean_rmse, std_rmse, (ci_lo, ci_hi) = mean_std_ci(rmses)
    print("=" * 60)
    print(f"DeepKriging on IRF_{cfg.kappa} synthetic | r={cfg.r:.2f} | Lmax={cfg.lmax}")
    ell_desc = "auto" if cfg.rbf_ell is None else f"{cfg.rbf_ell:.4f}"
    ell_median = float(np.median(ells)) if len(ells) > 0 else float('nan')
    print(f"RBF centers={cfg.rbf_centers} | ell={ell_desc} (median used={ell_median:.6f} rad) | hidden={cfg.hidden}")
    if cfg.seeds > 1:
        print(f"Test RMSE over {cfg.seeds} seed(s): {mean_rmse:.4f} ± {std_rmse:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    else:
        print(f"Test RMSE over {cfg.seeds} seed(s): {mean_rmse:.4f} ± {std_rmse:.4f} (95% CI n/a)")

    # CSV logging
    if cfg.csv:
        ts = datetime.now().isoformat(timespec='seconds')
        run_id = cfg.run_name or ts
        fieldnames = [
            'timestamp','run_id','record_type','seed','rmse','rmse_std','ci_low','ci_high','ell',
            'n_points','kappa','r','lmax','test_frac','rbf_centers','rbf_ell','hidden','dropout','lr','batch_size','epochs','patience','device'
        ]
        rows = []
        for s, r_i, e_i in zip(range(cfg.base_seed, cfg.base_seed + cfg.seeds), rmses, ells):
            rows.append({
                'timestamp': ts,
                'run_id': run_id,
                'record_type': 'seed',
                'seed': s,
                'rmse': f"{r_i:.6f}",
                'rmse_std': '', 'ci_low': '', 'ci_high': '',
                'ell': f"{e_i:.8f}",
                'n_points': cfg.n_points,
                'kappa': cfg.kappa,
                'r': cfg.r,
                'lmax': cfg.lmax,
                'test_frac': cfg.test_frac,
                'rbf_centers': cfg.rbf_centers,
                'rbf_ell': ell_desc,
                'hidden': 'x'.join(map(str, cfg.hidden)),
                'dropout': cfg.dropout,
                'lr': cfg.lr,
                'batch_size': cfg.batch_size,
                'epochs': cfg.epochs,
                'patience': cfg.patience,
                'device': cfg.device,
            })
        rows.append({
            'timestamp': ts,
            'run_id': run_id,
            'record_type': 'summary',
            'seed': '',
            'rmse': f"{mean_rmse:.6f}",
            'rmse_std': f"{std_rmse:.6f}",
            'ci_low': f"{ci_lo:.6f}" if cfg.seeds > 1 else '',
            'ci_high': f"{ci_hi:.6f}" if cfg.seeds > 1 else '',
            'ell': f"{ell_median:.8f}",
            'n_points': cfg.n_points,
            'kappa': cfg.kappa,
            'r': cfg.r,
            'lmax': cfg.lmax,
            'test_frac': cfg.test_frac,
            'rbf_centers': cfg.rbf_centers,
            'rbf_ell': ell_desc,
            'hidden': 'x'.join(map(str, cfg.hidden)),
            'dropout': cfg.dropout,
            'lr': cfg.lr,
            'batch_size': cfg.batch_size,
            'epochs': cfg.epochs,
            'patience': cfg.patience,
            'device': cfg.device,
        })
        append_csv(cfg.csv, rows, fieldnames)
        print(f"Logged results to {cfg.csv}")


if __name__ == "__main__":
    main()
