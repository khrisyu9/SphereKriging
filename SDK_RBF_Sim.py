"""
Spherical DeepKriging with Great-Circle RBFs (Option B) [ASCII version]
Method (short): SphDK-GC-RBF

This script is the geometry-aware DeepKriging variant using compactly
supported Wendland basis functions with great-circle (haversine)
distances on the sphere. Multi-resolution centers are placed with a
Fibonacci sphere layout. This is the ASCII-only version (no unicode
symbols) and includes automatic theta selection and coverage checks.

Pipeline
  1) Sample N points uniformly on the sphere -> (lat_deg, lon_deg)
  2) Simulate a smooth isotropic spherical field via spherical-harmonic
     expansion with a Matern-like angular power spectrum
  3) Build spherical multi-resolution RBF features using Wendland C^4
     with great-circle distances
  4) Train a 3-layer MLP (50-50-50 -> 1) on [X | phi]
  5) Report RMSE/MAE and save a PNG to ./Results with key knobs in name

Run:  python SDK_RBF_Sim.py
Deps: numpy, matplotlib, scipy, scikit-learn, tensorflow/keras
"""
from __future__ import annotations
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Sequence

from scipy.special import sph_harm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)

# ---------------------------------
# Spherical geometry utilities
# ---------------------------------

def uniform_points_on_sphere(n: int) -> np.ndarray:
    """Draw n points uniformly on the sphere.
    Returns array of shape (n, 2): (lat_deg, lon_deg).
    """
    u = np.random.uniform(-1.0, 1.0, size=n)
    lon = np.random.uniform(-180.0, 180.0, size=n)
    lat = np.degrees(np.arcsin(u))
    return np.column_stack([lat, lon])


def latlon_to_colat_lon(coords_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    theta = np.pi / 2.0 - lat  # colatitude in [0, pi]
    phi = lon                  # longitude in [-pi, pi]
    return theta, phi


def haversine_gc_distance_matrix(A_deg: np.ndarray, B_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Great-circle (haversine) distances between two lat/lon sets.
    A_deg: (N,2), B_deg: (M,2) in degrees.
    Returns D in shape (N,M) in units of `radius` (1.0 -> radians; 6371 -> km).
    """
    A_lat = np.radians(A_deg[:, 0])[:, None]
    A_lon = np.radians(A_deg[:, 1])[:, None]
    B_lat = np.radians(B_deg[:, 0])[None, :]
    B_lon = np.radians(B_deg[:, 1])[None, :]
    dlat = B_lat - A_lat
    dlon = B_lon - A_lon
    h = np.sin(dlat/2.0)**2 + np.cos(A_lat) * np.cos(B_lat) * np.sin(dlon/2.0)**2
    ang = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))  # radians
    return radius * ang

# --- NEW: chordal distance utils and metric switch ---

def latlon_to_xyz(coords_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.column_stack([x, y, z]).astype(np.float32)

def chordal_distance_matrix(A_deg: np.ndarray, B_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    A = latlon_to_xyz(A_deg, radius=radius)
    B = latlon_to_xyz(B_deg, radius=radius)
    # Euclidean distance in R^3 (chord length)
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return D.astype(np.float32)

# ---------------------------------
# Wendland C^4 (compact support)
# ---------------------------------

def wendland_c4(r: np.ndarray) -> np.ndarray:
    """Wendland C^4 radial basis on [0,1], element-wise for arbitrary shape.
    phi(r) = (1 - r)^6 * (35*r^2 + 18*r + 3)/3 for 0 <= r <= 1, else 0.
    """
    out = np.zeros_like(r)
    m = (r >= 0) & (r <= 1)
    rm = r[m]
    out[m] = (1.0 - rm)**6 * (35.0*rm**2 + 18.0*rm + 3.0) / 3.0
    return out

# ---------------------------------
# Multi-resolution spherical centers
# ---------------------------------

def fibonacci_sphere_points(n: int, seed: int = 0) -> np.ndarray:
    """Nearly uniform points on S^2 via the golden-angle spiral.
    Returns (n,2) of (lat_deg, lon_deg).
    """
    rng = np.random.default_rng(seed)
    ga = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n)
    z = 1 - 2*(i + 0.5)/n
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    theta = ga * i
    x, y = r*np.cos(theta), r*np.sin(theta)
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x*x + y*y)))
    lon = np.rad2deg(np.arctan2(y, x))
    return np.stack([lat, lon], axis=1)


def multires_spherical_centers(level_sizes: Sequence[int], seed: int = 0) -> np.ndarray:
    centers = []
    for k, n in enumerate(level_sizes):
        centers.append(fibonacci_sphere_points(n, seed=seed + k))
    return np.vstack(centers)

# ---------------------------------
# Build spherical DK features (Option B)
# ---------------------------------

def build_phi_wendland_sphere(
    coords_deg: np.ndarray,
    level_sizes: Sequence[int] = (64, 256),
    thetas: Sequence[float] | None = None,
    radius: float = 1.0,
    seed: int = 0,
    auto_theta: bool = True,
    theta_mult: float = 1.6,
    min_active: int = 8,
    max_auto_iters: int = 4,
    metric: str = "geodesic",   # "geodesic" (original) or "chordal" (PD-safe)
    return_per_level_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Construct spherical multi-resolution RBF features.

    Returns:
        phi: (N,K)
        centers: (K,2) degrees
        theta_vec: (K,)
        stats: dict with coverage diagnostics (per-level activation, etc.)
    """
    assert metric in ("geodesic", "chordal")
    centers = multires_spherical_centers(level_sizes, seed=seed)
    N = coords_deg.shape[0]
    level_ids = np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(level_sizes)])

    # pick pairwise distance function
    if metric == "geodesic":
        dist_fn = lambda A, B: haversine_gc_distance_matrix(A, B, radius=radius)
    else:
        dist_fn = lambda A, B: chordal_distance_matrix(A, B, radius=radius)

    # per-level theta
    if thetas is not None:
        assert len(thetas) == len(level_sizes)
        theta_levels = np.asarray(list(thetas), dtype=float)
    else:
        theta_levels = np.zeros(len(level_sizes), dtype=float)
        off = 0
        for i, n in enumerate(level_sizes):
            C_lvl = centers[off:off+n]; off += n
            if n == 1:
                # cover entire sphere
                theta_levels[i] = (math.pi * 1.05 * radius) if metric == "geodesic" else (2.0 * radius * 1.05)
            else:
                Dcc = dist_fn(C_lvl, C_lvl)
                Dcc = Dcc + np.eye(n) * 1e9
                nn = Dcc.min(axis=1)
                med_nn = float(np.median(nn))
                theta_levels[i] = theta_mult * med_nn

    theta_vec = theta_levels[level_ids]

    # distances data->centers
    D = dist_fn(coords_deg, centers)

    # coverage inflation
    inflate = 0
    active = (D <= theta_vec[None, :]).sum(axis=1)
    while active.min() < min_active and inflate < max_auto_iters:
        theta_vec *= 1.2
        active = (D <= theta_vec[None, :]).sum(axis=1)
        inflate += 1

    # features
    R = D / theta_vec[None, :]
    phi = wendland_c4(R).astype(np.float32)

    # optional per-level stats
    stats = {}
    if return_per_level_stats:
        per_level = []
        for li, n in enumerate(level_sizes):
            mask = (level_ids == li)
            act_l = (D[:, mask] <= theta_vec[mask][None, :]).sum(axis=1)
            per_level.append({
                "level": li,
                "n_centers": int(n),
                "theta": float(theta_levels[li]),
                "active_min": int(act_l.min()),
                "active_med": float(np.median(act_l)),
                "active_mean": float(act_l.mean()),
            })
        stats["per_level"] = per_level
        stats["active_overall_min"] = int(active.min())
        stats["inflations"] = int(inflate)
        stats["metric"] = metric

    return phi, centers, theta_vec, stats


# -------------------------
# DeepKriging MLP head
# -------------------------

def make_deepkriging_head(input_dim: int, width: int = 50, depth: int = 3, lr: float = 1e-3, l2w: float = 1e-5) -> Sequential:
    model = Sequential([tf.keras.Input(shape=(input_dim,))])
    for _ in range(depth):
        model.add(Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2w)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mse', 'mae'])
    return model

# --------------------------------------
# Spherical field simulation (for truth)
# --------------------------------------

def real_sph_harm_features(coords_deg: np.ndarray, L_max: int):
    theta, phi = latlon_to_colat_lon(coords_deg)
    feats = []
    degrees = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            if m < 0:
                feat = np.sqrt(2.0) * Y_lm.imag
            elif m == 0:
                feat = Y_lm.real
            else:
                feat = np.sqrt(2.0) * Y_lm.real
            feats.append(np.asarray(feat, dtype=float))
            degrees.append(l)
    F = np.stack(feats, axis=1)
    return F, degrees


def simulate_spherical_field(
    coords_deg: np.ndarray,
    L_max: int = 8,
    alpha: float = 0.02,
    nu: float = 1.5,
    sigma_f: float = 1.0,
    sigma_n: float = 0.1,
    include_intercept: bool = True,
):
    """y = f_sphere(coords) + noise using SH with Matern-like spectrum.
       C_l = sigma_f^2 * (1 + alpha * l*(l+1))^(-nu)
    Returns: y (N,), X (N,1) intercept, F_sh (N,D) for reference.
    """
    F_sh, degs = real_sph_harm_features(coords_deg, L_max)
    var_cols = np.array([sigma_f**2 * (1.0 + alpha * l * (l + 1))**(-nu) for l in degs])
    w = np.random.normal(0.0, np.sqrt(var_cols), size=F_sh.shape[1])
    f = F_sh @ w
    noise = np.random.normal(0.0, sigma_n, size=f.shape[0])
    y = (f + noise).astype(np.float32)
    X = np.ones((len(y), 1), dtype=np.float32) if include_intercept else None
    return y, X, F_sh.astype(np.float32)

# -------------------------
# Config & experiment
# -------------------------
@dataclass
class Config:
    method_tag: str = "SphDK_GCRBF"
    N: int = 4000
    test_size: float = 0.2
    # Simulation hyper-params
    L_max: int = 8
    alpha: float = 0.02
    nu: float = 1.5
    sigma_f: float = 1.0
    sigma_n: float = 0.1
    # Feature construction (Option B)
    level_sizes: Sequence[int] = (64, 256)
    thetas: Sequence[float] | None = None   # in same units as radius (default radians)
    radius: float = 1.0                     # 1.0 -> radians; 6371 -> km
    seed: int = 123
    auto_theta: bool = True
    theta_mult: float = 1.6
    min_active: int = 8
    max_auto_iters: int = 4
    # Training
    epochs: int = 60
    batch_size: int = 64
    # Output
    results_dir: str = "./Plots"
    save_plots: bool = True
    dpi: int = 300


def main(cfg: Config):
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    t0 = time.time()
    # 1) Sample spherical locations
    coords = uniform_points_on_sphere(cfg.N)

    # 2) Simulate spherical field
    y, X, _ = simulate_spherical_field(
        coords,
        L_max=cfg.L_max,
        alpha=cfg.alpha,
        nu=cfg.nu,
        sigma_f=cfg.sigma_f,
        sigma_n=cfg.sigma_n,
        include_intercept=True,
    )

    # 3) Build spherical DK features (great-circle + Wendland)
    phi_sph, centers, theta_vec, stats = build_phi_wendland_sphere(
        coords,
        level_sizes=cfg.level_sizes,
        thetas=cfg.thetas,
        radius=cfg.radius,
        seed=cfg.seed,
        auto_theta=cfg.auto_theta,
        theta_mult=cfg.theta_mult,
        min_active=cfg.min_active,
        max_auto_iters=cfg.max_auto_iters,
        metric="chordal",  # <--- PD-safe
        return_per_level_stats=True  # diagnostics
    )

    F = phi_sph if X is None else np.hstack([X, phi_sph])

    # 4) Split and train
    F_tr, F_te, y_tr, y_te, coords_tr, coords_te = train_test_split(
        F, y, coords, test_size=cfg.test_size, random_state=cfg.seed
    )

    start_col = 1 if X is not None else 0
    mu = F_tr[:, start_col:].mean(axis=0, keepdims=True)
    sd = F_tr[:, start_col:].std(axis=0, keepdims=True) + 1e-8
    F_tr[:, start_col:] = (F_tr[:, start_col:] - mu) / sd
    F_te[:, start_col:] = (F_te[:, start_col:] - mu) / sd

    model = make_deepkriging_head(F_tr.shape[1], width=50, depth=3, lr=1e-3, l2w=1e-5)
    cbs = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
    ]
    hist = model.fit(F_tr, y_tr, validation_split=0.1, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0,
                     callbacks=cbs)

    # 5) Evaluate
    yhat = model.predict(F_te, verbose=0).ravel()
    rmse = math.sqrt(mean_squared_error(y_te, yhat))
    mae = mean_absolute_error(y_te, yhat)

    t1 = time.time()
    print("\n=== Spherical DeepKriging (GC-RBF) ===")
    print(f"N={cfg.N}, features={F.shape[1]}, epochs={cfg.epochs}")
    # coverage diagnostics
    D_dbg = haversine_gc_distance_matrix(coords, centers, radius=cfg.radius)
    active_dbg = (D_dbg <= theta_vec[None, :]).sum(axis=1)
    print("Metric:", stats.get("metric", "geodesic"))
    print("Per-level theta (units match chosen metric):", [pl["theta"] for pl in stats.get("per_level", [])])
    for pl in stats.get("per_level", []):
        print(
            f"  L{pl['level']} (n={pl['n_centers']}): active min/med/mean = {pl['active_min']}/{pl['active_med']:.1f}/{pl['active_mean']:.1f}")
    print(f"Overall active min = {stats.get('active_overall_min', 'NA')} (inflations={stats.get('inflations', 'NA')})")
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Elapsed: {t1 - t0:.1f}s")

    # 6) Diagnostics
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # a) learning curves
    axs[0].plot(hist.history['loss'], label='train')
    axs[0].plot(hist.history['val_loss'], label='val')
    axs[0].set_title('MSE loss')
    axs[0].legend()

    # b) scatter truth vs pred
    axs[1].scatter(y_te, yhat, s=8, alpha=0.5)
    lim = [min(y_te.min(), yhat.min()), max(y_te.max(), yhat.max())]
    axs[1].plot(lim, lim, 'r--', lw=1)
    axs[1].set_xlabel('Truth')
    axs[1].set_ylabel('Pred')
    axs[1].set_title(f'Test: RMSE={rmse:.3f}, MAE={mae:.3f}')

    # c) residual map on sphere (lon,lat scatter)
    sc = axs[2].scatter(
        coords_te[:, 1], coords_te[:, 0], c=(y_te - yhat), cmap='coolwarm', s=8
    )
    axs[2].set_xlabel('Longitude')
    axs[2].set_ylabel('Latitude')
    axs[2].set_title('Residuals (truth - pred)')
    plt.colorbar(sc, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if cfg.save_plots:
        os.makedirs(cfg.results_dir, exist_ok=True)
        levels_tag = "-".join(str(n) for n in cfg.level_sizes)
        fname = (
            f"{cfg.method_tag}_N{cfg.N}_L{cfg.L_max}_a{cfg.alpha}_nu{cfg.nu}_"
            f"sf{cfg.sigma_f}_sn{cfg.sigma_n}_lv-{levels_tag}_R{cfg.radius}_"
            f"aut{int(cfg.auto_theta)}_tm{cfg.theta_mult}_minA{cfg.min_active}_"
            f"ep{cfg.epochs}_bs{cfg.batch_size}_seed{cfg.seed}_metric_chordal.png"
        )
        out_path = os.path.join(cfg.results_dir, fname)
        fig.savefig(out_path, dpi=cfg.dpi, bbox_inches='tight')
        print(f"Saved plot -> {out_path}")

    plt.show()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)