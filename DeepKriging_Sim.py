"""
Euclidean DeepKriging on Spherical Data (Baseline / Mismatch Test)
------------------------------------------------------------------
This script simulates a *spherical* random field on S^2 and then fits a
standard Euclidean DeepKriging model that treats (lat, lon) as *flat* 2D
coordinates with Euclidean distances. This provides a baseline to compare
against geometry-aware spherical versions.

What it does
============
1) Sample N points uniformly on the sphere -> (lat_deg, lon_deg).
2) Simulate a smooth, isotropic field on the sphere via a truncated
   spherical-harmonic expansion with a Matérn-like angular power spectrum.
3) Build Euclidean multi-resolution Wendland C^4 RBF features using *flat*
   Euclidean distances in normalized (lat, lon) degrees.
4) Train a 3-layer MLP (50-50-50) to predict y from [X | phi_euclid].
5) Report RMSE/MAE and draw quick diagnostic plots.

Dependencies
============
- numpy, matplotlib, scikit-learn, scipy, tensorflow/keras

How to run
==========
$ python DeepKriging_Sim.py

Notes
=====
- This is intentionally *geometry-mismatched*: the data live on the sphere
  but the model uses Euclidean distances in lat-lon. Expect good local fits
  but artifacts across longitudes/near poles. We'll compare this with true
  spherical versions next.
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

np.random.seed(42)

# -----------------------------
# Geometry + simulation helpers
# -----------------------------

def uniform_points_on_sphere(n: int) -> np.ndarray:
    """Draw n points uniformly on the sphere -> (lat_deg, lon_deg)."""
    u = np.random.uniform(-1.0, 1.0, size=n)
    lon = np.random.uniform(-180.0, 180.0, size=n)
    lat = np.degrees(np.arcsin(u))
    return np.column_stack([lat, lon])


def latlon_to_colat_lon(coords_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    theta = np.pi/2.0 - lat  # colatitude in [0, pi]
    phi = lon                # longitude in [-pi, pi]
    return theta, phi


def real_sph_harm_features(coords_deg: np.ndarray, L_max: int):
    theta, phi = latlon_to_colat_lon(coords_deg)
    feats, degrees = [], []
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


def simulate_spherical_field(coords_deg: np.ndarray,
                             L_max: int = 8,
                             alpha: float = 0.02,
                             nu: float = 1.5,
                             sigma_f: float = 1.0,
                             sigma_n: float = 0.1,
                             include_intercept: bool = True):
    """y = f(coords) + noise using SH with Matern-like spectrum.
       C_l = sigma_f^2 * (1 + alpha*l*(l+1))^(-nu)
    Returns: y, X(intercept), F_sh (for reference)
    """
    F_sh, degs = real_sph_harm_features(coords_deg, L_max)
    var_cols = np.array([sigma_f**2 * (1.0 + alpha * l * (l + 1))**(-nu) for l in degs])
    w = np.random.normal(0.0, np.sqrt(var_cols), size=F_sh.shape[1])
    f = F_sh @ w
    noise = np.random.normal(0.0, sigma_n, size=f.shape[0])
    y = (f + noise).astype(np.float32)
    X = np.ones((len(y), 1), dtype=np.float32) if include_intercept else None
    return y, X, F_sh.astype(np.float32)

# -----------------------------
# Euclidean RBF features
# -----------------------------

def wendland_c4(r: np.ndarray) -> np.ndarray:
    """Wendland C^4 on [0,1]. phi(r)=(1-r)^6*(35 r^2 + 18 r + 3)/3 for 0<=r<=1."""
    out = np.zeros_like(r)
    m = (r >= 0) & (r <= 1)
    rm = r[m]
    out[m] = (1.0 - rm)**6 * (35.0*rm**2 + 18.0*rm + 3.0) / 3.0
    return out


def multires_grid_centers(levels: Sequence[Tuple[int, int]]) -> np.ndarray:
    centers = []
    for (nla, nlo) in levels:
        lat = np.linspace(-90.0, 90.0, nla)
        lon = np.linspace(-180.0, 180.0, nlo)
        L, O = np.meshgrid(lat, lon, indexing='ij')
        centers.append(np.column_stack([L.ravel(), O.ravel()]))
    return np.vstack(centers)


def build_phi_wendland_euclid(
    coords_deg: np.ndarray,
    levels: Sequence[Tuple[int, int]] = ((9, 18), (18, 36)),
    thetas: Sequence[float] | None = None,
    auto_theta: bool = True,
    theta_mult: float = 1.6,
    min_active: int = 8,
    max_auto_iters: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euclidean RBF features on normalized (lat, lon) degrees with auto-theta.
    Distances computed in the flat plane: x=(lat/90, lon/180) in [-1,1]^2.
    Returns (phi, centers_deg, theta_vec).
    """
    N = coords_deg.shape[0]
    centers_deg = multires_grid_centers(levels)

    # normalize to comparable scales
    P = np.column_stack([coords_deg[:, 0] / 90.0, coords_deg[:, 1] / 180.0])
    C = np.column_stack([centers_deg[:, 0] / 90.0, centers_deg[:, 1] / 180.0])

    # per-center level ids
    level_ids = np.concatenate([
        np.full(nla*nlo, i, dtype=int) for i, (nla, nlo) in enumerate(levels)
    ])

    # choose per-level thetas
    if thetas is not None:
        assert len(thetas) == len(levels)
        theta_levels = np.asarray(list(thetas), dtype=float)
    else:
        if auto_theta:
            theta_levels = np.zeros(len(levels), dtype=float)
            offset = 0
            for i, (nla, nlo) in enumerate(levels):
                n = nla * nlo
                C_lvl = C[offset:offset+n]
                offset += n
                if n == 1:
                    theta_levels[i] = 2.2  # very wide in normalized units
                    continue
                # center-center distances and median nearest neighbor
                Dcc = np.sqrt(((C_lvl[:, None, :] - C_lvl[None, :, :])**2).sum(axis=2))
                Dcc = Dcc + np.eye(n) * 1e9
                nn = Dcc.min(axis=1)
                med_nn = float(np.median(nn))
                theta_levels[i] = theta_mult * med_nn
        else:
            # fallback heuristic from grid spacing
            theta_levels = []
            base = 1.5
            for (nla, nlo) in levels:
                dlat = 2.0 / (nla - 1)
                dlon = 2.0 / (nlo - 1)
                theta_levels.append(base * 0.5 * np.sqrt(dlat**2 + dlon**2))
            theta_levels = np.array(theta_levels, dtype=float)

    theta_vec = theta_levels[level_ids]

    # pairwise distances P (N x 2) to C (K x 2)
    D = np.sqrt(((P[:, None, :] - C[None, :, :])**2).sum(axis=2))

    # ensure coverage
    inflate = 0
    active = (D <= theta_vec[None, :]).sum(axis=1)
    while active.min() < min_active and inflate < max_auto_iters:
        theta_vec *= 1.2
        active = (D <= theta_vec[None, :]).sum(axis=1)
        inflate += 1

    R = D / theta_vec[None, :]
    phi = wendland_c4(R).astype(np.float32)
    return phi, centers_deg, theta_vec

# -------------------------
# DeepKriging MLP head
# -------------------------

def make_deepkriging_head(input_dim: int, width: int = 50, depth: int = 3, lr: float = 1e-3) -> Sequential:
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(Dense(width, activation='relu', kernel_initializer='he_normal'))
    for _ in range(depth - 1):
        model.add(Dense(width, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))
    opt = Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    return model

# -------------------------
# Config & experiment
# -------------------------
@dataclass
class Config:
    method_tag: str = "EuDK_RBF"
    N: int = 4000
    test_size: float = 0.2
    # Simulation hyper-params
    L_max: int = 8
    alpha: float = 0.02
    nu: float = 1.5
    sigma_f: float = 1.0
    sigma_n: float = 0.1
    # Feature construction (Euclidean)
    levels: Sequence[Tuple[int, int]] = ((9, 18), (18, 36))
    thetas: Sequence[float] | None = None
    auto_theta: bool = True
    theta_mult: float = 1.6
    min_active: int = 8
    max_auto_iters: int = 4
    # Training
    epochs: int = 60
    batch_size: int = 64
    # Output
    results_dir: str = "./Results"
    save_plots: bool = True
    dpi: int = 300
    seed: int = 123


def main(cfg: Config):
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    t0 = time.time()
    coords = uniform_points_on_sphere(cfg.N)

    y, X, _ = simulate_spherical_field(
        coords, L_max=cfg.L_max, alpha=cfg.alpha, nu=cfg.nu,
        sigma_f=cfg.sigma_f, sigma_n=cfg.sigma_n, include_intercept=True
    )

    phi_euc, centers, theta_vec = build_phi_wendland_euclid(
        coords,
        levels=cfg.levels,
        thetas=cfg.thetas,
        auto_theta=cfg.auto_theta,
        theta_mult=cfg.theta_mult,
        min_active=cfg.min_active,
        max_auto_iters=cfg.max_auto_iters,
    )
    F = phi_euc if X is None else np.hstack([X, phi_euc])

    F_tr, F_te, y_tr, y_te, coords_tr, coords_te = train_test_split(
        F, y, coords, test_size=cfg.test_size, random_state=cfg.seed
    )

    model = make_deepkriging_head(F_tr.shape[1])
    hist = model.fit(
        F_tr, y_tr, validation_split=0.1,
        epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0
    )

    yhat = model.predict(F_te, verbose=0).ravel()
    rmse = math.sqrt(mean_squared_error(y_te, yhat))
    mae = mean_absolute_error(y_te, yhat)

    t1 = time.time()
    print("\n=== Euclidean DeepKriging (RBF) ===")
    print(f"N={cfg.N}, features={F.shape[1]}, epochs={cfg.epochs}")
    # coverage diagnostics
    P = np.column_stack([coords[:, 0]/90.0, coords[:, 1]/180.0])
    C = np.column_stack([centers[:, 0]/90.0, centers[:, 1]/180.0])
    D_dbg = np.sqrt(((P[:, None, :] - C[None, :, :])**2).sum(axis=2))
    active_dbg = (D_dbg <= theta_vec[None, :]).sum(axis=1)
    print("Per-level theta (unique values; normalized units):")
    print(np.unique(theta_vec, return_counts=True)[0])
    print(f"Active bases per point - min/median/mean: {active_dbg.min()}/{np.median(active_dbg):.1f}/{active_dbg.mean():.1f}")
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Elapsed: {t1 - t0:.1f}s")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(hist.history['loss'], label='train')
    axs[0].plot(hist.history['val_loss'], label='val')
    axs[0].set_title('MSE loss')
    axs[0].legend()

    axs[1].scatter(y_te, yhat, s=8, alpha=0.5)
    lim = [min(y_te.min(), yhat.min()), max(y_te.max(), yhat.max())]
    axs[1].plot(lim, lim, 'r--', lw=1)
    axs[1].set_xlabel('Truth')
    axs[1].set_ylabel('Pred')
    axs[1].set_title(f'Test: RMSE={rmse:.3f}, MAE={mae:.3f}')

    sc = axs[2].scatter(coords_te[:, 1], coords_te[:, 0], c=(y_te - yhat), cmap='coolwarm', s=8)
    axs[2].set_xlabel('Longitude')
    axs[2].set_ylabel('Latitude')
    axs[2].set_title('Residuals (truth - pred)')
    plt.colorbar(sc, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if cfg.save_plots:
        os.makedirs(cfg.results_dir, exist_ok=True)
        levels_tag = "-".join([f"{nla}x{nlo}" for (nla, nlo) in cfg.levels])
        fname = (
            f"{cfg.method_tag}_N{cfg.N}_L{cfg.L_max}_a{cfg.alpha}_nu{cfg.nu}_"
            f"sf{cfg.sigma_f}_sn{cfg.sigma_n}_lv-{levels_tag}_"
            f"aut{int(cfg.auto_theta)}_tm{cfg.theta_mult}_minA{cfg.min_active}_"
            f"ep{cfg.epochs}_bs{cfg.batch_size}_seed{cfg.seed}.png"
        )
        out_path = os.path.join(cfg.results_dir, fname)
        plt.savefig(out_path, dpi=cfg.dpi, bbox_inches='tight')
        print(f"Saved plot -> {out_path}")

    plt.show()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
