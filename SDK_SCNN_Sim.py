"""
Spherical DeepKriging with Spherical CNN (Chebyshev) — v2 (robust scaling)
---------------------------------------------------------------------------
This script simulates a *spherical* random field on S^2 and then fits a
geometry‑aware DeepKriging model that learns **spherical filters** via a
Chebyshev graph‑spectral CNN built on a great‑circle k‑NN graph. It is a
clean, runnable, and more robust version that addresses common failure
modes (vanishing filters, graph scaling) and matches the 01/02 scripts.

What it does
============
1) Sample N points uniformly on the sphere -> (lat_deg, lon_deg).
2) Simulate a smooth, isotropic field via a truncated spherical‑harmonic
   expansion with a Matern‑like angular power spectrum.
3) Build a geodesic k‑NN graph with **binary adjacency** (stable scale),
   compute the normalized Laplacian L, and **rescale by its largest
   eigenvalue** (power iteration) so Chebyshev polynomials are well‑posed.
4) Learn spherical features with two Chebyshev graph‑conv layers, then a
   DeepKriging MLP head (50‑50‑50 -> 1). Inputs include positional
   encodings and optional low‑order harmonics for global structure.
5) Report RMSE/MAE and save a high‑res plot in ./Results.

How to run
==========
$ python SDK_SCNN_Sim.py

Dependencies
============
- numpy, matplotlib, scipy, scikit‑learn, tensorflow/keras
"""
from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.special import sph_harm

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, LeakyReLU, Concatenate, Add, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam

np.random.seed(42)

# -----------------------------
# Geometry + simulation helpers
# -----------------------------

def uniform_points_on_sphere(n: int) -> np.ndarray:
    u = np.random.uniform(-1.0, 1.0, size=n)
    lon = np.random.uniform(-180.0, 180.0, size=n)
    lat = np.degrees(np.arcsin(u))
    return np.column_stack([lat, lon]).astype(np.float32)


def latlon_to_colat_lon(coords_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    theta = np.pi/2.0 - lat  # [0, pi]
    phi = lon                # [-pi, pi]
    return theta, phi


def simulate_spherical_field(coords_deg: np.ndarray,
                             L_max: int = 8,
                             alpha: float = 0.02,
                             nu: float = 1.5,
                             sigma_f: float = 1.0,
                             sigma_n: float = 0.1,
                             include_intercept: bool = True):
    """y = f(coords) + noise via SH with Matern‑like spectrum.
       C_l = sigma_f^2 * (1 + alpha * l*(l+1))^(-nu)
    Returns: y (N,), X (N,1) intercept, F_sh (N,D) for reference.
    """
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
    F_sh = np.stack(feats, axis=1)
    var_cols = np.array([sigma_f**2 * (1.0 + alpha * l * (l + 1))**(-nu) for l in degrees])
    w = np.random.normal(0.0, np.sqrt(var_cols), size=F_sh.shape[1])
    f = F_sh @ w
    noise = np.random.normal(0.0, sigma_n, size=f.shape[0])
    y = (f + noise).astype(np.float32)
    X = np.ones((len(y), 1), dtype=np.float32) if include_intercept else None
    return y, X, F_sh.astype(np.float32)

# -----------------------------
# Graph on the sphere (robust)
# -----------------------------

def haversine_gc_distance_matrix(A_deg: np.ndarray, B_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    A_lat = np.radians(A_deg[:, 0])[:, None]
    A_lon = np.radians(A_deg[:, 1])[:, None]
    B_lat = np.radians(B_deg[:, 0])[None, :]
    B_lon = np.radians(B_deg[:, 1])[None, :]
    dlat = B_lat - A_lat
    dlon = B_lon - A_lon
    h = np.sin(dlat/2.0)**2 + np.cos(A_lat) * np.cos(B_lat) * np.sin(dlon/2.0)**2
    ang = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))  # radians
    return (radius * ang).astype(np.float32)


def power_iteration_lmax(L: np.ndarray, iters: int = 50) -> float:
    N = L.shape[0]
    b = np.random.randn(N).astype(np.float32)
    b /= np.linalg.norm(b) + 1e-12
    for _ in range(iters):
        b = L @ b
        b_norm = np.linalg.norm(b) + 1e-12
        b /= b_norm
    lam = float(b.T @ (L @ b) / (b.T @ b + 1e-12))
    return max(lam, 1e-6)


def build_knn_graph(coords_deg: np.ndarray, k: int = 12, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Binary k‑NN adjacency with symmetric normalization and Laplacian rescaling.
    Returns (W, L_tilde, lambda_max).
    """
    N = coords_deg.shape[0]
    D = haversine_gc_distance_matrix(coords_deg, coords_deg, radius=radius)
    np.fill_diagonal(D, np.inf)
    k_eff = min(k, N - 1)
    idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    W = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        W[i, idx[i]] = 1.0
    # symmetrize
    W = np.maximum(W, W.T)

    # normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    D_inv_sqrt = np.outer(d_inv_sqrt, d_inv_sqrt)
    I = np.eye(N, dtype=np.float32)
    L = I - (D_inv_sqrt * W)

    # rescale to [-1,1] using largest eigenvalue
    lmax = power_iteration_lmax(L.astype(np.float32))
    L_tilde = (2.0 * L / lmax) - I
    return W, L_tilde.astype(np.float32), lmax

# -----------------------------
# Chebyshev graph convolution
# -----------------------------
class ChebConv(Layer):
    def __init__(self, L_tilde: np.ndarray, K: int, channels: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.K = int(K)
        self.channels = int(channels)
        self.use_bias = bool(use_bias)
        self.L_tilde = tf.constant(L_tilde, dtype=tf.float32)

    def build(self, input_shape):
        Fin = int(input_shape[-1])

        # Chebyshev weights: (K, Fin, channels)
        self.theta = self.add_weight(
            name="theta",
            shape=(self.K, Fin, self.channels),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",                          # <-- use keyword, not positional
                shape=(self.channels,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, X):
        T0 = X
        T_list = [T0]
        if self.K > 1:
            T1 = tf.matmul(self.L_tilde, X)
            T_list.append(T1)
        for _ in range(2, self.K):
            Tk = 2.0 * tf.matmul(self.L_tilde, T_list[-1]) - T_list[-2]
            T_list.append(Tk)
        T_stack = tf.stack(T_list, axis=0)  # (K, N, Fin)
        out = tf.einsum('kni,kic->nc', T_stack, self.theta)
        if self.bias is not None:
            out = out + self.bias
        return out

# -----------------------------
# Feature builders
# -----------------------------

def posenc(coords_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    Xp = np.column_stack([
        np.ones_like(lat),
        np.sin(lat), np.cos(lat),
        np.sin(lon), np.cos(lon),
    ]).astype(np.float32)
    # z‑score excluding intercept
    Xp[:, 1:] = (Xp[:, 1:] - Xp[:, 1:].mean(axis=0)) / (Xp[:, 1:].std(axis=0) + 1e-8)
    return Xp


def real_sph_harm_features(coords_deg: np.ndarray, L_feat: int) -> np.ndarray:
    theta, phi = latlon_to_colat_lon(coords_deg)
    feats = []
    for l in range(L_feat + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            if m < 0:
                feat = np.sqrt(2.0) * Y_lm.imag
            elif m == 0:
                feat = Y_lm.real
            else:
                feat = np.sqrt(2.0) * Y_lm.real
            feats.append(np.asarray(feat, dtype=float))
    F = np.stack(feats, axis=1).astype(np.float32)
    # z‑score
    F = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-8)
    return F

# -----------------------------
# Config & experiment
# -----------------------------
@dataclass
class Config:
    method_tag: str = "SphDK_SCNN"
    N: int = 4000
    test_size: float = 0.2
    # Simulation
    L_max: int = 8
    alpha: float = 0.02
    nu: float = 1.5
    sigma_f: float = 1.0
    sigma_n: float = 0.1
    # Graph/CNN
    knn: int = 16
    K_cheb: int = 5
    channels1: int = 32
    channels2: int = 32
    include_harmonics: bool = True
    L_feat: int = 4
    lr: float = 5e-4
    # Training
    epochs: int = 100
    batch_size: int | None = None  # full‑batch if None
    # Output
    results_dir: str = "./Results"
    save_plots: bool = True
    dpi: int = 300
    seed: int = 123


def main(cfg: Config):
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    t0 = time.time()
    # 1) locations
    coords = uniform_points_on_sphere(cfg.N)

    # 2) simulate
    y, _, _ = simulate_spherical_field(coords, L_max=cfg.L_max, alpha=cfg.alpha, nu=cfg.nu,
                                       sigma_f=cfg.sigma_f, sigma_n=cfg.sigma_n, include_intercept=True)

    # 3) graph
    W, L_tilde, lmax = build_knn_graph(coords, k=cfg.knn, radius=1.0)
    deg_sum = W.sum(axis=1)
    print(f"Graph degree (sum of weights) min/median/mean: {deg_sum.min():.2f}/{np.median(deg_sum):.2f}/{deg_sum.mean():.2f}")
    print(f"Estimated lambda_max(L): {lmax:.3f}")

    # 4) inputs
    Xp = posenc(coords)
    if cfg.include_harmonics:
        Fh = real_sph_harm_features(coords, cfg.L_feat)
        X_input = np.hstack([Xp, Fh]).astype(np.float32)
    else:
        X_input = Xp

    # train/test split via masks
    idx = np.arange(cfg.N)
    idx_tr, idx_te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed)

    # target scaling
    y_mean = float(y[idx_tr].mean())
    y_std = float(y[idx_tr].std() + 1e-8)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    sw_train = np.zeros((cfg.N,), dtype=np.float32)
    sw_val = np.zeros((cfg.N,), dtype=np.float32)
    sw_train[idx_tr] = 1.0
    sw_val[idx_te] = 1.0

    effective_bs = cfg.N if cfg.batch_size is None else cfg.batch_size

    # 5) model
    inputs = Input(shape=(X_input.shape[1],), batch_size=effective_bs)
    h = ChebConv(L_tilde, K=cfg.K_cheb, channels=cfg.channels1)(inputs)
    h = LayerNormalization()(h)
    h = LeakyReLU(0.1)(h)
    h = ChebConv(L_tilde, K=cfg.K_cheb, channels=cfg.channels2)(h)
    h = LayerNormalization()(h)
    h = LeakyReLU(0.1)(h)
    h = Concatenate()([h, inputs])
    h = Dense(50, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(50, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(50, activation='relu', kernel_initializer='he_normal')(h)
    y_mlp = Dense(1, activation='linear')(h)
    y_lin = Dense(1, use_bias=False)(inputs)
    outputs = Add()([y_mlp, y_lin])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=cfg.lr), metrics=['mse','mae'])

    # callbacks: early stop + LR reduce
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]

    hist = model.fit(X_input, y_scaled,
                     sample_weight=sw_train,
                     validation_data=(X_input, y_scaled, sw_val),
                     epochs=cfg.epochs,
                     batch_size=effective_bs,
                     verbose=0,
                     callbacks=cb)

    # diagnostics
    conv1_out = tf.keras.Model(inputs=model.inputs, outputs=model.layers[3].output)(X_input, training=False).numpy()
    print("Conv1 feature std (first 5 channels):", np.std(conv1_out, axis=0)[:5].round(3))

    # predictions
    yhat_scaled = model.predict(X_input, batch_size=effective_bs, verbose=0).ravel()
    yhat = yhat_scaled * y_std + y_mean

    rmse = math.sqrt(mean_squared_error(y[idx_te], yhat[idx_te]))
    mae = mean_absolute_error(y[idx_te], yhat[idx_te])

    t1 = time.time()
    print("\n=== Spherical DeepKriging (SCNN/Chebyshev, v2) ===")
    print(f"N={cfg.N}, knn={cfg.knn}, K_cheb={cfg.K_cheb}, channels=({cfg.channels1},{cfg.channels2}), epochs={cfg.epochs}")
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"Elapsed: {t1 - t0:.1f}s")

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(hist.history['loss'], label='train')
    axs[0].plot(hist.history['val_loss'], label='val')
    axs[0].set_title('MSE loss (scaled y)')
    axs[0].legend()

    axs[1].scatter(y[idx_te], yhat[idx_te], s=8, alpha=0.5)
    lim = [min(y[idx_te].min(), yhat[idx_te].min()), max(y[idx_te].max(), yhat[idx_te].max())]
    axs[1].plot(lim, lim, 'r--', lw=1)
    axs[1].set_xlabel('Truth')
    axs[1].set_ylabel('Pred')
    axs[1].set_title(f'Test: RMSE={rmse:.3f}, MAE={mae:.3f}')

    sc = axs[2].scatter(coords[idx_te, 1], coords[idx_te, 0], c=(y[idx_te] - yhat[idx_te]), cmap='coolwarm', s=8)
    axs[2].set_xlabel('Longitude')
    axs[2].set_ylabel('Latitude')
    axs[2].set_title('Residuals (truth - pred)')
    plt.colorbar(sc, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if cfg.save_plots:
        os.makedirs(cfg.results_dir, exist_ok=True)
        fname = (f"{cfg.method_tag}_N{cfg.N}_L{cfg.L_max}_a{cfg.alpha}_nu{cfg.nu}_"
                 f"sf{cfg.sigma_f}_sn{cfg.sigma_n}_knn{cfg.knn}_K{cfg.K_cheb}_ch{cfg.channels1}-{cfg.channels2}_"
                 f"harm{int(cfg.include_harmonics)}_Lf{cfg.L_feat}_ep{cfg.epochs}_seed{cfg.seed}.png")
        out_path = os.path.join(cfg.results_dir, fname)
        plt.savefig(out_path, dpi=cfg.dpi, bbox_inches='tight')
        print(f"Saved plot -> {out_path}")

    plt.show()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)