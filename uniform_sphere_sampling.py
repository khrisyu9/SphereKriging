import numpy as np
from typing import Iterable, Dict, Any, List, Tuple


def uniform_points_on_sphere(n: int) -> np.ndarray:
    """Draw n points uniformly on S^2 via Marsaglia -> (lat_deg, lon_deg)."""
    a = np.empty(n, dtype=float)
    b = np.empty(n, dtype=float)
    k = 0
    # Rejection sampling to fill (a,b) ~ Unif(unit disk)
    while k < n:
        m = max(1, n - k)
        cand = np.random.uniform(-1.0, 1.0, size=(2 * m, 2))
        s = (cand ** 2).sum(axis=1)
        keep = s < 1.0
        if not np.any(keep):
            continue
        take = min(np.count_nonzero(keep), n - k)
        sel = cand[keep][:take]
        a[k:k + take] = sel[:, 0]
        b[k:k + take] = sel[:, 1]
        k += take

    s = a * a + b * b
    # Marsaglia map: (a,b) -> (x,y,z) on the unit sphere
    w = np.sqrt(1.0 - s)
    x = 2.0 * a * w
    y = 2.0 * b * w
    z = 1.0 - 2.0 * s

    # Convert to (lat, lon) in degrees
    lat = np.degrees(np.arcsin(z))           # [-90, 90]
    lon = np.degrees(np.arctan2(y, x))       # (-180, 180]
    return np.column_stack([lat, lon])


# ---------- Uniformity checks ----------

def spherical_cap_fraction_theory(alpha_deg: float) -> float:
    """Theoretical fraction of surface area within geodesic radius alpha (north-pole cap)."""
    alpha = np.radians(alpha_deg)
    return (1.0 - np.cos(alpha)) / 2.0


def cap_fractions(lat_deg: np.ndarray,
                  alphas_deg: Iterable[float]) -> List[Dict[str, float]]:
    """
    Empirical vs theoretical fractions for spherical caps around the north pole.
    A point is in the cap iff geodesic distance <= alpha, i.e., lat >= 90° - alpha.
    """
    out = []
    for a in alphas_deg:
        emp = np.mean(lat_deg >= (90.0 - a))
        th = spherical_cap_fraction_theory(a)
        out.append({"alpha_deg": float(a), "empirical": float(emp), "theory": float(th)})
    return out


def hemisphere_fraction(lat_deg: np.ndarray) -> Dict[str, float]:
    """Sanity check: each hemisphere should have probability 1/2."""
    emp = np.mean(lat_deg >= 0.0)
    return {"empirical": float(emp), "theory": 0.5}


def equal_area_band_fractions(lat_deg: np.ndarray, K: int = 6) -> Dict[str, Any]:
    """
    Split the sphere into K equal-area latitude bands by uniform 'z = sin(lat)' edges.
    Each band should have probability 1/K.
    """
    z = np.sin(np.radians(lat_deg))
    edges = np.linspace(-1.0, 1.0, K + 1)
    fracs = []
    for i in range(K):
        left, right = edges[i], edges[i + 1]
        # include right endpoint on the last bin for completeness
        if i < K - 1:
            mask = (z >= left) & (z < right)
        else:
            mask = (z >= left) & (z <= right)
        fracs.append(float(np.mean(mask)))
    theory = [1.0 / K] * K
    return {"K": K, "empirical": fracs, "theory": theory, "edges_z": edges.tolist()}


def equal_lat_band_fractions(lat_deg: np.ndarray, K: int = 6) -> Dict[str, Any]:
    """
    Split by *equal latitude* edges (not equal area). The theoretical band
    probabilities are (sin(phi2) - sin(phi1)) / 2, so they are not equal.
    This helps show why 'uniform latitude' would be wrong.
    """
    edges = np.linspace(-90.0, 90.0, K + 1)
    emp, th = [], []
    for i in range(K):
        lo, hi = edges[i], edges[i + 1]
        if i < K - 1:
            mask = (lat_deg >= lo) & (lat_deg < hi)
        else:
            mask = (lat_deg >= lo) & (lat_deg <= hi)
        emp.append(float(np.mean(mask)))
        th.append(float((np.sin(np.radians(hi)) - np.sin(np.radians(lo))) / 2.0))
    return {"K": K, "empirical": emp, "theory": th, "edges_lat_deg": edges.tolist()}


def check_uniformity(n: int = 200_000,
                     alphas_deg: Iterable[float] = (5, 15, 30, 60),
                     K_bands: int = 6,
                     seed: int = 0) -> Dict[str, Any]:
    """
    Generate n points and return a dictionary of fractions you can inspect.
    """
    if seed is not None:
        np.random.seed(seed)
    latlon = uniform_points_on_sphere(n)
    lat_deg = latlon[:, 0]

    caps = cap_fractions(lat_deg, alphas_deg)
    hemi = hemisphere_fraction(lat_deg)
    eq_area = equal_area_band_fractions(lat_deg, K_bands)
    eq_lat = equal_lat_band_fractions(lat_deg, K_bands)

    return {"n": n, "caps": caps, "hemisphere": hemi,
            "equal_area_bands": eq_area, "equal_lat_bands": eq_lat}


# ---------- CLI for quick inspection ----------

def _pretty_print(res: Dict[str, Any]) -> None:
    print(f"\nGenerated n = {res['n']} points on S^2 (Marsaglia).")
    print("\nSpherical cap test (north pole):")
    print("  alpha_deg   empirical      theory       abs_err")
    for item in res["caps"]:
        emp, th = item["empirical"], item["theory"]
        print(f"  {item['alpha_deg']:9.1f}   {emp:9.6f}   {th:9.6f}   {abs(emp - th):9.6f}")

    hemi = res["hemisphere"]
    print("\nHemisphere test:")
    print(f"  empirical = {hemi['empirical']:.6f}   theory = {hemi['theory']:.6f}   abs_err = {abs(hemi['empirical'] - hemi['theory']):.6f}")

    eq_area = res["equal_area_bands"]
    print(f"\nEqual-area latitude bands (K={eq_area['K']}, by z = sin(lat)) — each should be ~1/K:")
    for i, (emp, th) in enumerate(zip(eq_area["empirical"], eq_area["theory"])):
        print(f"  band {i+1:2d}: empirical = {emp:.6f}   theory = {th:.6f}   abs_err = {abs(emp - th):.6f}")

    eq_lat = res["equal_lat_bands"]
    print(f"\nEqual-latitude bands (K={eq_lat['K']}) — theoretical masses differ:")
    print("  idx    [lat_lo, lat_hi]    empirical      theory       abs_err")
    for i, (emp, th) in enumerate(zip(eq_lat["empirical"], eq_lat["theory"])):
        lo, hi = eq_lat["edges_lat_deg"][i], eq_lat["edges_lat_deg"][i + 1]
        print(f"  {i+1:2d}   [{lo:6.1f},{hi:6.1f}]   {emp:9.6f}   {th:9.6f}   {abs(emp - th):9.6f}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Uniform sampling on S^2 + sanity checks.")
    parser.add_argument("--n", type=int, default=200_000, help="number of points")
    parser.add_argument("--alphas", type=float, nargs="*", default=[5, 15, 30, 60],
                        help="cap radii (degrees) to test")
    parser.add_argument("--bands", type=int, default=6, help="number of latitude bands to test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    results = check_uniformity(n=args.n, alphas_deg=args.alphas, K_bands=args.bands, seed=args.seed)
    _pretty_print(results)

