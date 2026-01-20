# verify_r2_internal_schur_v7.py
# R2 verifier with robust Aut(D) fitting based on least-squares phase alignment.

import argparse
import json
import math
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ----------------------------
# utils
# ----------------------------
def parse_complex(s: Any) -> complex:
    if isinstance(s, complex):
        return s
    if s is None:
        raise ValueError("parse_complex: got None")
    t = str(s).strip()
    t = t.replace(" ", "")
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    if "j" not in t and "i" in t:
        t = t.replace("i", "j")
    return complex(t)


def stats_dict(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"count": 0, "max": float("nan"), "mean": float("nan"),
                "median": float("nan"), "p90": float("nan")}
    return {
        "count": int(x.size),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
    }


def disk_auto_kaw(w: np.ndarray, a: complex, k: complex) -> np.ndarray:
    w = np.asarray(w, dtype=np.complex128)
    den = 1.0 - np.conj(a) * w
    return k * (w - a) / den


def mobius_apply_matrix(A: np.ndarray, x: complex) -> complex:
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    den = c * x + d
    if abs(den) == 0:
        return complex(np.inf)
    return (a * x + b) / den


def cayley_m_to_w(m: complex, form: str = "alt", b: float = 1.0) -> complex:
    if form == "alt":
        return (m - 1j * b) / (m + 1j * b)
    if form == "std":
        return (b + 1j * m) / (b - 1j * m)
    raise ValueError(f"unknown cayley form: {form}")


def spectral_cayley(lam: complex, lam_sign: int, zeta_sign: int, zeta_scale: float) -> complex:
    lam2 = lam_sign * lam
    z = 1j * (1.0 + lam2) / (1.0 - lam2)
    return (zeta_sign * zeta_scale) * z


def schur_eval_from_alphas(alphas: np.ndarray, lam: complex) -> complex:
    f = 0.0 + 0.0j
    for a in alphas[::-1]:
        den = 1.0 + np.conj(a) * lam * f
        if den == 0:
            return complex(np.nan)
        f = (a + lam * f) / den
    return f


def canonical_step_M(zeta: complex, H: np.ndarray, Jside: str = "JH") -> np.ndarray:
    J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    H = np.asarray(H, dtype=np.complex128)

    if Jside == "JH":
        A = J @ H
    elif Jside == "HJ":
        A = H @ J
    else:
        raise ValueError("Jside must be 'JH' or 'HJ'")

    I = np.eye(2, dtype=np.complex128)
    L = I - (zeta / 2.0) * A
    R = I + (zeta / 2.0) * A
    M = np.linalg.solve(L, R)
    return M


def canonical_product(Hk: np.ndarray,
                      zeta: complex,
                      Kuse: int,
                      Jside: str,
                      prod_order: str,
                      invert_each: bool) -> np.ndarray:
    P = np.eye(2, dtype=np.complex128)
    for k in range(Kuse):
        Mk = canonical_step_M(zeta, Hk[k], Jside=Jside)
        if invert_each:
            Mk = np.linalg.inv(Mk)

        if prod_order == "left":
            P = Mk @ P
        elif prod_order == "right":
            P = P @ Mk
        else:
            raise ValueError("prod_order must be 'left' or 'right'")
    return P


def taylor_coeffs_from_circle_samples(Svals: np.ndarray, r: float, Ncoeff: int) -> np.ndarray:
    M = int(Svals.size)
    F = np.fft.fft(Svals) / M
    n = np.arange(Ncoeff, dtype=np.float64)
    rn = np.power(r, n)
    if r == 0:
        raise ValueError("r must be > 0")
    c = F[:Ncoeff] / rn
    return np.asarray(c, dtype=np.complex128)


def eval_taylor(c: np.ndarray, lam: complex) -> complex:
    v = 0.0 + 0.0j
    for ck in c[::-1]:
        v = v * lam + ck
    return v


def read_hk_report_mobius(path: str) -> Tuple[bool, Optional[complex], Optional[complex]]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    def pick(keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    mob_used = pick(["mobius_used", "mobiusUsed", "use_mobius", "useMobius"])
    mob_used = bool(mob_used) if mob_used is not None else False

    if "mobius" in d and isinstance(d["mobius"], dict):
        md = d["mobius"]
        if "used" in md:
            mob_used = bool(md["used"])
        a = md.get("a")
        k = md.get("k")
    else:
        a = pick(["mobius_a", "mobius-a", "a", "mobiusA"])
        k = pick(["mobius_k", "mobius-k", "k", "mobiusK"])

    mob_a = None
    mob_k = None
    if a is not None:
        try:
            mob_a = parse_complex(a)
        except Exception:
            mob_a = None
    if k is not None:
        try:
            mob_k = parse_complex(k)
        except Exception:
            mob_k = None

    return mob_used, mob_a, mob_k


def apply_mobius_to_samples(Svals: np.ndarray, a: complex, k: complex) -> np.ndarray:
    den = 1.0 - np.conj(a) * Svals
    return k * (Svals - a) / den


def fit_autD_random(w_src: np.ndarray,
                    w_tgt: np.ndarray,
                    tries: int,
                    seed: int,
                    a_rmax: float) -> Tuple[complex, float, float]:
    rng = random.Random(seed)
    best_score = float("inf")
    best_a = 0.0 + 0.0j
    best_phi = 0.0

    w_src = np.asarray(w_src, dtype=np.complex128)
    w_tgt = np.asarray(w_tgt, dtype=np.complex128)

    for _ in range(tries):
        u = rng.random()
        rr = math.sqrt(u) * a_rmax
        th = 2 * math.pi * rng.random()
        a = rr * complex(math.cos(th), math.sin(th))

        phi = 2 * math.pi * rng.random()
        k = complex(math.cos(phi), math.sin(phi))

        w_map = disk_auto_kaw(w_src, a, k)
        err = np.abs(w_map - w_tgt)
        score = float(np.max(err))

        if score < best_score:
            best_score = score
            best_a = a
            best_phi = phi

    return best_a, best_phi, best_score


def best_k_for_a(w_src: np.ndarray, w_tgt: np.ndarray, a: complex) -> Optional[complex]:
    f_a = (w_src - a) / (1.0 - np.conj(a) * w_src)
    c = np.sum(w_tgt * np.conj(f_a))
    if abs(c) == 0:
        return None
    return c / abs(c)


def aut_score_for_a(w_src: np.ndarray, w_tgt: np.ndarray, a: complex) -> float:
    k = best_k_for_a(w_src, w_tgt, a)
    if k is None:
        return float("inf")
    w_map = disk_auto_kaw(w_src, a, k)
    err = np.abs(w_map - w_tgt)
    return float(np.max(err))


def fit_autD_phase_opt(w_src: np.ndarray,
                       w_tgt: np.ndarray,
                       tries: int,
                       seed: int,
                       a_rmax: float) -> Tuple[complex, float, float]:
    rng = random.Random(seed)
    w_src = np.asarray(w_src, dtype=np.complex128)
    w_tgt = np.asarray(w_tgt, dtype=np.complex128)

    best_a = 0.0 + 0.0j
    best_score = float("inf")

    for _ in range(tries):
        u = rng.random()
        rr = math.sqrt(u) * a_rmax
        th = 2 * math.pi * rng.random()
        a = rr * complex(math.cos(th), math.sin(th))
        score = aut_score_for_a(w_src, w_tgt, a)
        if score < best_score:
            best_score = score
            best_a = a

    step = 0.15
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                cand = best_a + step * complex(dx, dy)
                if abs(cand) >= a_rmax:
                    continue
                score = aut_score_for_a(w_src, w_tgt, cand)
                if score < best_score:
                    best_score = score
                    best_a = cand
                    improved = True
        step *= 0.5

    best_k = best_k_for_a(w_src, w_tgt, best_a)
    if best_k is None:
        return best_a, 0.0, best_score
    best_phi = float(np.angle(best_k))
    return best_a, best_phi, best_score


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svals-npy", required=True)
    ap.add_argument("--hk-npy", required=True)
    ap.add_argument("--alphas-npy", default=None)
    ap.add_argument("--hk-report-json", default=None)

    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--r", type=float, required=True)

    ap.add_argument("--rho-test", type=float, default=0.95)
    ap.add_argument("--Kuse", type=int, default=160)

    ap.add_argument("--nsamp", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tol-alpha", type=float, default=1e-6)
    ap.add_argument("--tol-hk", type=float, default=1e-2)

    ap.add_argument("--cayley-form", choices=["std", "alt"], default="alt")
    ap.add_argument("--cayley-b", type=float, default=1.0)

    ap.add_argument("--lam-sign", type=int, default=1)
    ap.add_argument("--zeta-sign", type=int, default=1)
    ap.add_argument("--zeta-scale", type=float, default=1.0)

    ap.add_argument("--Jside", choices=["JH", "HJ"], default="JH")
    ap.add_argument("--prod-order", choices=["left", "right"], default="left")
    ap.add_argument("--invert-each", action="store_true")

    ap.add_argument("--m0", default="1j", help="initial m-value, complex string; default 1j")

    ap.add_argument("--sync-target-to-hk-report", action="store_true")
    ap.add_argument("--target-mobius-a", default=None)
    ap.add_argument("--target-mobius-k", default=None)

    ap.add_argument("--apply-aut-a", default=None)
    ap.add_argument("--apply-aut-phi", default=None)

    ap.add_argument("--fit-aut", action="store_true")
    ap.add_argument("--fit-method", choices=["random", "phase-opt"], default="phase-opt")
    ap.add_argument("--fit-tries", type=int, default=20000)
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--fit-a-rmax", type=float, default=0.99)
    ap.add_argument("--apply-fit-aut", action="store_true",
                    help="if set, apply fitted Aut(D) to canonical and report AFTER-FIT stats")

    ap.add_argument("--out-json", default=None)
    ap.add_argument("--save-prefix", default=None,
                    help="if set, save npy arrays: <prefix>_w_tar.npy, _w_alpha.npy, _w_hk.npy, _w_hk_adj.npy, _err_*.npy")

    args = ap.parse_args()

    aut_a = None
    aut_k = None
    if args.apply_aut_a is not None and args.apply_aut_phi is not None:
        aut_a = parse_complex(args.apply_aut_a)
        aut_phi = float(args.apply_aut_phi)
        aut_k = np.exp(1j * aut_phi)
        if abs(aut_a) >= 1.0:
            print(f"[warn] apply-aut-a |a|>=1 (|a|={abs(aut_a)}). disabling apply-aut.")
            aut_a = None
            aut_k = None

    Svals = np.load(args.svals_npy)
    Svals = np.asarray(Svals)
    if Svals.ndim != 1 or not np.iscomplexobj(Svals):
        raise ValueError("Svals must be 1D complex npy")

    Hk = np.load(args.hk_npy, allow_pickle=True)
    if isinstance(Hk, dict):
        Hk = Hk.get("Hk", Hk)
    Hk = np.asarray(Hk)
    if Hk.ndim != 3 or Hk.shape[1:] != (2, 2):
        raise ValueError("Hk must be shape (K,2,2)")

    Kmax = Hk.shape[0]
    Kuse = min(args.Kuse, Kmax)

    alphas = None
    if args.alphas_npy is not None:
        alphas = np.load(args.alphas_npy)
        alphas = np.asarray(alphas)
        if alphas.ndim != 1 or not np.iscomplexobj(alphas):
            raise ValueError("alphas must be 1D complex npy")
        if alphas.size < Kuse:
            raise ValueError(f"alphas length {alphas.size} < Kuse {Kuse}")
        alphas_use = alphas[:Kuse]
    else:
        alphas_use = None

    target_mobius_used = False
    target_mobius_a = None
    target_mobius_k = None

    if args.sync_target_to_hk_report:
        if args.target_mobius_a is not None and args.target_mobius_k is not None:
            target_mobius_used = True
            target_mobius_a = parse_complex(args.target_mobius_a)
            target_mobius_k = parse_complex(args.target_mobius_k)
        else:
            if args.hk_report_json is not None and os.path.exists(args.hk_report_json):
                used, aa, kk = read_hk_report_mobius(args.hk_report_json)
                if used and aa is not None and kk is not None:
                    target_mobius_used = True
                    target_mobius_a = aa
                    target_mobius_k = kk

    Svals_eff = Svals
    if target_mobius_used:
        Svals_eff = apply_mobius_to_samples(Svals_eff, target_mobius_a, target_mobius_k)

    M = int(Svals_eff.size)
    Ncoeff = min(M // 2, 4 * Kuse + 32)
    coeffs = taylor_coeffs_from_circle_samples(Svals_eff, args.r, Ncoeff)

    rng = np.random.default_rng(args.seed)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=args.nsamp)
    lams = args.rho_test * np.exp(1j * thetas)

    m0 = parse_complex(args.m0)

    w_tar_list = []
    w_alpha_list = []
    w_hk_list = []
    w_hk_adj_list = []

    err_alpha = []
    err_hk = []
    err_hk_adj = []

    for lam in lams:
        w_tar = eval_taylor(coeffs, lam)
        w_tar_list.append(w_tar)

        if alphas_use is not None:
            w_a = schur_eval_from_alphas(alphas_use, lam)
            w_alpha_list.append(w_a)
            err_alpha.append(abs(w_a - w_tar))

        zeta = spectral_cayley(lam, args.lam_sign, args.zeta_sign, args.zeta_scale)
        P = canonical_product(Hk, zeta, Kuse, args.Jside, args.prod_order, args.invert_each)
        m = mobius_apply_matrix(P, m0)
        w_hk = cayley_m_to_w(m, form=args.cayley_form, b=args.cayley_b)

        w_hk_list.append(w_hk)
        err_hk.append(abs(w_hk - w_tar))

        w_hk_adj = w_hk
        if aut_a is not None:
            w_hk_adj = disk_auto_kaw(np.array([w_hk_adj], dtype=np.complex128), aut_a, aut_k)[0]
        w_hk_adj_list.append(w_hk_adj)
        err_hk_adj.append(abs(w_hk_adj - w_tar))

    w_tar_arr = np.array(w_tar_list, dtype=np.complex128)
    w_hk_arr = np.array(w_hk_list, dtype=np.complex128)
    w_hk_adj_arr = np.array(w_hk_adj_list, dtype=np.complex128)

    print("=" * 98)
    print("[verify_r2_internal_schur_v7]")
    print(f"svals   : {args.svals_npy}")
    print(f"hk      : {args.hk_npy}")
    if args.alphas_npy is not None:
        print(f"alphas  : {args.alphas_npy}")
    if args.hk_report_json is not None:
        print(f"hk_rep  : {args.hk_report_json}")
    print(f"(t0,eta,r)=({args.t0},{args.eta},{args.r})  rho={args.rho_test}  M={M}")
    print(f"Kuse={Kuse}  Ncoeff={Ncoeff}  nsamp={args.nsamp}  seed={args.seed}")
    print(f"canonical: cayley={args.cayley_form}(b={args.cayley_b}), Jside={args.Jside}, prod={args.prod_order}, invert_each={args.invert_each}")
    print(f"convs: lam_sign={args.lam_sign}, zeta_sign={args.zeta_sign}, zeta_scale={args.zeta_scale}")
    if target_mobius_used:
        print(f"target pre-mobius applied: a={target_mobius_a}, k={target_mobius_k}")
    else:
        print("target pre-mobius applied: (none)")

    if alphas_use is not None:
        stA = stats_dict(np.array(err_alpha, dtype=float))
        print("-" * 98)
        print(f"A) alpha-Schur vs target(FFT)   stats: {stA}")
        print(f"tol_alpha={args.tol_alpha:g} => {'PASS' if stA['max'] <= args.tol_alpha else 'FAIL'}")

    stB = stats_dict(np.array(err_hk, dtype=float))
    print("-" * 98)
    print(f"B) canonical(Hk) vs target(FFT) stats: {stB}")
    print(f"tol_hk={args.tol_hk:g} => {'PASS' if stB['max'] <= args.tol_hk else 'FAIL'}")

    if aut_a is not None:
        stB2 = stats_dict(np.array(err_hk_adj, dtype=float))
        print("-" * 98)
        print(f"B2) canonical(Hk) AFTER apply-aut stats: {stB2}")
        print(f"apply-aut-a={aut_a}  apply-aut-phi(rad)={np.angle(aut_k)}")
        print(f"tol_hk={args.tol_hk:g} => {'PASS' if stB2['max'] <= args.tol_hk else 'FAIL'}")

    fit_info = None
    if args.fit_aut:
        print("-" * 98)
        print(f"[fit-aut] method={args.fit_method}")
        print(f"tries   : {args.fit_tries}  seed={args.fit_seed}  a_rmax={args.fit_a_rmax}")
        if args.fit_method == "random":
            best_a, best_phi, best_score = fit_autD_random(
                w_hk_arr, w_tar_arr, args.fit_tries, args.fit_seed, args.fit_a_rmax
            )
        else:
            best_a, best_phi, best_score = fit_autD_phase_opt(
                w_hk_arr, w_tar_arr, args.fit_tries, args.fit_seed, args.fit_a_rmax
            )
        best_k = np.exp(1j * best_phi)
        print(f"best a      : {best_a}")
        print(f"best phi(rad): {best_phi}   (deg={best_phi*180.0/np.pi})")
        print(f"best score  : {best_score}")

        fit_info = {
            "best_a": [best_a.real, best_a.imag],
            "best_phi": float(best_phi),
            "best_score": float(best_score),
            "fit_method": args.fit_method,
        }

        if args.apply_fit_aut:
            if abs(best_a) >= 1.0:
                print(f"[warn] fitted |a|>=1 (|a|={abs(best_a)}). not applying.")
            else:
                w_fit = disk_auto_kaw(w_hk_arr, best_a, best_k)
                err_fit = np.abs(w_fit - w_tar_arr)
                stFit = stats_dict(err_fit.astype(float))
                print("AFTER FIT stats:", stFit)
                print(f"tol_hk={args.tol_hk:g} => {'PASS' if stFit['max'] <= args.tol_hk else 'FAIL'}")

    print("=" * 98)

    if args.save_prefix is not None:
        pref = args.save_prefix
        np.save(pref + "_w_tar.npy", w_tar_arr)
        np.save(pref + "_w_hk.npy", w_hk_arr)
        np.save(pref + "_w_hk_adj.npy", w_hk_adj_arr)
        np.save(pref + "_err_hk.npy", np.array(err_hk, dtype=float))
        np.save(pref + "_err_hk_adj.npy", np.array(err_hk_adj, dtype=float))
        if alphas_use is not None:
            np.save(pref + "_w_alpha.npy", np.array(w_alpha_list, dtype=np.complex128))
            np.save(pref + "_err_alpha.npy", np.array(err_alpha, dtype=float))
        print(f"[saved] npy arrays with prefix: {pref}")

    if args.out_json is not None:
        rep = {
            "args": vars(args),
            "derived": {
                "M": M,
                "Kuse": Kuse,
                "Ncoeff": Ncoeff,
                "target_mobius_used": target_mobius_used,
                "target_mobius_a": None if target_mobius_a is None else [target_mobius_a.real, target_mobius_a.imag],
                "target_mobius_k": None if target_mobius_k is None else [target_mobius_k.real, target_mobius_k.imag],
                "apply_aut_used": aut_a is not None,
                "apply_aut_a": None if aut_a is None else [aut_a.real, aut_a.imag],
                "apply_aut_phi": None if aut_a is None else float(np.angle(aut_k)),
            },
            "stats": {
                "alpha_vs_target": None if alphas_use is None else stats_dict(np.array(err_alpha, dtype=float)),
                "hk_vs_target": stats_dict(np.array(err_hk, dtype=float)),
                "hk_adj_vs_target": stats_dict(np.array(err_hk_adj, dtype=float)),
            },
            "fit_aut": fit_info,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"[saved] json report: {args.out_json}")


if __name__ == "__main__":
    main()
