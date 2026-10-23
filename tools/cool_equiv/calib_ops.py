"""Float64 geometry ops on identical inputs, and MAGSAC's sensitivity to float noise.

run: every op gets inputs that are bit-identical on every build (board
observations and LightGlue matches recorded in a bisect dump, or fixed
numpy constants), so any output difference belongs to that op on that build.
It also re-runs findEssentialMat(USAC_MAGSAC) on every sweep_21 pair with the
points nudged by 1e-9 px on the SAME build, which measures how far MAGSAC
amplifies float noise by itself.

compare: which ops differ between two builds' run outputs, and by how much.

Usage:
  python calib_ops.py run --dump DUMP.pkl --out OUT.pkl
  python calib_ops.py compare A.pkl B.pkl [OUT.json]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def run(dump: str, out: str) -> None:
    import cv2
    cv2.setNumThreads(1)
    d = pickle.loads(Path(dump).read_bytes())
    from mombe.vision.board import make_board
    board = make_board()
    obs = d["board_obs"]
    objp = [board.getChessboardCorners()[ids].astype(np.float32) for c, ids in obs.values()]
    imgp = [np.asarray(c, np.float32) for c, _ in obs.values()]
    size = (1920, 1080)
    w, h = size
    K0 = np.array([[0.85 * w, 0, w / 2], [0, 0.85 * w, h / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    o = {"opencv": cv2.__version__}
    rms, K, dist, rv, tv = cv2.calibrateCamera(objp, imgp, size, K0.copy(), np.zeros(5), flags=flags)
    o["calibrateCamera"] = {"rms": rms, "K": K, "dist": dist, "rvecs": np.array(rv), "tvecs": np.array(tv)}
    o0, i0 = objp[0].astype(np.float64), imgp[0].astype(np.float64)
    o["findHomography_lsq"] = cv2.findHomography(o0[:, :2], i0, 0)[0]
    o["solvePnP_iterative"] = np.concatenate([np.ravel(x) for x in cv2.solvePnP(o0, i0, K0, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)[1:]])
    o["solvePnP_sqpnp"] = np.concatenate([np.ravel(x) for x in cv2.solvePnP(o0, i0, K0, None, flags=cv2.SOLVEPNP_SQPNP)[1:]])
    o["solvePnP_ippe"] = np.concatenate([np.ravel(x) for x in cv2.solvePnP(o0, i0, K0, None, flags=cv2.SOLVEPNP_IPPE)[1:]])
    # fixed pose and lens: pure functions of constants
    r0, t0 = np.array([0.1, -0.2, 0.05]), np.array([0.05, -0.1, 1.2])
    distc = np.array([0.05, -0.3, 0, 0, 0])
    pts, J = cv2.projectPoints(o0, r0, t0, K0, distc)
    o["projectPoints"], o["projectPoints_jacobian"] = pts, J
    Rm, RJ = cv2.Rodrigues(r0)
    o["Rodrigues"], o["Rodrigues_jacobian"] = Rm, RJ
    o["Rodrigues_inverse"] = cv2.Rodrigues(np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]]))[0]
    o["undistortPoints"] = cv2.undistortPoints(i0.reshape(-1, 1, 2), K0, distc, P=K0)
    rng = np.random.default_rng(5)
    Jf = rng.standard_normal((120, 9))
    A = Jf.T @ Jf + 1e-3 * np.eye(9)
    b = rng.standard_normal((9, 1))
    o["mulTransposed_f64"] = cv2.mulTransposed(Jf, True)
    o["gemm_f64"] = cv2.gemm(Jf, Jf, 1.0, None, 0.0, flags=cv2.GEMM_1_T)
    for name, flag in [("SVD", cv2.DECOMP_SVD), ("LU", cv2.DECOMP_LU), ("CHOLESKY", cv2.DECOMP_CHOLESKY),
                       ("QR", cv2.DECOMP_QR), ("EIG", cv2.DECOMP_EIG)]:
        o[f"solve_{name}"] = cv2.solve(A, b, flags=flag)[1]
    o["SVDecomp_singular_values"] = cv2.SVDecomp(Jf)[0]
    o["eigen_values"] = cv2.eigen(A)[1]
    o["invert_SVD"] = cv2.invert(A, flags=cv2.DECOMP_SVD)[1]
    o["SVD_40x9_values"] = cv2.SVDecomp(rng.standard_normal((40, 9)))[0]
    M32 = rng.standard_normal((256, 128)).astype(np.float32)
    o["gemm_f32_256x128"] = cv2.gemm(M32, M32, 1.0, None, 0.0, flags=cv2.GEMM_2_T)
    o["exp_f32"] = cv2.exp(rng.standard_normal(4096).astype(np.float32).reshape(64, 64))
    o["exp_f64"] = cv2.exp(rng.standard_normal(4096).reshape(64, 64))
    o["log_f32"] = cv2.log(np.abs(rng.standard_normal(4096)).astype(np.float32).reshape(64, 64) + 1e-3)
    o["sqrt_f64"] = cv2.sqrt(np.abs(rng.standard_normal(4096)).reshape(64, 64))
    o["cartToPolar_f32"] = np.stack(cv2.cartToPolar(*rng.standard_normal((2, 64, 64)).astype(np.float32)))
    # MAGSAC on the recorded sweep_21 pipeline matches: identical inputs on every build
    Kc = np.asarray(d["calib"]["K"], np.float64)
    masks, Es, nudged = {}, {}, {}
    for (i, j), v in sorted(d["lg"].items()):
        if not isinstance(v, dict) or len(v["idx"]) < 16:
            continue
        p1 = d["kpts"][i][v["idx"][:, 0]].astype(np.float32)
        p2 = d["kpts"][j][v["idx"][:, 1]].astype(np.float32)
        E, m = cv2.findEssentialMat(p1, p2, Kc, cv2.USAC_MAGSAC, 0.999, 1.0)
        masks[(i, j)] = None if m is None else m.ravel().astype(bool)
        Es[(i, j)] = E
        # the same build, the points nudged by about one float32 ulp at 1000 px
        E2, m2 = cv2.findEssentialMat(p1 + np.float32(1e-4), p2, Kc, cv2.USAC_MAGSAC, 0.999, 1.0)
        a, bb = masks[(i, j)], (None if m2 is None else m2.ravel().astype(bool))
        nudged[(i, j)] = {"changed": a is None or bb is None or not np.array_equal(a, bb),
                          "inliers": int(a.sum()) if a is not None else 0, "inliers_nudged": int(bb.sum()) if bb is not None else 0,
                          "jaccard": float((a & bb).sum() / max((a | bb).sum(), 1)) if a is not None and bb is not None else 0.0}
    o["magsac_masks"], o["magsac_E"], o["magsac_nudge"] = masks, Es, nudged
    Path(out).write_bytes(pickle.dumps(o))
    ch = sum(1 for v in nudged.values() if v["changed"])
    print("calib_ops", cv2.__version__, "rms", rms, "fx", K[0, 0], "| magsac pairs", len(nudged), "changed by a 1e-4 px nudge:", ch)


def flat(x):
    if isinstance(x, (tuple, list)):
        return np.concatenate([np.asarray(v, np.float64).ravel() for v in x])
    if isinstance(x, dict):
        return np.concatenate([np.asarray(x[k], np.float64).ravel() for k in sorted(x)])
    return np.asarray(x, np.float64).ravel()


def compare(a: str, b: str) -> dict:
    A, B = pickle.loads(Path(a).read_bytes()), pickle.loads(Path(b).read_bytes())
    rows = {}
    for k in A:
        if k in ("opencv", "magsac_masks", "magsac_E", "magsac_nudge"):
            continue
        x, y = flat(A[k]), flat(B[k])
        if x.shape != y.shape:
            rows[k] = {"identical": False, "shape": [x.shape, y.shape]}
            continue
        dmax = float(np.abs(x - y).max()) if len(x) else 0.0
        rows[k] = {"identical": bool(np.array_equal(x, y)), "max_abs": dmax,
                   "max_rel": dmax / (float(np.abs(x).max()) or 1.0), "frac": float((x != y).mean()) if len(x) else 0.0}
    ma, mb = A["magsac_masks"], B["magsac_masks"]
    diff = []
    for p in ma:
        x, y = ma[p], mb.get(p)
        if x is None or y is None or not np.array_equal(x, y):
            j = float((x & y).sum() / max((x | y).sum(), 1)) if x is not None and y is not None else 0.0
            diff.append({"pair": list(p), "inliers_a": int(x.sum()) if x is not None else 0,
                         "inliers_b": int(y.sum()) if y is not None else 0, "jaccard": round(j, 4)})
    nA = A["magsac_nudge"]
    nB = B["magsac_nudge"]
    rows["findEssentialMat_magsac_identical_inputs"] = {
        "identical": not diff, "pairs": len(ma), "pairs_differing": len(diff),
        "median_jaccard_differing": float(np.median([d["jaccard"] for d in diff])) if diff else 1.0,
        "worst": sorted(diff, key=lambda d: d["jaccard"])[:6]}
    rows["magsac_same_build_nudge_1e-4px"] = {
        "pairs": len(nA), "changed_a": sum(1 for v in nA.values() if v["changed"]),
        "changed_b": sum(1 for v in nB.values() if v["changed"]),
        "median_jaccard_changed_a": float(np.median([v["jaccard"] for v in nA.values() if v["changed"]] or [1.0])),
        "median_jaccard_changed_b": float(np.median([v["jaccard"] for v in nB.values() if v["changed"]] or [1.0]))}
    return {"a": A["opencv"], "b": B["opencv"], "ops": rows}


if __name__ == "__main__":
    import json
    import os
    sys.path.insert(0, os.environ.get("MOMBE_CODE", str(Path(__file__).resolve().parents[2])))
    if sys.argv[1] == "run":
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        run(args["--dump"], args["--out"])
    else:
        r = compare(sys.argv[2], sys.argv[3])
        if len(sys.argv) > 4:
            Path(sys.argv[4]).write_text(json.dumps(r, indent=1))
        for k, v in r["ops"].items():
            print(f"{k:40s} {json.dumps(v)[:300]}")
