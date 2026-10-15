"""End to end on a synthetic walk-around: frames -> reconstruction -> dense surface -> tape measurements.

Runs the measurement twice: once on a TSDF fused from the renderer's true depth
and poses (isolates measurement error), once on the full pipeline (adds
reconstruction and Depth-Anything error). Prints both against the truth.

Usage: python tools/eval_body.py --name a3c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.body import dense as dn  # noqa: E402
from mombe.body.measure import measure, schaeffer_kg  # noqa: E402
from mombe.body.recon import Reconstructor  # noqa: E402
from mombe.sim.render import Animal  # noqa: E402
from tools.eval_recon import load, score  # noqa: E402


def gt_surface(d: Path, truth: dict, step: int = 2) -> np.ndarray:
    cv2.ocl.setUseOpenCL(False)
    W, H = 960, 540
    K = np.array(truth["K"]) * 0.5
    K[2, 2] = 1
    vs = cv2.VolumeSettings(cv2.VolumeType_TSDF)
    vs.setVoxelSize(dn.VOXEL_M)
    vs.setVolumeResolution(np.array([round(v / dn.VOXEL_M) for v in dn.VOLUME_M], np.int32))
    vs.setTsdfTruncateDistance(4 * dn.VOXEL_M)
    vs.setCameraIntegrateIntrinsics(K.astype(np.float32))
    vs.setIntegrateWidth(W)
    vs.setIntegrateHeight(H)
    vs.setDepthFactor(1.0)
    vs.setMaxDepth(dn.MAX_DEPTH_M)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [-dn.VOLUME_M[0] / 2, -dn.VOLUME_M[1] / 2, dn.GROUND_CLEAR_M]
    vs.setVolumePose(pose)
    vol = cv2.Volume(cv2.VolumeType_TSDF, vs)
    for f in range(0, len(truth["R"]), step):
        z = cv2.resize(np.load(d / f"depth_{f:02d}.npy").astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        z[z > dn.MAX_DEPTH_M] = 0
        R, t = np.array(truth["R"][f]), np.array(truth["t"][f])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        vol.integrate(z, c2w)
    pts, _ = vol.fetchPointsNormals()
    return pts.reshape(-1, pts.shape[-1])[:, :3].astype(np.float64)


def true_boards(truth: dict) -> list[dict]:
    a = Animal(**{k: tuple(v) if isinstance(v, list) else v for k, v in truth["animal"].items()})
    out = []
    for side in ("left", "right"):
        c, u, v = a.board_pose(side)
        out.append({"centre": c, "normal": np.cross(v, u)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--skip-gt", action="store_true")
    a = ap.parse_args()
    d, truth, frames = load(a.name)
    out = {"name": a.name, "truth": {"heart_girth_m": round(truth["heart_girth_m"], 4),
                                     "body_length_m": round(truth["body_length_m"], 4)}}
    if not a.skip_gt:
        m = measure(gt_surface(d, truth), true_boards(truth))
        out["true_depth"] = {k: m.get(k) for k in ("heart_girth_m", "body_length_m", "withers_height_m")}
        out["true_depth"]["girth_err"] = round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4) if m.get("ok") else m
    t0 = time.time()
    rec = Reconstructor().run(frames, cache=d)
    t1 = time.time()
    dense = dn.DenseFusion().run_stereo(rec, frames)
    m = measure(dense.points, rec.boards)
    out["pipeline"] = {"recon_s": round(t1 - t0, 1), "dense_s": dense.seconds, "scale": rec.scale_detail.get("scale_rel_err"),
                       "focal": rec.stats.get("focal"), "stereo_valid": round(float(np.mean([r["valid"] for r in dense.frames])), 3),
                       **({k: m.get(k) for k in ("heart_girth_m", "body_length_m", "withers_height_m", "girth")} if m.get("ok") else m)}
    if m.get("ok"):
        out["pipeline"]["girth_err"] = round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4)
        out["pipeline"]["schaeffer_kg"] = round(schaeffer_kg(m["heart_girth_m"], m["body_length_m"]), 1)
    rec.feats_kpts = rec.feats_kpts or []
    out["recon_score"] = score(rec, truth)
    np.save(d / "dense_points.npy", dense.points.astype(np.float32))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
