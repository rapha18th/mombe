import cv2
import numpy as np

from mombe.vision import liveness


def _views(depth_jitter: float, seed: int = 0, n: int = 300):
    """Keypoints of one surface seen from two camera positions, with matching descriptors."""
    rng = np.random.default_rng(seed)
    K = np.array([[700.0, 0, 320], [0, 700.0, 256], [0, 0, 1]])
    X = np.c_[rng.uniform(-6, 6, n), rng.uniform(-5, 5, n), 30 + rng.uniform(-1, 1, n) * depth_jitter]
    R, _ = cv2.Rodrigues(np.array([0.0, np.radians(4), 0.0]))
    t = np.array([-2.0, 0.3, 0.0])

    def project(P):
        x = P @ K.T
        return (x[:, :2] / x[:, 2:]) + rng.normal(0, 0.3, (n, 2))

    ka, kb = project(X), project(X @ R.T + t)
    desc = rng.normal(0, 1, (n, 128)).astype(np.float32)
    db = desc + rng.normal(0, 0.05, desc.shape).astype(np.float32)
    return ka.astype(np.float32), desc, kb.astype(np.float32), db


def test_plane_is_explained_by_a_homography():
    p = liveness.plane_parallax(*_views(depth_jitter=0.0))
    assert p["available"] and p["score"] < 0


def test_depth_prefers_the_fundamental_matrix():
    p = liveness.plane_parallax(*_views(depth_jitter=3.0))
    assert p["available"] and p["score"] > 0
    assert p["score"] > liveness.plane_parallax(*_views(depth_jitter=0.0))["score"]


def test_too_few_matches_leave_parallax_unavailable():
    ka, da, kb, db = _views(depth_jitter=3.0, n=20)
    assert not liveness.plane_parallax(ka, da, kb, db)["available"]


def test_replayed_keypoints_are_twins():
    ka, da, kb, db = _views(depth_jitter=3.0)
    same = liveness.twin_stats(ka, da, ka.copy(), da.copy())
    other = liveness.twin_stats(ka, da, kb, db)
    assert same["twin_distance"] < 1e-3 and other["twin_distance"] > 0.05


def test_screen_grid_raises_a_spectral_peak():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.normal(128, 40, (600, 700)).astype(np.float32), (0, 0), 2.0)
    yy, xx = np.mgrid[0:600, 0:700]
    grid = tex + 12 * np.sin(2 * np.pi * (0.21 * xx + 0.03 * yy))
    as_bgr = lambda g: cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    plain, screen = liveness.moire_cue(as_bgr(tex)), liveness.moire_cue(as_bgr(grid))
    assert screen["peak"] > plain["peak"] + 1.0
    assert abs(screen["peak_cycles_per_px"] - np.hypot(0.21, 0.03)) < 0.02


def test_glints_spread_over_a_wet_nose_and_vanish_on_paper():
    rng = np.random.default_rng(2)
    wet = np.clip(rng.normal(110, 20, (480, 640, 3)), 0, 255).astype(np.uint8)
    for x, y in zip(rng.integers(10, 630, 60), rng.integers(10, 470, 60)):
        cv2.circle(wet, (int(x), int(y)), 2, (255, 255, 255), -1)
    paper = (wet.astype(np.float32) * 0.8 + 20).astype(np.uint8)    # paper white caps the highlights
    a, b = liveness.glint_cue([wet, wet]), liveness.glint_cue([paper, paper])
    assert a["score"] >= 12 and b["score"] == 0


def test_fusion_names_the_most_suspicious_cue():
    live = {"parallax": list(np.linspace(0, 2, 101)), "moire": list(np.linspace(-3, -1, 101))}
    good = {"parallax": {"available": True, "score": 1.0}, "moire": {"score": -2.0}}
    flat = {"parallax": {"available": True, "score": -1.0}, "moire": {"score": -2.0}}
    none = {"parallax": {"available": False}, "moire": {"score": -2.0}}
    for fusion in ("fisher", "min"):
        calib = {"cues": ["parallax", "moire"], "threshold": 0.02, "live": live, "fusion": fusion}
        assert liveness.fuse(good, calib)[0] > 0.2
        s, worst, _ = liveness.fuse(flat, calib)
        assert s < 0.02 and worst == "parallax"
        assert liveness.fuse(none, calib)[0] == 0.0


def test_two_weak_signs_add_up_under_fisher():
    live = {"parallax": list(np.linspace(0, 1, 101)), "glint": list(np.linspace(0, 1, 101))}
    weak = {"parallax": {"available": True, "score": 0.08}, "glint": {"score": 0.08}}
    fisher = liveness.fuse(weak, {"cues": ["parallax", "glint"], "live": live, "fusion": "fisher"})[0]
    single = liveness.fuse(weak, {"cues": ["parallax", "glint"], "live": live, "fusion": "min"})[0]
    assert fisher < single


def test_a_silenced_cue_counts_as_no_evidence():
    live = {"parallax": list(np.linspace(0, 2, 101)), "glint": list(np.linspace(2, 12, 101))}
    calib = {"cues": ["parallax", "glint"], "threshold": 0.02, "live": live, "fusion": "fisher"}
    dry = {"parallax": {"available": True, "score": 1.0}, "glint": {"available": False, "silenced": True, "score": 0.0}}
    s, _, ps = liveness.fuse(dry, calib)
    assert ps["glint"] == 1.0 and s > 0.4
