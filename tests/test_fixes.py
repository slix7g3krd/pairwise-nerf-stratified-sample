"""针对六个修复点的语义测试与端到端管线冒烟测试。"""

import threading

import numpy as np
import pytest

import graphics.nerf_render as mod
from graphics.nerf_render import (
    CameraCache,
    LevelFence,
    StaleLevelError,
    audit_nerf,
    cache_key_focal,
    camera_cache_invalidate,
    cast_rays,
    determinism_nerf,
    evaluate_radiance,
    exposure_joint_calib,
    fallback_nerf,
    intrinsic_recompute,
    load_nerf_scene,
    metrics_nerf,
    nerf_level_fence,
    nerf_stratified_sample,
    present_nerf,
    validate_nerf,
)


# ---------------------------------------------------------------------------
# 1. 分层自适应采样
# ---------------------------------------------------------------------------

class TestStratifiedSample:
    def test_bounds_and_max_gap(self):
        res = nerf_stratified_sample(2.0, 6.0, 32, rng=0)
        t = res["t"]
        assert t.min() >= 2.0 and t.max() < 6.0
        # 粗层分层 + 细层补点：任意相邻采样间距不超过两个粗桶宽，
        # 细杆/薄片不会整体落在两个采样之间。
        assert res["max_gap"] <= 2.0 * (6.0 - 2.0) / 32 + 1e-9

    def test_adaptive_concentrates_near_peak(self):
        rng = np.random.default_rng(1)
        n = 32
        t0 = np.linspace(2.0, 6.0, n)
        weights = np.exp(-((t0 - 5.0) / 0.05) ** 2)  # 接近 5.0 的薄面
        res = nerf_stratified_sample(2.0, 6.0, n, weights=weights, n_fine=64, rng=rng)
        fine = res["t"][res["levels"] == 1]
        # 细层绝大多数采样落在峰值附近区间 [4.5, 5.5]
        ratio = np.mean((fine > 4.5) & (fine < 5.5))
        assert ratio > 0.8

    def test_deterministic_with_seed(self):
        a = nerf_stratified_sample(2.0, 6.0, 16, rng=7)["t"]
        b = nerf_stratified_sample(2.0, 6.0, 16, rng=7)["t"]
        assert np.array_equal(a, b)

    def test_per_ray_near_far_and_points(self):
        R = 4
        origins = np.zeros((R, 3))
        dirs = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (R, 3)).copy()
        near = np.linspace(1.0, 2.0, R)
        far = np.linspace(5.0, 6.0, R)
        res = nerf_stratified_sample(near, far, 12, origins=origins,
                                     directions=dirs, rng=0)
        assert res["points"].shape == (R, 24, 3)
        for r in range(R):
            assert res["t"][r].min() >= near[r] - 1e-9
            assert res["t"][r].max() <= far[r] + 1e-9

    def test_solo_int_arg(self):
        res = nerf_stratified_sample(8)
        assert res["t"].shape[1] == 16  # 粗 8 + 细 8

    def test_background_only_enters_through_terminal_transmittance(self):
        samples = nerf_stratified_sample(0.0, 1.0, 40, rng=0)
        sigma = np.full(samples["t"].shape, 1e4)  # 完全不透明薄层
        rgb = np.ones(samples["t"].shape + (3,))
        bg = np.array([1.0, 0.0, 0.0])
        out = evaluate_radiance(samples, sigma=sigma, rgb=rgb, background=bg)
        # 透射率几乎为 0：背景红无法混入前景
        assert out["transmittance"].max() < 1e-6
        assert np.abs(out["rgb"] - 1.0).max() < 1e-4


# ---------------------------------------------------------------------------
# 2. 层级推理栅栏
# ---------------------------------------------------------------------------

class TestLevelFence:
    def test_fine_waits_for_coarse(self):
        fence = nerf_level_fence(2)
        token = fence.begin_frame()
        order = []

        def fine():
            fence.wait(token, 0)
            order.append("fine")

        t = threading.Thread(target=fine)
        t.start()
        order.append("coarse-before-post")
        fence.post(token, 0, {"ok": True})
        t.join(timeout=2)
        assert fence.wait(token, 0) == {"ok": True}
        assert order == ["coarse-before-post", "fine"]

    def test_stale_token_rejected_on_fast_camera_motion(self):
        fence = LevelFence()
        old = fence.begin_frame()
        fence.begin_frame()  # 相机快速运动，世代推进
        with pytest.raises(StaleLevelError):
            fence.post(old, 0)
        with pytest.raises(StaleLevelError):
            fence.wait(old, 0)

    def test_run_level_ordering(self):
        fence = LevelFence(2)
        token = fence.begin_frame()
        log = []
        fence.run_level(token, 0, lambda: log.append("coarse") or "c")
        fence.run_level(token, 1, lambda: log.append("fine") or "f")
        assert log == ["coarse", "fine"]
        assert fence.wait(token, 1) == "f"


# ---------------------------------------------------------------------------
# 3. 曝光联合标定
# ---------------------------------------------------------------------------

class TestExposureCalib:
    def _synthetic_scene(self):
        rng = np.random.default_rng(3)
        R = rng.uniform(0.15, 0.95, size=(200, 3))
        true_gain = np.array([[1.2, 1.1, 0.95], [0.9, 1.05, 1.15]])
        true_bias = np.array([[0.02, -0.01, 0.0], [0.0, 0.01, 0.03]])
        X = true_gain[:, None, :] * R[None] + true_bias[:, None, :]
        return R, true_gain, true_bias, X

    def test_recovers_gains_and_biases(self):
        R, true_gain, true_bias, X = self._synthetic_scene()
        calib = exposure_joint_calib(X, reference=R, iters=40)
        # 量纲固定只允许一个全局缩放自由度，直接比对标定后的成像重建误差
        rebuilt = calib["gains"][:, None] * calib["radiance"][None] + calib["biases"][:, None]
        assert np.abs(rebuilt - X).max() < 0.02
        # 两台相机的增益比值应被恢复
        ratio = true_gain[0] / true_gain[1]
        ratio_hat = calib["gains"][0] / calib["gains"][1]
        assert np.allclose(ratio, ratio_hat, atol=0.05)

    def test_gain_gauge_geometric_mean_one(self):
        _, _, _, X = self._synthetic_scene()
        calib = exposure_joint_calib(X, iters=20)
        geo = np.exp(np.mean(np.log(calib["gains"]), axis=0))
        assert np.allclose(geo, 1.0, atol=1e-6)

    def test_white_balance_shared_across_cameras(self):
        calib = exposure_joint_calib(3)
        assert calib["white_balance"].shape == (3,)
        assert np.allclose(calib["gains"], 1.0)
        assert np.allclose(calib["biases"], 0.0)

    def test_shadow_downweighted(self):
        _, _, _, X = self._synthetic_scene()
        calib = exposure_joint_calib(X, iters=5)
        assert calib["weights"].shape == (200,)
        assert calib["weights"].min() >= 0.25 - 1e-9


# ---------------------------------------------------------------------------
# 4. 内参重算
# ---------------------------------------------------------------------------

class TestIntrinsicRecompute:
    def test_crop_keeps_ray_direction_invariant(self):
        K = np.array([[100.0, 0.0, 50.0],
                      [0.0, 100.0, 40.0],
                      [0.0, 0.0, 1.0]])
        x0, y0 = 20.0, 10.0
        K2 = intrinsic_recompute(K, crop=(x0, y0), scale=1.0)
        u, v = 35.0, 25.0  # 原图坐标
        u2, v2 = u - x0, v - y0
        d1 = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        d2 = np.array([(u2 - K2[0, 2]) / K2[0, 0], (v2 - K2[1, 2]) / K2[1, 1], 1.0])
        assert np.allclose(d1, d2)

    def test_scale_changes_focal_and_principal(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 100.0
        K[0, 2], K[1, 2] = 60.0, 40.0
        K2 = intrinsic_recompute(K, crop=(0.0, 0.0), scale=0.5)
        assert K2[0, 0] == 50.0 and K2[1, 1] == 50.0
        assert K2[0, 2] == 30.0 and K2[1, 2] == 20.0

    def test_size_derived_scale(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 200.0
        K[0, 2], K[1, 2] = 100.0, 80.0
        K2 = intrinsic_recompute(K, crop=(10, 5, 200, 160), size=(100, 80))
        assert np.isclose(K2[0, 0], 100.0)
        assert np.isclose(K2[0, 2], 45.0)  # (100-10)*0.5
        assert np.isclose(K2[1, 2], 37.5)  # (80-5)*0.5


# ---------------------------------------------------------------------------
# 5/6. 相机缓存失效 + 焦距键
# ---------------------------------------------------------------------------

class TestCameraCache:
    def test_key_contains_focal_and_distinguishes(self):
        k1 = cache_key_focal("camA", fx=50.0, fy=50.0)
        k2 = cache_key_focal("camA", fx=80.0, fy=80.0)
        assert k1 != k2
        flat = [str(x) for x in k1]
        assert "fx" in flat and "fy" in flat
        assert cache_key_focal("camA", fx=50.0, fy=50.0) == k1

    def test_invalidation_forces_miss(self):
        cache = camera_cache_invalidate(4)
        cache.put("camA", {"model": "near"}, fx=50, fy=50)
        assert cache.get("camA", fx=50, fy=50) == {"model": "near"}
        result = camera_cache_invalidate(cache, "camA")
        assert result["dropped"] >= 1
        assert cache.get("camA", fx=50, fy=50, default=None) is None

    def test_focal_models_not_mixed(self):
        cache = CameraCache()
        cache.put("camA", "near-model", fx=50, fy=50)
        cache.put("camA", "far-model", fx=200, fy=200)
        assert cache.get("camA", fx=50, fy=50) == "near-model"
        assert cache.get("camA", fx=200, fy=200) == "far-model"

    def test_lru_capacity(self):
        cache = CameraCache(capacity=2)
        cache.put("a", 1, fx=10)
        cache.put("b", 2, fx=10)
        cache.put("c", 3, fx=10)
        assert cache.get("a", fx=10, default=None) is None
        assert len(cache) == 2

    def test_global_invalidation_bumps_epoch_in_key(self):
        cache = CameraCache()
        cache.put("a", 1, fx=10)
        old_key = cache_key_focal("a", fx=10, epoch=cache.epoch)
        cache.invalidate()
        assert old_key not in cache


# ---------------------------------------------------------------------------
# 端到端管线与辅助函数
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_load_scene_defaults(self):
        scene = load_nerf_scene(2)
        assert scene["n_cameras"] == 2
        for cam in scene["cameras"]:
            assert cam["K"].shape == (3, 3)
            assert cam["c2w"].shape == (4, 4)

    def test_cast_rays_recomputed_intrinsics_after_crop(self):
        scene = load_nerf_scene(1)
        cam = scene["cameras"][0]
        full = cast_rays(cam["K"], cam["c2w"], width=8, height=4)
        cropped = cast_rays(cam["K"], cam["c2w"], width=8, height=4,
                            crop=(4.0, 2.0), scale=1.0)
        # 裁切区域中心的射线方向应与原图对应像素一致
        d_full = full["directions"].reshape(4, 8, 3)
        d_crop = cropped["directions"].reshape(4, 8, 3)
        assert np.allclose(d_crop[0, 0], d_full[2, 4], atol=1e-10)

    def test_end_to_end_render_and_metrics(self):
        scene = load_nerf_scene(1)
        rays = cast_rays(scene["cameras"][0]["K"], width=4, height=4)
        samples = nerf_stratified_sample(scene["near"], scene["far"], 24,
                                         origins=rays["origins"],
                                         directions=rays["directions"], rng=0)
        out = evaluate_radiance(samples, background=np.zeros(3))
        frame = present_nerf(out["rgb"].reshape(4, 4, 3))
        assert frame.dtype == np.uint8 and frame.shape == (4, 4, 3)
        m = metrics_nerf(frame, frame)
        assert m["psnr"] == float("inf")
        assert validate_nerf(scene)["ok"]

    def test_audit_all_pass(self):
        report = audit_nerf()
        assert all(item["status"] == "pass" for item in report)
        assert {item["check"] for item in report} == {
            "hierarchical_sampling", "level_fence", "exposure_calib",
            "intrinsics", "cache_invalidation", "cache_key_focal",
        }

    def test_fallback_never_raises(self):
        out = fallback_nerf(RuntimeError("boom"), shape=(2, 2, 3))
        assert out["fallback"] and out["rgb"].shape == (2, 2, 3)

    def test_determinism(self):
        def run(rng):
            return nerf_stratified_sample(0.0, 1.0, 10, rng=rng)["t"]
        res = determinism_nerf(42, run)
        assert res["deterministic"]
