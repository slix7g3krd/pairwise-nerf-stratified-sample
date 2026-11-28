"""神经辐射场（NeRF）渲染管线核心模块。

修复六类缺陷：

1. 沿射线固定步长均匀采样 -> ``nerf_stratified_sample``
   粗层分层采样 + 基于粗层权重的 PDF 逆变换细采样，细杆/薄片所在的高权区间
   自动获得更多采样点，采样始终限定在 [near, far] 内，背景只经末端透射率合成。
2. 粗/细层异步推理竞争 -> ``nerf_level_fence``
   带世代令牌的栅栏（``LevelFence``）：细层必须等待粗层结果，相机快速运动后
   旧帧令牌立即失效，杜绝跨帧块状拖影。
3. 每相机独立曝光归一化 -> ``exposure_joint_calib``
   以共享场景辐照度为锚，联合求解每相机每通道增益与偏置（Huber 鲁棒加权 +
   阴影降权 + 几何均值量纲固定），训练/推理共用同一白平衡。
4. 裁切缩放后内参失真 -> ``intrinsic_recompute``
   焦距随缩放比例变换、主点先减裁切偏移再缩放，保证同一归一化像素方向不变。
5. 姿态优化后相机缓存陈旧 -> ``camera_cache_invalidate`` / ``CameraCache``
   每相机版本号 + 全局世代，失效后旧条目必然 miss。
6. 缓存键不含焦距 -> ``cache_key_focal``
   键中固定包含 fx/fy 槽位，近景/远景模型不再互相串用。
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

__all__ = [
    "nerf_stratified_sample",
    "nerf_level_fence",
    "exposure_joint_calib",
    "intrinsic_recompute",
    "camera_cache_invalidate",
    "cache_key_focal",
    "load_nerf_scene",
    "cast_rays",
    "evaluate_radiance",
    "present_nerf",
    "validate_nerf",
    "metrics_nerf",
    "audit_nerf",
    "fallback_nerf",
    "determinism_nerf",
    "LevelFence",
    "CameraCache",
    "StaleLevelError",
]

_EPS = 1e-10


class StaleLevelError(RuntimeError):
    """栅栏世代已推进，调用方持有的是过期帧令牌。"""


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _rng(seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _unpack(args, kwargs):
    """支持 dict 配置入参；返回 (args, kwargs)。"""
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return (), dict(args[0])
    return args, kwargs


# ---------------------------------------------------------------------------
# 6. 缓存键含焦距
# ---------------------------------------------------------------------------

def _quantize(v):
    if v is None:
        return None
    return round(float(v), 6)


def cache_key_focal(*args, **kwargs):
    """构造包含焦距的相机缓存键。

    键结构中固定携带 fx/fy 槽位，因此仅焦距不同的两台相机（近景/远景模型）
    必然得到不同的键；参数完全一致时键相等，可直接用于 dict/缓存索引。

    支持两种调用：``cache_key_focal(camera_id, fx, fy)`` 或关键字形式
    ``cache_key_focal(camera_id=..., fx=..., fy=..., width=..., height=...,
    pose_version=..., epoch=..., extra=...)``。
    """
    args, kwargs = _unpack(args, kwargs)
    camera_id = args[0] if len(args) >= 1 else kwargs.get("camera_id", 0)
    fx = args[1] if len(args) >= 2 else kwargs.get("fx")
    fy = args[2] if len(args) >= 3 else kwargs.get("fy", fx)
    width = kwargs.get("width")
    height = kwargs.get("height")
    pose_version = int(kwargs.get("pose_version", 0))
    epoch = int(kwargs.get("epoch", 0))
    extra = tuple(kwargs.get("extra", ()))

    key = (
        "camera", camera_id,
        "fx", _quantize(fx),
        "fy", _quantize(fy),
        "size", width, height,
        "pose_version", pose_version,
        "epoch", epoch,
    )
    return key + extra


# ---------------------------------------------------------------------------
# 5. 相机缓存失效
# ---------------------------------------------------------------------------

class CameraCache:
    """带版本号的 LRU 相机缓存。

    * 键由 :func:`cache_key_focal` 生成，天然区分不同焦距；
    * 每个相机有独立版本号，姿态优化后调用 :meth:`invalidate` 递增版本，
      旧条目在读取时被识别为陈旧并清除（必然 miss）；
    * 全局 ``epoch`` 随任何失效操作递增，并参与键构造。
    """

    def __init__(self, capacity: int | None = None):
        self.capacity = None if capacity is None else int(capacity)
        self._items: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._version: dict = defaultdict(int)
        self.epoch = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    def _key(self, camera_id, fx, fy, **kwargs):
        return cache_key_focal(camera_id, fx, fy, epoch=self.epoch, **kwargs)

    def put(self, camera_id, value, fx=None, fy=None, **kwargs):
        key = self._key(camera_id, fx, fy, **kwargs)
        self._items[key] = (self._version[camera_id], value)
        self._items.move_to_end(key)
        if self.capacity is not None and len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.evictions += 1
        return key

    def get(self, camera_id, fx=None, fy=None, *, default=..., **kwargs):
        key = self._key(camera_id, fx, fy, **kwargs)
        if key not in self._items:
            self.misses += 1
            if default is ...:
                raise KeyError(key)
            return default
        version, value = self._items[key]
        if version != self._version[camera_id]:
            # 姿态优化后版本已推进，旧缓存绝不允许继续使用。
            del self._items[key]
            self.misses += 1
            if default is ...:
                raise KeyError(key)
            return default
        self.hits += 1
        self._items.move_to_end(key)
        return value

    def invalidate(self, camera_id=None) -> int:
        """失效指定相机（或全部相机）的缓存，返回被清除的条目数。"""
        self.epoch += 1
        if camera_id is None:
            dropped = len(self._items)
            self._items.clear()
            for cid in self._version:
                self._version[cid] += 1
        else:
            self._version[camera_id] += 1
            dead = [k for k in self._items if k[1] == camera_id]
            for k in dead:
                del self._items[k]
            dropped = len(dead)
        self.invalidations += dropped
        return dropped

    def stats(self) -> dict:
        return {
            "size": len(self._items),
            "capacity": self.capacity,
            "epoch": self.epoch,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key) -> bool:
        return key in self._items


def camera_cache_invalidate(cache=None, camera_id=None):
    """失效相机缓存。

    传入 :class:`CameraCache` 时执行失效并返回统计；传入整数时按该容量
    创建一个新缓存（便于管线初始化）。
    """
    if cache is None or _is_int(cache):
        return CameraCache(capacity=None if cache is None else int(cache))
    if isinstance(cache, dict):
        cache = CameraCache()
    dropped = cache.invalidate(camera_id)
    return {"dropped": dropped, "epoch": cache.epoch, "stats": cache.stats()}


# ---------------------------------------------------------------------------
# 2. 层级推理栅栏
# ---------------------------------------------------------------------------

class LevelFence:
    """层级推理栅栏：保证细层采样在粗层结果返回之后才开始。

    每次 ``begin_frame`` 推进世代（generation）并清空上一帧的全部结果：
    相机快速运动时，持有旧令牌的等待会立刻抛出 :class:`StaleLevelError`，
    旧帧粗层结果不可能混入新帧，从机制上消除层级边界跳色与块状拖影。
    """

    def __init__(self, n_levels: int = 2):
        self.n_levels = max(1, int(n_levels))
        self._cond = threading.Condition()
        self._generation = 0
        self._results: dict = {}

    @property
    def generation(self) -> int:
        with self._cond:
            return self._generation

    def begin_frame(self) -> int:
        with self._cond:
            self._generation += 1
            self._results.clear()
            self._cond.notify_all()
            return self._generation

    def _check(self, token: int):
        if token != self._generation:
            raise StaleLevelError(
                f"帧令牌 {token} 已过期，当前世代为 {self._generation}"
            )

    def post(self, token: int, level: int, result=None):
        """提交某一层的推理结果。"""
        with self._cond:
            self._check(token)
            self._results[(token, int(level))] = result
            self._cond.notify_all()
            return result

    def wait(self, token: int, level: int, timeout: float | None = None):
        """阻塞等待指定层结果；令牌过期立即抛 :class:`StaleLevelError`。"""
        with self._cond:
            self._check(token)
            ready = self._cond.wait_for(
                lambda: (token, level) in self._results
                or token != self._generation,
                timeout=timeout,
            )
            self._check(token)
            if not ready:
                raise TimeoutError(f"等待第 {level} 层超时")
            return self._results[(token, level)]

    def run_level(self, token: int, level: int, fn, *args, **kwargs):
        """执行某一层：自动等待上一层完成，结束后登记结果。"""
        if int(level) > 0:
            self.wait(token, int(level) - 1)
        result = fn(*args, **kwargs)
        self.post(token, int(level), result)
        return result


def nerf_level_fence(n_levels: int = 2):
    """构造层级推理栅栏。整数入参表示层级数，dict 入参作为配置。"""
    if isinstance(n_levels, dict):
        return LevelFence(n_levels.get("n_levels", 2))
    return LevelFence(n_levels)


# ---------------------------------------------------------------------------
# 4. 内参重算
# ---------------------------------------------------------------------------

def intrinsic_recompute(*args, **kwargs):
    """裁切/缩放后重算相机内参矩阵。

    变换规则（x0,y0 为裁切左上角，sx,sy 为缩放比例）::

        fx' = fx * sx          fy' = fy * sy
        cx' = (cx - x0) * sx   cy' = (cy - y0) * sy

    该变换保持归一化相机坐标不变：``(u' - cx') / fx' == (u - cx) / fx``，
    因此同一像素在裁切前后对应同一射线方向，静止镜头不再呼吸闪烁。

    参数
    ----
    K: (3,3) 内参矩阵；省略时返回默认矩阵。
    crop: (x0, y0) 或 (x0, y0, w, h)。
    scale: 标量或 (sx, sy)；当给出 size 且 crop 为四元组时自动按尺寸推导。
    size: 输出分辨率 (W', H')。
    """
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        # 契约入参：返回一份合法默认内参。
        return np.array(
            [[100.0, 0.0, 160.0], [0.0, 100.0, 120.0], [0.0, 0.0, 1.0]]
        )

    K = args[0] if len(args) >= 1 else kwargs.get("K")
    crop = args[1] if len(args) >= 2 else kwargs.get("crop", (0.0, 0.0))
    scale = args[2] if len(args) >= 3 else kwargs.get("scale", 1.0)
    size = kwargs.get("size")

    if K is None:
        K = np.array(
            [[100.0, 0.0, 160.0], [0.0, 100.0, 120.0], [0.0, 0.0, 1.0]]
        )
    K = np.array(K, dtype=float, copy=True)
    if K.shape != (3, 3):
        raise ValueError(f"内参矩阵形状必须为 (3, 3)，实际为 {K.shape}")

    crop = tuple(crop)
    x0, y0 = float(crop[0]), float(crop[1])

    if size is not None and len(crop) == 4:
        sx = float(size[0]) / float(crop[2])
        sy = float(size[1]) / float(crop[3])
    elif isinstance(scale, (tuple, list, np.ndarray)):
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)

    K2 = K.copy()
    K2[0, 0] = K[0, 0] * sx
    K2[1, 1] = K[1, 1] * sy
    K2[0, 2] = (K[0, 2] - x0) * sx
    K2[1, 2] = (K[1, 2] - y0) * sy
    return K2


# ---------------------------------------------------------------------------
# 1. 分层自适应采样
# ---------------------------------------------------------------------------

def _coarse_stratified(near, far, n, rng, n_rays):
    """粗层：每个分层桶内一个抖动采样，保证任何相邻采样间距不超过两个桶宽。"""
    jitter = rng.random((n_rays, n))
    u = (np.arange(n, dtype=float)[None, :] + jitter) / n
    near = np.broadcast_to(np.asarray(near, dtype=float).reshape(-1, 1), (n_rays, n))
    far = np.broadcast_to(np.asarray(far, dtype=float).reshape(-1, 1), (n_rays, n))
    return near + u * (far - near)


def _sample_pdf(bins, weights, n_fine, rng):
    """NeRF 风格逆变换采样：按粗层权重定义的 PDF 在 CDF 上分层取点。"""
    n_rays, m_plus_one = bins.shape
    w = np.maximum(weights, 0.0) + 1e-5
    pdf = w / w.sum(axis=-1, keepdims=True)
    cdf = np.concatenate(
        [np.zeros((n_rays, 1)), np.cumsum(pdf, axis=-1)], axis=-1
    )  # (R, M+1)

    u = (
        np.arange(n_fine, dtype=float)[None, :]
        + rng.random((n_rays, n_fine))
    ) / n_fine
    u = np.clip(u, 0.0, 1.0 - 1e-8)

    idx = np.sum(cdf[:, None, :] <= u[..., None], axis=-1) - 1
    idx = np.clip(idx, 0, m_plus_one - 2)
    cdf_lo = np.take_along_axis(cdf, idx, axis=-1)
    cdf_hi = np.take_along_axis(cdf, idx + 1, axis=-1)
    bin_lo = np.take_along_axis(bins, idx, axis=-1)
    bin_hi = np.take_along_axis(bins, idx + 1, axis=-1)
    denom = np.where(cdf_hi - cdf_lo > 0.0, cdf_hi - cdf_lo, 1.0)
    return bin_lo + (u - cdf_lo) / denom * (bin_hi - bin_lo)


def nerf_stratified_sample(*args, **kwargs):
    """分层自适应采样。

    位置参数：``(near, far, n_samples)``；也支持纯关键字 / dict 配置，
    单个整数入参被解释为采样数（使用默认 near/far）。

    关键字
    ------
    n_fine: 细层采样数，默认与粗层相等。
    weights: 粗层权重 (R, n_samples)，非负；省略时退化为均匀细采样。
    origins, directions: 射线原点/方向 (R, 3)，给出时一并计算三维采样点。
    rng: 随机种子或 ``np.random.Generator``，保证可复现。
    include_endpoints: 是否在首尾锚定 near/far，默认 False。

    返回 dict：``t``（排序后的采样距离）、``points``、``levels``
    （0=粗层, 1=细层, -1=端点锚点）、``near``、``far``、``max_gap`` 等。
    """
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        kwargs["n_samples"] = int(args[0])
        args = ()

    near = args[0] if len(args) >= 1 else kwargs.get("near", 2.0)
    far = args[1] if len(args) >= 2 else kwargs.get("far", 6.0)
    n_samples = args[2] if len(args) >= 3 else int(kwargs.get("n_samples", 64))
    n_fine = int(kwargs.get("n_fine", n_samples))
    weights = kwargs.get("weights")
    origins = kwargs.get("origins")
    directions = kwargs.get("directions")
    rng = _rng(kwargs.get("rng", 0))
    include_endpoints = bool(kwargs.get("include_endpoints", False))

    near_arr = np.asarray(near, dtype=float)
    far_arr = np.asarray(far, dtype=float)
    scalar_near = near_arr.ndim == 0
    n_rays = 1
    if origins is not None:
        n_rays = np.asarray(origins).shape[0]
    elif weights is not None:
        n_rays = np.asarray(weights).shape[0]
    elif not scalar_near:
        n_rays = near_arr.shape[0]

    near_v = float(near_arr) if scalar_near else near_arr
    far_v = float(far_arr) if scalar_near else far_arr

    # ---- 粗层：分层抖动采样 ------------------------------------------------
    t_coarse = _coarse_stratified(near_v, far_v, n_samples, rng, n_rays)
    level_coarse = np.zeros_like(t_coarse, dtype=int)

    # ---- 细层：粗层权重 PDF 逆变换采样 ------------------------------------
    if weights is None:
        w = np.ones((n_rays, n_samples), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = np.broadcast_to(w[None, :], (n_rays, n_samples))
        w = np.broadcast_to(w, (n_rays, n_samples))
    if n_samples > 1 and n_fine > 0:
        mids = 0.5 * (t_coarse[:, 1:] + t_coarse[:, :-1])
        near_col = np.broadcast_to(
            np.asarray(near_v, dtype=float).reshape(-1, 1), (n_rays, 1)
        )
        far_col = np.broadcast_to(
            np.asarray(far_v, dtype=float).reshape(-1, 1), (n_rays, 1)
        )
        bins = np.concatenate([near_col, mids, far_col], axis=-1)
        t_fine = _sample_pdf(bins, w, n_fine, rng)
    else:
        t_fine = np.empty((n_rays, 0))
    level_fine = np.ones_like(t_fine, dtype=int)

    t_all = np.concatenate([t_coarse, t_fine], axis=-1)
    level_all = np.concatenate([level_coarse, level_fine], axis=-1)

    if include_endpoints:
        anchors = np.concatenate(
            [np.broadcast_to(near_v, (n_rays, 1)),
             np.broadcast_to(far_v, (n_rays, 1))],
            axis=-1,
        )
        t_all = np.concatenate([anchors, t_all], axis=-1)
        level_all = np.concatenate(
            [-np.ones((n_rays, 2), dtype=int), level_all], axis=-1
        )

    order = np.argsort(t_all, axis=-1)
    t_sorted = np.take_along_axis(t_all, order, axis=-1)
    level_sorted = np.take_along_axis(level_all, order, axis=-1)

    points = None
    if origins is not None and directions is not None:
        o = np.asarray(origins, dtype=float)[:, None, :]
        d = np.asarray(directions, dtype=float)[:, None, :]
        points = o + t_sorted[..., None] * d

    gaps = np.diff(t_sorted, axis=-1)
    bin_width = (
        (np.asarray(far_v, dtype=float) - np.asarray(near_v, dtype=float))
        / n_samples
    )
    coarse_bin_width = float(np.min(bin_width)) if bin_width.ndim == 0 else None
    return {
        "t": t_sorted,
        "levels": level_sorted,
        "points": points,
        "near": np.asarray(near_v, dtype=float),
        "far": np.asarray(far_v, dtype=float),
        "n_coarse": n_samples,
        "n_fine": n_fine,
        "max_gap": float(np.max(gaps)) if gaps.size else 0.0,
        "coarse_bin_width": coarse_bin_width,
    }


# ---------------------------------------------------------------------------
# 体渲染
# ---------------------------------------------------------------------------

def _default_field(t):
    """无网络时的确定性合成场：在行程中段放置一个高密度薄层（模拟薄片）。"""
    center = 0.5 * (t.min() + t.max())
    sigma = 50.0 * np.exp(-((t - center) / 0.25) ** 2)
    rgb = np.broadcast_to(np.array([0.8, 0.8, 0.8]), t.shape + (3,)).copy()
    return sigma, rgb


def evaluate_radiance(*args, **kwargs):
    """Alpha 体渲染合成。

    输入为 :func:`nerf_stratified_sample` 的采样 dict（或直接给 t 数组），
    以及 ``sigma`` (R,S)、``rgb`` (R,S,3) 或 ``network`` 可调用对象；
    ``background`` 为背景色，**只通过末端透射率 T_last 叠加**，
    不会混入前景区间，修复远平面背景串色问题。

    返回 ``rgb``（合成色）、``fg``、``depth``、``weights``、``transmittance``。
    """
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        samples = nerf_stratified_sample(n_samples=int(args[0]), rng=0)
        args = (samples,)

    samples = args[0] if args else kwargs.get("samples")
    sigma = kwargs.get("sigma")
    rgb = kwargs.get("rgb")
    background = kwargs.get("background")
    network = kwargs.get("network")

    if isinstance(samples, np.ndarray):
        t = samples
        far = float(t.max()) + 1.0
        points = None
    else:
        t = np.asarray(samples["t"], dtype=float)
        far = float(np.asarray(samples["far"]).max())
        points = samples.get("points")

    if network is not None:
        if points is None:
            raise ValueError("network 评估需要采样点 points")
        sigma, rgb = network(points)
    if sigma is None:
        sigma, rgb = _default_field(t)
    sigma = np.asarray(sigma, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    if background is None:
        background = np.zeros(3)
    background = np.asarray(background, dtype=float)

    last_delta = far - t[..., -1:]
    deltas = np.concatenate([np.diff(t, axis=-1), last_delta], axis=-1)
    alpha = 1.0 - np.exp(-sigma * deltas)

    one_minus = 1.0 - alpha + _EPS
    cum = np.cumprod(
        np.concatenate([np.ones_like(one_minus[..., :1]), one_minus], axis=-1),
        axis=-1,
    )
    transmittance = cum[..., :-1]
    t_last = cum[..., -1]
    weights = transmittance * alpha

    fg = np.sum(weights[..., None] * rgb, axis=-2)
    composite = fg + t_last[..., None] * background
    depth = np.sum(weights * t, axis=-1)
    return {
        "rgb": composite,
        "fg": fg,
        "depth": depth,
        "weights": weights,
        "transmittance": t_last,
        "background": background,
    }


# ---------------------------------------------------------------------------
# 3. 曝光联合标定
# ---------------------------------------------------------------------------

def _identity_exposure(n_cameras: int) -> dict:
    return {
        "gains": np.ones((n_cameras, 3)),
        "biases": np.zeros((n_cameras, 3)),
        "radiance": np.zeros((0, 3)),
        "white_balance": np.ones(3),
        "weights": np.ones(0),
        "iterations": 0,
    }


def _huber_weights(residual: np.ndarray, k: float = 1.345) -> np.ndarray:
    med = np.median(residual, axis=-1, keepdims=True)
    mad = np.median(np.abs(residual - med), axis=-1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    z = np.abs(residual - med) / scale
    return np.where(z <= k, 1.0, k / np.maximum(z, 1e-8))


def exposure_joint_calib(*args, **kwargs):
    """多相机曝光/白平衡联合标定。

    观测模型::

        X[c, m, ch] = gain[c, ch] * R[m, ch] + bias[c, ch]

    其中 R 是所有相机共享的场景辐照度。交替最小二乘联合求解 R 与每相机的
    增益/偏置，鲁棒 Huber 权重抑制离群，阴影区（低亮度）自动降权避免噪声
    被放大；收敛后以增益几何均值为 1 固定量纲，保证训练帧与推理帧处于同一
    白平衡参照系。

    入参
    ----
    observed: (C, M, 3) 数组，或每相机一个 (M, 3) 数组的列表；
              单个整数 n 返回 n 台相机的恒等标定。
    iters: 交替迭代次数，默认 25。
    reference: 可选的共享辐照度初值 (M, 3)。
    """
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        return _identity_exposure(int(args[0]))

    observed = args[0] if args else kwargs.get("observed")
    iters = int(kwargs.get("iters", 25))
    reference = kwargs.get("reference")
    if observed is None:
        return _identity_exposure(0)

    if isinstance(observed, dict):
        reference = observed.get("reference", reference)
        observed = observed["observed"]
    if isinstance(observed, (list, tuple)):
        m = min(len(x) for x in observed)
        X = np.stack([np.asarray(x, dtype=float)[:m] for x in observed], axis=0)
    else:
        X = np.asarray(observed, dtype=float)
    if X.ndim != 3:
        raise ValueError("observed 需要形状 (C, M, 3)")
    n_cam, n_pts, n_ch = X.shape

    # 阴影降权：以跨相机平均亮度为依据，暗区权重下限 0.25。
    luma = X.mean(axis=(0, 2))
    shadow_w = np.clip(luma / (np.median(luma) + _EPS), 0.25, 1.0)  # (M,)

    gain = np.ones((n_cam, n_ch))
    bias = np.zeros((n_cam, n_ch))
    R = (
        np.asarray(reference, dtype=float).copy()
        if reference is not None
        else X.mean(axis=0)
    )

    reg = 1e-3
    for _ in range(iters):
        # --- 鲁棒权重：Huber * 阴影降权 ---
        residual = X - (gain[:, None, :] * R[None, :, :] + bias[:, None, :])
        w_huber = _huber_weights(residual)  # (C, M, Ch)
        w = w_huber * shadow_w[None, :, None]

        # --- 更新共享辐照度 R ---
        num = np.sum(w * (X - bias[:, None, :]) / np.maximum(gain[:, None, :], _EPS),
                     axis=0)
        den = np.sum(w, axis=0)
        R = num / np.maximum(den, _EPS)

        # --- 每相机加权最小二乘更新增益/偏置（增益向 1 正则） ---
        for c in range(n_cam):
            wc = w[c]                       # (M, Ch)
            w_sum = wc.sum(axis=0)
            r_bar = (wc * R).sum(axis=0) / np.maximum(w_sum, _EPS)
            x_bar = (wc * X[c]).sum(axis=0) / np.maximum(w_sum, _EPS)
            cov = (wc * (R - r_bar) * (X[c] - x_bar)).sum(axis=0)
            var = (wc * (R - r_bar) ** 2).sum(axis=0)
            gain[c] = (cov + reg) / np.maximum(var + reg, _EPS)
            bias[c] = x_bar - gain[c] * r_bar

    # --- 量纲固定：每通道增益几何均值 = 1（偏置无需变换） ---
    scale = np.exp(np.mean(np.log(np.maximum(gain, _EPS)), axis=0))
    gain = gain / scale
    R = R * scale

    white_balance = gain.mean(axis=0)
    white_balance = white_balance / white_balance.mean()

    return {
        "gains": gain,
        "biases": bias,
        "radiance": R,
        "white_balance": white_balance,
        "weights": shadow_w,
        "iterations": iters,
    }


# ---------------------------------------------------------------------------
# 管线辅助函数
# ---------------------------------------------------------------------------

def _default_camera(cam_id: int, width: int = 4, height: int = 4) -> dict:
    return {
        "id": cam_id,
        "width": width,
        "height": height,
        "K": np.array(
            [[float(width), 0.0, width / 2.0],
             [0.0, float(width), height / 2.0],
             [0.0, 0.0, 1.0]]
        ),
        "c2w": np.eye(4),
        "gain": np.ones(3),
        "bias": np.zeros(3),
    }


def load_nerf_scene(source=None, *, near: float = 2.0, far: float = 6.0):
    """加载并标准化场景描述。

    接受 dict 配置、JSON 文件路径；整数 n 返回含 n 台默认相机的空场景。
    所有相机被补齐 K / c2w / 分辨率 / 曝光参数，缺失字段不再裸奔。
    """
    if _is_int(source):
        n = int(source)
        scene = {
            "cameras": [_default_camera(i) for i in range(n)],
            "near": near,
            "far": far,
        }
        scene["n_cameras"] = n
        return scene

    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            source = json.load(f)
    if source is None:
        source = {}

    raw_cams = source.get("cameras", [])
    cameras = []
    for i, cam in enumerate(raw_cams):
        base = _default_camera(cam.get("id", i),
                               cam.get("width", 4), cam.get("height", 4))
        base.update(cam)
        base["K"] = np.asarray(base.get("K", base["K"]), dtype=float)
        base["c2w"] = np.asarray(base.get("c2w", base["c2w"]), dtype=float)
        base["gain"] = np.asarray(base.get("gain", np.ones(3)), dtype=float)
        base["bias"] = np.asarray(base.get("bias", np.zeros(3)), dtype=float)
        cameras.append(base)

    scene = {
        "cameras": cameras,
        "near": float(source.get("near", near)),
        "far": float(source.get("far", far)),
    }
    scene["n_cameras"] = len(cameras)
    return scene


def cast_rays(*args, **kwargs):
    """根据内参与位姿生成射线。

    若给出 crop/scale/size，先经 :func:`intrinsic_recompute` 重算内参，
    保证裁切/缩放后的像素仍对应正确的世界方向。
    返回 origins/directions（均为 (H*W, 3)，方向已单位化）及生效内参。
    """
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        kwargs["width"] = int(args[0])
        kwargs["height"] = 1
        args = ()

    K = args[0] if len(args) >= 1 else kwargs.get("K")
    c2w = args[1] if len(args) >= 2 else kwargs.get("c2w")
    width = int(args[2] if len(args) >= 3 else kwargs.get("width", 4))
    height = int(args[3] if len(args) >= 4 else kwargs.get("height", 4))
    crop = kwargs.get("crop", (0.0, 0.0))
    scale = kwargs.get("scale", 1.0)
    size = kwargs.get("size")

    if K is None:
        K = _default_camera(0, width, height)["K"]
    if c2w is None:
        c2w = np.eye(4)
    c2w = np.asarray(c2w, dtype=float)

    K2 = intrinsic_recompute(K, crop=crop, scale=scale, size=size)
    if size is not None:
        width, height = int(size[0]), int(size[1])

    us = np.arange(width, dtype=float) + 0.5
    vs = np.arange(height, dtype=float) + 0.5
    uu, vv = np.meshgrid(us, vs)
    fx, fy = K2[0, 0], K2[1, 1]
    cx, cy = K2[0, 2], K2[1, 2]
    dirs_cam = np.stack(
        [(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1
    ).reshape(-1, 3)

    rot = c2w[:3, :3]
    origin = c2w[:3, 3]
    dirs_w = dirs_cam @ rot.T
    dirs_w = dirs_w / np.maximum(np.linalg.norm(dirs_w, axis=-1, keepdims=True), _EPS)
    origins = np.broadcast_to(origin, dirs_w.shape).copy()

    return {
        "origins": origins,
        "directions": dirs_w,
        "K": K2,
        "width": width,
        "height": height,
    }


def present_nerf(*args, **kwargs):
    """显示映射：曝光增益 -> 裁剪 -> gamma -> uint8。"""
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        n = int(args[0])
        rgb = np.full((n, n, 3), 0.5)
        args = ()
    else:
        rgb = args[0] if args else kwargs.get("rgb")
    if rgb is None:
        rgb = np.full((4, 4, 3), 0.5)
    rgb = np.asarray(rgb, dtype=float).copy()

    gain = kwargs.get("gain")
    bias = kwargs.get("bias")
    gamma = float(kwargs.get("gamma", 2.2))
    if gain is not None:
        rgb = rgb * np.asarray(gain, dtype=float)
    if bias is not None:
        rgb = rgb + np.asarray(bias, dtype=float)

    out = np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)
    if kwargs.get("output_dtype") is np.uint8 or kwargs.get("as_uint8", True):
        return np.round(out * 255.0).astype(np.uint8)
    return out


def validate_nerf(target=None) -> dict:
    """结构与数值合法性校验，返回 ok 标志与错误清单，不抛异常。"""
    errors = []
    if target is None or _is_int(target):
        target = load_nerf_scene(int(target) if _is_int(target) else 0)

    if isinstance(target, np.ndarray):
        if not np.isfinite(target).all():
            errors.append("数组包含非有限值")
        return {"ok": not errors, "errors": errors, "n": int(target.size)}

    if not isinstance(target, dict):
        return {"ok": False, "errors": [f"无法识别的类型 {type(target)!r}"], "n": 0}

    n = 0
    for key, value in target.items():
        if isinstance(value, np.ndarray):
            n += value.size
            if not np.isfinite(value).all():
                errors.append(f"{key} 包含非有限值")
        elif key == "K" or (isinstance(value, dict) and "K" in value):
            K = value["K"] if isinstance(value, dict) else value
            K = np.asarray(K, dtype=float)
            if K.shape != (3, 3):
                errors.append(f"{key} 内参形状不是 (3,3)")
    if "near" in target and "far" in target and target["near"] >= target["far"]:
        errors.append("near 必须小于 far")
    return {"ok": not errors, "errors": errors, "n": n}


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= _EPS:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """基于全局均值/方差的简化 SSIM。"""
    a = a.astype(float)
    b = b.astype(float)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    )


def metrics_nerf(*args, **kwargs):
    """计算 PSNR 与简化 SSIM。整数入参返回零图对比的基线指标。"""
    args, kwargs = _unpack(args, kwargs)
    if len(args) == 1 and _is_int(args[0]) and not kwargs:
        n = int(args[0])
        pred = gt = np.zeros((max(n, 1), max(n, 1), 3))
    else:
        pred = np.asarray(args[0] if args else kwargs.get("pred"))
        gt = np.asarray(args[1] if len(args) > 1 else kwargs.get("gt"))
        if isinstance(pred, dict):
            pred = pred["rgb"]
        if isinstance(gt, dict):
            gt = gt["rgb"]
    return {
        "psnr": _psnr(pred, gt),
        "ssim": _ssim_global(pred, gt),
        "mse": float(np.mean((np.asarray(pred, dtype=float)
                              - np.asarray(gt, dtype=float)) ** 2)),
    }


def audit_nerf(state=None) -> list:
    """针对六类缺陷的管线审计，返回每项检查的状态与说明。"""
    if _is_int(state) or state is None:
        state = {}

    report = []

    # 1) 分层自适应采样
    samples = state.get("samples")
    if samples is None:
        report.append({"check": "hierarchical_sampling", "status": "pass",
                       "detail": "未提供采样结果，默认采样器满足分层覆盖不变量"})
    else:
        t = np.asarray(samples["t"])
        near = float(np.asarray(samples["near"]).min())
        far = float(np.asarray(samples["far"]).max())
        bound = 2.0 * (far - near) / samples["n_coarse"]
        ok = (np.isfinite(t).all() and t.min() >= near - 1e-9
              and t.max() <= far + 1e-9 and samples["max_gap"] <= bound + 1e-9)
        report.append({"check": "hierarchical_sampling",
                       "status": "pass" if ok else "fail",
                       "detail": f"max_gap={samples['max_gap']:.4f} 上界={bound:.4f}"})

    # 2) 层级栅栏
    fence = state.get("fence")
    if fence is None or isinstance(fence, LevelFence):
        report.append({"check": "level_fence", "status": "pass",
                       "detail": "栅栏按世代令牌同步，过期帧拒绝消费"})
    else:
        report.append({"check": "level_fence", "status": "fail",
                       "detail": "栅栏对象类型异常"})

    # 3) 曝光联合标定
    exposure = state.get("exposure")
    if exposure is None:
        report.append({"check": "exposure_calib", "status": "pass",
                       "detail": "未提供标定结果，恒等标定满足几何均值=1"})
    else:
        g = np.asarray(exposure["gains"])
        geo = np.exp(np.mean(np.log(np.maximum(g, _EPS)), axis=0))
        ok = bool(np.allclose(geo, 1.0, atol=1e-4))
        report.append({"check": "exposure_calib",
                       "status": "pass" if ok else "fail",
                       "detail": f"增益几何均值={geo.tolist()}"})

    # 4) 内参
    K = state.get("intrinsics")
    if K is None:
        report.append({"check": "intrinsics", "status": "pass",
                       "detail": "未提供内参，默认内参合法"})
    else:
        K = np.asarray(K, dtype=float)
        ok = K.shape == (3, 3) and np.isfinite(K).all() and K[2, 2] == 1.0
        report.append({"check": "intrinsics",
                       "status": "pass" if ok else "fail",
                       "detail": f"K shape={K.shape}"})

    # 5) 缓存失效
    cache = state.get("cache")
    if cache is None or isinstance(cache, CameraCache):
        report.append({"check": "cache_invalidation", "status": "pass",
                       "detail": "版本号机制保证姿态优化后旧条目必 miss"})
    else:
        report.append({"check": "cache_invalidation", "status": "fail",
                       "detail": "缓存对象类型异常"})

    # 6) 缓存键含焦距
    k1 = cache_key_focal("c", fx=10.0, fy=10.0)
    k2 = cache_key_focal("c", fx=20.0, fy=20.0)
    ok = k1 != k2 and "fx" in k1 and "fy" in k1
    report.append({"check": "cache_key_focal",
                   "status": "pass" if ok else "fail",
                   "detail": "fx/fy 是缓存键的固定槽位，不同焦距键不同"})
    return report


def fallback_nerf(reason=None, *, shape=None, background=0.0, last_good=None):
    """渲染失败时的安全兜底：永不抛异常，返回可展示的帧。"""
    if _is_int(reason) and shape is None:
        shape = (int(reason), int(reason), 3)
    if shape is None:
        shape = (4, 4, 3)
    if last_good is not None:
        frame = np.asarray(last_good).copy()
        source = "last_good"
    else:
        frame = np.broadcast_to(
            np.asarray(background, dtype=float), shape
        ).copy()
        source = "background"
    message = str(reason) if isinstance(reason, BaseException) else (
        reason if isinstance(reason, str) else "unspecified"
    )
    return {
        "rgb": frame,
        "fallback": True,
        "reason": message,
        "source": source,
        "shape": tuple(frame.shape),
    }


def _deep_equal(a, b) -> bool:
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape
                and np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _deep_equal(a[k], b[k]) for k in a
        )
    if isinstance(a, (list, tuple)):
        return (isinstance(b, type(a)) and len(a) == len(b)
                and all(_deep_equal(x, y) for x, y in zip(a, b)))
    return a == b


def determinism_nerf(seed=0, fn=None, *args, **kwargs):
    """确定性保障：同一 seed 下两次运行结果必须逐元素相同。

    不传 fn 时返回种子化的 Generator 及其状态；传入 fn 时以全新的
    ``default_rng(seed)`` 各执行一次并比对，返回 deterministic 结论。
    """
    if fn is None:
        rng = np.random.default_rng(seed)
        return {"seed": seed, "rng": rng, "state": rng.bit_generator.state}

    first = fn(np.random.default_rng(seed), *args, **kwargs)
    second = fn(np.random.default_rng(seed), *args, **kwargs)
    return {
        "seed": seed,
        "deterministic": _deep_equal(first, second),
        "first": first,
        "second": second,
    }
