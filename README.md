# pairwise-nerf-stratified-sample

## 领域

游戏、图形与多媒体

## 主题

神经辐射场分层采样与相机曝光

## 说明

分层自适应采样、层级推理栅栏、曝光联合标定。

初始状态下 `graphics/nerf_render.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复点与实现

纯 numpy 实现，无额外依赖。

| 缺陷 | 修复函数 | 机制 |
| --- | --- | --- |
| 固定步长采样跳过细杆/薄片、远平面背景串色 | `nerf_stratified_sample` + `evaluate_radiance` | 粗层分层抖动采样（相邻间距不超过 2 个桶宽），按粗层权重做 PDF 逆变换细采样，采样始终在 `[near, far]` 内；背景仅经末端透射率 `T_last` 合成 |
| 粗/细层异步推理竞争、快速运动块状拖影 | `nerf_level_fence` / `LevelFence` | 世代令牌栅栏，细层 `wait` 粗层结果；`begin_frame` 推进世代后旧令牌立即抛 `StaleLevelError` |
| 每相机独立曝光归一化导致偏色 | `exposure_joint_calib` | 共享辐照度 + 每相机每通道增益/偏置交替最小二乘，Huber 鲁棒加权、阴影降权，增益几何均值固定为 1 |
| 裁切缩放后主点错误、静止镜头呼吸闪烁 | `intrinsic_recompute` | `fx'=fx·sx`、`cx'=(cx-x0)·sx`，保持归一化像素射线方向不变 |
| 姿态优化后旧相机缓存未失效 | `camera_cache_invalidate` / `CameraCache` | 每相机版本号 + 全局 epoch，失效后旧条目读取即清除，必然 miss |
| 缓存键不含焦距、近远景模型串用 | `cache_key_focal` | 键中固定包含 `fx/fy` 槽位，焦距不同键必不同 |

其余管线函数：`load_nerf_scene`、`cast_rays`（内部调用 `intrinsic_recompute`）、
`present_nerf`、`validate_nerf`、`metrics_nerf`、`audit_nerf`（对六类缺陷逐项审计）、
`fallback_nerf`、`determinism_nerf`。所有入口同时支持位置参数、关键字与 dict 配置，
单个整数入参（契约测试用法）返回合法默认结果。

测试：`tests/test_nerf_render.py`（契约）与 `tests/test_fixes.py`（六个修复的语义测试与端到端冒烟）。

