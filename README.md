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
