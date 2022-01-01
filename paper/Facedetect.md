## [MogFace](https://arxiv.org/pdf/2103.11139.pdf)
1. 全局语义有助于较少误报
2. 选择性尺度增强策略  并不是金字塔层匹配的anchor越多效果越好

## [TinaFace](https://arxiv.org/pdf/2011.13183.pdf)
1. 可变形卷积增强检测形变能力
2. inception模块增强多尺度检测（除此之前外，多尺度训练，FPN等方法）
3. IOU-aware 帮助一阶段算法提高分类分数，抑制假阳。
4. Distance-Iou Loss 增加小目标的损失，提高小目标检测性能
![img.png](../img/TinaFace.png)