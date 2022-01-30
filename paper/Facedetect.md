## [MogFace](https://arxiv.org/pdf/2103.11139.pdf)
1. 全局语义有助于较少误报
2. 选择性尺度增强策略  并不是金字塔层匹配的anchor越多效果越好

## [TinaFace](https://arxiv.org/pdf/2011.13183.pdf)
1. 可变形卷积增强检测形变能力：提高对几何形变的检测能力
2. inception模块增强多尺度检测（除此之前外，多尺度训练，FPN等方法）
3. IOU-aware 帮助一阶段算法提高分类分数，抑制假阳。
4. Distance-Iou Loss 增加小目标的损失，提高小目标检测性能
    + 普通IOU 在候选框和真实框无重叠时没有loss，损失始终为1<br>
      ![img.png](../img/IOU.png)
    + GIOU 增加无重叠时的损失，除以最小box面积<br>
      ![img.png](../img/GIOU.png)
    + DIOU 弥补了GIOU在两者重叠时，损失不变的缺点，除以最小box对角线长度<br>
      ![img.png](../img/DIOU.png)
    + CIOU 解決中心点重合，但宽高比不同时，loss不变的情况。 在Diou的基础上增加影响因子a,v把长宽比的一致性考虑进去。
![img.png](../img/TinaFace.png)
      