## 元初智能
### 初面
1. mobilenet
2. BN
3. BN与GN的差别  （回答的不是很清楚）
4. 一道算法题 达到终点的车辆批次 （手生了，有思路，没写出来）
### CTO面
回答不好的点

1. 端侧部署的详细流程
2. 添加噪声的具体方法
   + ‘gaussian’ Gaussian-distributed additive noise.
   + ‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
   + ‘poisson’ Poisson-distributed noise generated from the data.
   + ‘salt’ Replaces random pixels with 1.
   + ‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
   + ‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signedimages.
   + ‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.

3. centerloss的具体实现
4. 反卷积上采样具体策略

## 提问面试官
1. 数据与模型
2. 文本行多行文本
3. 未来的发展趋势