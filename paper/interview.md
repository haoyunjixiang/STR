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
   + 使用贪心策略获取最大概率的一条路径
   + 根据最大概率的输出与标签确定对应特征向量
   + 若有重复去重 取后面的特征向量
   + 若预测与标签相同则更新参数，否则跳过centerloss更新
   + 存储的center与预测进行作差，并更新
4. 反卷积上采样具体策略
   + 反卷积 就是元素之间补0后卷积


超参数科技：
1. relu6的作用及公式
2. NMS手撕
3. 小目标的size

深兰2面：
1. GIOU何时退化到普通IOU
2. inception的好处
3. 可变形卷积
4. 别的LOSS函数
5. CTC的缺点粘连问题

商汤one:
1. 手撕softmax crossentroy
2. 推理速度与芯片的关系 roofline model
3. FPN的输入与输出是什么  FPN的通道设计

## 提问面试官
1. 数据与模型
2. 文本行多行文本
3. 未来的发展趋势