## 定义损失反向传播
1. optimizer.zero_grad()  梯度置零，也就是把loss关于weight的导数变成0
2. loss.backward()  反向传播计算当前梯度
3. optimizer.step()  更新参数

## 相关类
### dataset
1. 集成工具 from torchvision import datasets   
2. 集成数据增强 from torchvision import transforms
3. 自定义 Dataset    from torch.utils.data import Dataset
4. 自定义 Dataloader from torch.utils.data import DataLoader

### loss && optim
1. torch.nn.CTCLoss
2. torch.optim.Adam

### 优化器衰减
torch.optim.lr_scheduler 分为三大类
1. 有序调整：等间隔调整(Step)，按需调整学习率(MultiStep，按照指定的阶段衰减)，指数衰减调整(Exponential)和 余弦退火CosineAnnealing。
2. 自适应调整：自适应调整学习率 ReduceLROnPlateau。可以监控精确度或损失不变化时，进行lr更新。
3. 自定义调整：自定义调整学习率 LambdaLR。

经验：
1. SGD + momentum 大的学习率+大的动量（0.05+0.9）  可以有所提升，比肩Adam

### model.train() 和 model.eval()
这个两个函数是对于使用了BN或dropout的时候使用的
1. 对于model.train()  BN会使用每一个批次的均值和方差，dropout会随机丢弃
2. 对于model.eval() BN 会使用整个训练集的均值和方差，dropout不再丢弃

### 微调方法
1. 若新数据集与原来的相似，固定前面的层，只训练最后一层FC
2. 若差异较大，从中间开始训练
3. 微调训练时，使用较小的学习率，可以从原始学习率的0.1倍开始尝试。

### permute函数 和 transpose函数
1. permute可以对任意高维矩阵进行转置
2. 只能操作2D矩阵的转置

当我们在使用transpose或者permute函数之后，tensor数据将会变的不在连续，而此时，我们采用view之前要使用contiguous函数。

### torch.einsum
爱因斯坦求和约定：用于简洁的表示乘积、点积、转置等方法。  C = torch.einsum('ik,kj->ij', A, B)

