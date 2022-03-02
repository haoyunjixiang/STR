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
