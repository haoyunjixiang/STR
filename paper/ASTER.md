# ASTER解读
基本理论介绍参考[博客](https://blog.csdn.net/u011489887/article/details/120945135)
ASTER的修正模块主要基于STN网络，接下来我们从代码层面来看下STN

## STN
以minist数据集为例，网络代码如下：
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        print("before", x.shape)
        xs = self.localization(x)
        print("after", xs.shape)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        print("theta", theta.shape)
        grid = F.affine_grid(theta, x.size())
        print("grid", grid.shape)
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
self.stn的主要过程如下：
1. 通过CNN获得图像变换参数theta，theta是一个[N,2,3]的tensor
2. F.affine_grid得到的是输出图像对应原图像的坐标位置grid，是[N,h,w,2]的tensor
3. 由于grid得到的并不是整数，F.grid_sample通过采样得到变形后的图像。

pytorch参数初始化：
```python
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
```

paddle参数初始化：
```python
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.set_value(np.zeros(self.fc_loc[2].weight.shape).astype('float32'))
        self.fc_loc[2].bias.set_value(np.asarray([1, 0, 0, 0, 1, 0], dtype='float32'))
```