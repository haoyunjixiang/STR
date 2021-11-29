## 各层 FLOPs 和参数量 paras 的计算
### 参数量
(K*K*Cin)*Cout + Cout  = (K*K*Cin+1)*Cout 
### 计算量
计算量在参数量的基础上再乘以 输出特征图的大小，因为输出的每个点对应参数的计算
(K*K*Cin+1)*Cout*H*W

## 感受野大小的计算
*type* | *size* | *stride* |
:---: | :---: |:--- |
conv1 | 3 | 1 |
pool1 | 3 | 2 |
conv2 | 3 | 1 |
pool2 | 2 | 2 |
conv3 | 3 | 1 |
conv4 | 3 | 1 |
pool3 | 2 | 2 |

从pool3算起，其一个点对应2 * 2，那么conv4的输出为2 * 2的话，conv4的输入为4 * 4，以此类推：
conv3为 6 * 6<br>
pool2为 12 * 12<br>
conv2为 14 * 14<br>
pool1为 28 * 28<br>
conv1为 30 * 30<br>

RF = 1<br>
for layer from (down to top):<br>
&nbsp;&nbsp;&nbsp;&nbsp;RF = (RF-1)*stride + fsize