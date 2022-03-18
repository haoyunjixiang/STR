# 识别方法中的基本策略
*backbone* | *head* | *loss* | *conv* | *activation* |
:---: | :---: |:--- | :---: | :---: |
ReseNet | [CTC](https://blog.csdn.net/u011489887/article/details/120736691) | [CTCLoss](https://github.com/Wanger-SJTU/CTC-loss-introduction/blob/master/ctc_algo.md) |
...   | ...  | CenterLoss | 
MobileNet | Attention | AttentionLoss |



# LOSS
## 交叉熵损失
参考：https://zhuanlan.zhihu.com/p/35709485
1. 二分类
   ![img.png](img/EntropyLoss1.png)
2. 多分类
   ![img.png](img/EntropyLoss2.png)
