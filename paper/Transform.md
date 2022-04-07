## Transformer
解释参考：https://blog.csdn.net/longxinchen_ml/article/details/86533005
1. Q K V 的理解，一个词向量（512维）可以分解为Q K V三个向量（64维）
   ![img.png](../img/Transformer-QKV.png)
2. Muti-head的计算
   ![img.png](../img/Muti-head.png)
   每个此词向量与不同的权重矩阵相乘得到多组QKV向量。其好处：
    + 得到多个表示子空间
    + 扩展了模型专注于不同位置的能力
   
3. 位置编码表示序列顺序

4. 自注意力的计算步骤
   + 得到QKV向量： X1与WQ、WK、WV矩阵相乘得到q1,k1,v1
   + 打分：当前单词的查询向量与每个单词的键向量点积来计算
   + 除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)
   + 计算softmax，该分数决定了每个单词对编码当下位置（“Thinking”）的贡献
   + 每个值向量乘以softmax分数
   + 对加权值向量求和


## Vision Transform (VIT)
实现步骤：<br>
![img.png](../img/VIT.png)
1. 图片分块 224 * 224 -->  16 * 16 * 196
2. patch 转化为 embedding  （196 + 1） * 768  1是加了分类
3. 加入位置编码   是一个768的向量与embedding 加和 变为 197 * 768
4. transform处理  多层叠加处理仍然输出 197 * 768
5. 分类处理(两种⽅式，⼀种是使⽤CLS token，另⼀种就是对所有tokens的输出做⼀个平均)

VIT 与 Resnet的差异：两个网络在整合全局信息的能力上存在差异。
1. VIT无论是高层还是低层都是局部和全局信息混杂的。
2. Resnet 更为严格的遵守从局部特征提取到全局特征的过程。

VIT 代码实现：https://github.com/FrancescoSaverioZuppichini/ViT

## Self-attention based Text Knowledge Mining for Text Detection（STKM)
![img.png](../img/STKM.png)
文章的动机：
在Imagenet以及SynthText数据集上预训练，与实际应用的数据集是有gap的。
网络处理步骤：
1. CNN+FPN拿融合特征
2. ASPM获取位置空间信息特征F
3. Self-attention Decoder 解码特征F为文本序列