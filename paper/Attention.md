## RNN
1. 存在的问题： 对于较长的句子，RNN存在梯度消失的问题，为了解决长序列到定长向量的转化造成的信息损失问题引入Attention。
2. attention的改进: 计算encoder和decoder之间的关联性，使得输出会重点关注当前位置相关的隐藏层（权重比较大）。

## attention计算及其种类
### alignment-based attention 
![img.png](../img/attention_model.png)
![img.png](../img/attention.png)
输入 c（context，有的论文写s），y（input，有的地方也写作 h），输出 z。
1. score function:计算相似性.度量环境向量与输入向量的相似性，计算应该关注哪些重点。
   ![img.png](../img/score_function.png)
2. alignment function:获得attention权重, 使用softmax进行归一化
   ![img.png](../img/alignment.png)
3. generate context vector function: 根据attention权重，计算输出
   ![img.png](../img/generate_vector.png)

### memory-based attention
该模式是QKV模型，Q是输入，KV是Key-Value形式存储的上下文，Q是新来的问题与已存的Key进行相似度对比。感觉在Q&A任务中，这种设置比较合理。Transform采用的这种方式。
![img.png](../img/QKV.png)
1. address memory （score function）：
   
   ![img.png](../img/qkv1.png)
2. normalize（alignment function） ：
   
   ![img_1.png](../img/qkv2.png)
3. read content （gen context vector function） ：
   
   ![img_2.png](../img/qkv3.png)

### attention种类
![img.png](../img/attention_kind.png)


