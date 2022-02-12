import torch
import torch.nn as nn

def CELOSS():
    # ##############
    # init data
    # ##############
    input = torch.randn(3, 3,requires_grad=True)
    print("input:",input)
    target=torch.Tensor([0,2,1])
    target=target.to(int)

    # ##############
    # softmax 操作
    # ##############

    # 代码里其实每个model在infer的最后阶段都有这一步
    import torch.nn.functional as F

    # 这里dim的意思是计算Softmax的维度，这里设置dim=1，可以看到每一行的加和为1。
    # 如果设置dim=0，就是一列的和为1。
    sm=nn.Softmax(dim=1)
    softmax_input = sm(input)
    print("softmax_input:",softmax_input)

    # ##############
    # torch.log 操作
    # ##############
    # 对Softmax的结果取自然对数
    # https://pytorch.org/docs/stable/generated/torch.log.html
    log_input = torch.log(softmax_input)
    # Softmax后的数值都在0~1之间，所以ln之后值域是负无穷到0。
    print("log_input:",log_input)

    # ##############
    # 计算 Loss
    # ##############

    # NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值。【这句话其实就是求交叉熵】
    # 假设我们现在Target是[0,2,1]（第一张图片是猫，第二张是猪，第三张是狗）。
    # 第一行取第0个元素，第二行取第2个，第三行取第1个，去掉负号，结果是：2.4001, 0.5097, 0.7649。
    # 再求个均值，结果是 (2.4001 + 0.5097 + 0.7649) / 3 = 1.2249
    # 优化目标是这个3个值要趋于0
    target = torch.tensor([0,2,1])
    loss = nn.NLLLoss()
    output = loss(log_input, target)
    output.backward()
    print("NLLLoss loss:",output)


    # CEloss 将上面的步骤合而为一
    loss = nn.CrossEntropyLoss()
    output = loss(input,target)
    output.backward()
    print("CELoss loss:",output)
