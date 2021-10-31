import numpy as np

np.random.seed(1111)

T, V = 12, 5 # T表示时间步长，V表示字符集大小（包括blank）
h = 6  # h为隐藏层单元

x = np.random.random([T, h])  # T x h
w = np.random.random([h, V])  # weights, h x V

def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist

def toy_nw(x):
    y = np.matmul(x, w)  # T x n
    y = softmax(y)
    return y

y = toy_nw(x)
print(y)
print(y.sum(1, keepdims=True))
print("以上是 12时间步长，5字符集大小（包括blank）\n")


def forward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = alpha[t - 1, i]
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[t - 1, i - 2]

            alpha[t, i] = a * y[t, s]

    return alpha


labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
alpha = forward(y, labels)
print(alpha)

p = alpha[-1, -1] + alpha[-1, -2]
print("前向计算概率P:"+str(p))


def backward(y, labels):
    T, V = y.shape
    L = len(labels)
    beta = np.zeros([T, L])

    # init
    beta[-1, -1] = y[-1, labels[-1]]
    beta[-1, -2] = y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            a = beta[t + 1, i]
            if i + 1 < L:
                a += beta[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta[t + 1, i + 2]

            beta[t, i] = a * y[t, s]

    return beta


beta = backward(y, labels)
p = beta[0, 0] + beta[0, 1]  # 和alpha得到的结果完全一致
print("后向计算概率P:"+ str(p))


alpha_beta = alpha * beta

for i in range(alpha_beta.shape[0]):
    p = 0.
    for j in range(alpha_beta.shape[1]):
        p += alpha_beta[i, j] / y[i, labels[j]]
    print("利用前向后向计算任意时刻P:"+ str(p))


import cv2
import numpy as np
rgb_img = cv2.imread('../paper/img.png')
HSV = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(HSV)
print(HSV.shape,H.shape)
cv2.imshow("H",H)
cv2.imshow("S",S)
cv2.imshow("V",V)
cv2.waitKey(0)



