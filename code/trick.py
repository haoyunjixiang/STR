# 二维数组定义
mylist = [[0] * 2 for _ in range (3)]
print(mylist)

# 二维排序,第一维升序，第二维降序
sorted(mylist,key=(lambda x:[x[0],-x[1]]))

s = "/home/aistudio/data/data10879/test_images/0.jpg	都佳洗衣"
s = s.replace("/home/aistudio/data/data10879/test_images/","")
print(s)

