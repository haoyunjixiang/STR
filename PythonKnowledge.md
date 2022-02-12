## Python中可变和不可变对象
1. 可变对象是指变量指向的内存块的值是可变的，变量修改后，其指向的内存地址不变。如list dict set等。
2. 不可变对象与上描述相反，如 int  str tuple等。

## tuple、list 和 dict 的区别
1. tuple 不可变元组
2. list 列表
3. dict 字典

## iterables、generator 和 yield的区别
1. iterables是迭代器，for x in y  y是可迭代对象 是保存在内存中的。
2. generator是生成器 不是保存在内存中的, 而是惰性加载的, 也就是你用到它的时候, 它才临时去计算。适用于加载大型数据，节省内存。
```python
mygenerator = (x*x for x in range(3))
for i in mygenerator:
    print(i)
```
3. yield
```python
def createGenerator():
    mylist = (x*x for x range(3))
    for i in mylist:
       yield i

mygenerator = createGenerator() # 创建一个生成器
print(mygenerator) # 生成器就是一个object

for i in mygenerator:
    print(i)

mygenerator = createGenerator()
s = ""
for i in mygenerator:
   s += str(i)
```
上面的代码中当调用 createGenerator() 的时候, 其实方法内的代码并没有运行, 而在 for in 循环访问的时候, 才开始从头计算, 当运行到 yield 的时候返回第一个值, 然后就停下来, 当再次请求数据的时候继续运算直到再次碰到 yield ... 直到没有值可以返回

## python 函数装饰器
把函数传到注解的函数里。
```python
from functools import wraps
 
def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging
 
@logit
def addition_func(x):
   """Do some math."""
   return x + x
 
 
result = addition_func(4)
```
## 迭代器
1. 迭代是python中访问集合元素非常强大的一种方式，不像列表全部加载，而是随用随取，因此节省内存。迭代器
    有两个方法iter()和next()
2. for循环遍历时，会先判断是否是可迭代对象，若是迭代对象返回迭代器，然后next方法取元素

## append 与 extend
1. append 追加一个元素
2. extend 扩展列表，合并列表

## Python 垃圾回收
1. 引用计数为主
2. 分代回收为辅

## Pyhton list dict 添加删除元素
list 添加元素：append,insert
list 删除元素：remove pop del

dict添加元素：d[key] = '' update(dict)
删除元素 ： pop del

"".join([]) list转str

## python 数组筛选
[x for x in list if x>0]

## dict技巧
1. 统计字符串或list 单词频数
   
    from collection import Count
   
    dict(Count(str or list))

2. 可用 == 判断两个 字典是否相等
3. 字典排序
   mydict = {'a':3 , 'b':2 , 'c': 1}
   newdict = sorted(mydict.items(),key = (lambda x:[x[0],x[1]]))
    1. 关键是mydict.items() 转换成元组，得到可迭代的数组
    2. lambda x : [x[0],x[1]]  按照所需排序
## numpy特性
### 写NMS需要以下技巧
1. np.maximum 利用广播机制求最大值
2. array[:,1:2] 获取某一列值，list不具备该写法
3. np.where(array>0.7)[0]  获取符合条件的下标
4. array[iou>threshold] 返回符合条件的value，[]括号内可以传int或bool列表

## 判断二维数组最大值
max(map(max,list))

## 字符串反转
revers(str)  反转字符串 然后判断是否回文