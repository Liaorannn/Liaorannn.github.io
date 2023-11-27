---
layout: post
title: PyTorch Notes
subtitle: some pytorch learning notes
author: Liaoran
categories: Notes
tags:
  - PyTorch
  - notes
---


##### Long Tensor
***
> 专门用来储存整型，可以用在语料的embedding上面

##### View & Reshape & Contiguous
***
View和Reshape本质上都是改变tensor的维度，对tensor进行维度变换，但是在其工作机制上略有不同。

> View本质上是对原始tensor做不同的指针索引。
- 不会复制原始的tensor数据创建新的内存，类似pandas的切片
- 通过更改的访问内存的指针的stride实现切片。即创建新的指针。
- 当目标tensor的**逻辑顺序 不等于 物理内存顺序**时，view无法进行索引功能。【此时 `tensor.is_contiguous == False`】
```python
>>>a = torch.tensor([1,2,3,4,5,6])

>>>b = a.view((3,2))
[[1,2],
[3,4],
[5,6]]

>>>c = b.T  # 转置后逻辑顺序改变
[[1,3,5],
[2,4,6]]
>>> c.view((12, ))
Error !!
```

> Reshape能做到view做不到的。
- 当view能运行时，两者相同
- 当view无法进行时，reshape会强行复制原始的逻辑顺序数据，生成新的连续内存空间，然后再进行view改变形状

> Contiguous

开辟新的内存给tensor，并按照现有的逻辑顺序存放tensor
- **a.reshape = a.view() + a.contiguous().view()**
- 在满足 Tensor 连续性条件时，a.reshape 返回的结果与 a.view() 相同，否则返回的结果与a.contiguous().view() 相同。

Ref:
- [PyTorch 82. view() 与 reshape() 区别详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/436892343)
- [Pytorch: view()和reshape()的区别？他们与continues()的关系是什么？_view和reshape的区别_JacksonKim的博客-CSDN博客](https://blog.csdn.net/qq_40765537/article/details/112471341)

#### Transpose & Permute
---
用来转置或者变换维度


#### Register_buffer
---
用来表示不用更新参数，position encoding中使用
