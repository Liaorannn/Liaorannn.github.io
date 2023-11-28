---
layout: post
title: ML notes
subtitle: MAFM course notes
author: Liaoran
tags:
  - MAFM
  - notes
  - ML
categories: Notes
excerpt_image: NO_EXCERPT_IMAGE
---

#### Feb.17
1. MM algorithm: Majoritize and minimize 
   Out Lier and l1 loss function:
   l1损失函数 = 对l2损失函数赋予权重，之后在进行对权重的迭代，为极端值的权重更小，从的减少极端值对参数估计的影响.
   l1损失函数的问题在于无法求导，因此最优化的时候没办法求。 此时找l1的损失函数的RSS对应的lossfunction值，然后找这个点上的l2曲线。

2. High Leverage Point:
   考虑 估计的敏感性：即估计值对真实值变动的敏感性  $ y_hat / y_i $
   指的是 当真实值Y改变一点是，对整体的模型参数估计 以及 之后的Y_hat预测，都有巨大影响
   敏感性 = Hii  总和就是 H矩阵的迹 = tr(X(X_TX)-1XT) = p+1 刚好是模型参数的个数，这就跟之前讲的用模型参数评价flexibility相同
   但并不能完全用模型参数去评价：有可能有的模型参数被限制、更有可能并不是线性模型or sth，但是用上面的求导方式，就不会有这样的问题（任何模型、限制都能用）
   Stein Unbiased Risk Estimate  SURE  用求导总和去衡量模型的Flexibility

   Besides: the Model Flexibility 不仅取决于参数数量，更由输入的数据影响，因为当y=y_hat时，y对y_hat求导后的sum = n，which is 输入的样本数

   > Linear regression model H only depends on X

* * *

#### Feb.24

1. Ridge Regression:  $\hat{\beta} = (X^TX + \lambda I)^{-1} X^TY$
   - 线性回归中逆矩阵不可求解的问题
<br>

2. 回归模型中存在两个高相关性的自变量如何选取：
   - 用模型解决： LASSO 回归，会自动提取出其中一个，但是并不能告诉你它是否是更重要的
   - 因此，当你只关心用模型预测时，可以全都放进去，只是用更多的模型而已，但是当你要知道哪个自变量更好及其原因之类的，你需要更多的信息统计推断（我感觉是ABtest？）
<br>

3. 用几何现象解释当矩阵不存在逆的时候为什么不能求beta：
   - 本质就是多重共线性问题 【multi-co-linearity
<br>

4. Metric Learning: 
   > Metric learning 是学习一个度量相似度的距离函数:相似的目标离得近,不相似的离得远. Deep metric learning目前主要是用网络提取embedding,然后在embedding space用L2-distance来度量距离,见下图. 这两个稍微有点不一样,因为还有很多工作是在找距离函数,而不是L2-distance.
   > 一般来说,DML包含三个部分:特征提取网络来map embedding,一个采样策略来将一个mini-batch里的样本组合成很多个sub-set,最后loss function在每个sub-set上计算loss
<br/>

5. KNN: K=1时，拟合出阶梯式函数（一个样本就是一个簇）；K=n时，是一条直线.
   - KNN 中的K，是**每个集合簇中的data个数！！！**  【不是分组的组数
   - 当模型维数越来越多时，KNN表现效果比lr差，因为变量越多噪声越多，更难决定neighbor
<br/>

6. Logistic Regression:
   - Probit regression / Logistic regression / softmax
   - 损失函数设计： 用极大似然函数推导 == 最小化交叉信息熵
   - 最大交叉信息熵解释：https://zhuanlan.zhihu.com/p/38853901
    > 交叉熵损失函数 Cross Entropy：$$H p(q) = - \sum p(x)\log_{2}^{q(x)}$$
    > - 交叉熵函数是用来衡量两个不同分布p、q的差异程度的函数，交叉熵越大，p、q分布差异越大  【注意有负号
    > - 在二元logistic中解决二分类问题，估计分布是sigmoid作用后的linear regression（即广义线性回归），真实分布是y，带入上面的交叉熵函数后，极小化交叉熵函数，即得到logistic回归的损失函数【恰好和用MLE极大似然估计的得到相同
    > - 【注意】：CrossEntropy 显然可以分类不止二元，
<br/>

7. KL散度、交叉熵
   > 上面提到：在Logistic回归的损失函数设计中，极大似然估计和最小化交叉信息熵本质上是相同的，那么在这里，进一步介绍，在损失函数设计上的一些相关概念。
   - 参考：https://zhuanlan.zhihu.com/p/389179483

<br/>

8. logistic 不收敛：
   - 当数据集完美分割时，logistic回归无法收敛
   - > 因为此时，概率一定为1，sigmoid函数的beita必须趋近于无穷，才能使sigmoid函数作用后的linear regression变成1，因此beta无法得到
   - > 图形解释：即虽然损失函数（极大似然函数）是凸函数，存在最大值，但是只有beta趋近无穷时才能取到，因此此时logistic回归无法收敛。

***
#### Mar.3

1. 判别模型和生成式模型的区别：
   - 判别模型：估计的是条件概率分布p(y|x)
   - 生成式模型：估计的是联合概率分布p(x, y)，因此给定y，能得到x的分布从而得到抽样得到（x, y）
   - https://www.cnblogs.com/Harriett-Lin/p/9621107.html
<br/>
2. LDA:线性判别分析模型  **【推导】**

<br/>

3. Confusion Matrix：
   - 改变判断的 threshold 和 判断classification的优劣
<br/>

4. Concept:
   - False positive rate: 去伪 $P(\hat{y} = 1 | y = 0)$  Type1 Error
   - True negative rate: 去真  $P(\hat{y} = 0 | y = 1)$  Type2 Error
   - Sensitivity: $P(\hat{y} = 1 | y = 1)$
   - Specificity: $P(\hat{y} = 0 | y = 0)$
   - ROC curve: $\frac{Sensitivity}{False\ positive\ rate} = \frac{Sensitivity}{1 - Specificity}$
   - AUC: ROC下的面积
   - 当两个AUC差不多时，总时控制 FPR small，因此总是选择更靠左侧的
   - ROC curve： 由于给定x预测y=1的概率的threshold的变化，导致产生的不同TPR、FPR点，连起来绘制成了ROC曲线
   - 【ROC并不是由于模型的超参数导致的，而是threshold，不要弄混！】
   - 分类标准：AUC、misclassification error(指示函数相加的那个  【1.当label非常不平衡时，用这个不好 2.用分类算法需要迭代时，这个损失函数看不出overfitting现象（因为不是连续函数、并且不是凸函数）】)
<br/>

5. QDA：二次判别分析  **【推导】**

<br/>

6. 【考试】在什么情况背景下，LDA会更好，logistic会更好、QDA更好、blablabla
<br/>

7. K-fold CV ------LOOCV
   - Bias 高， Variance高

***
#### Mar.10

1. Cross Validation:
   - 重要：用cv的同时筛选自变量的话，需要把筛选自变量放在cv里面，不能放在外面，即根据每个train来筛，不能根据总的来筛。【万一晒出来的x不一样咋办？
   - 【**但是用pca的时候就不会哦！**  因为pca只用倒x的数据，不会用到y，所以在用变量相关性大小筛选数据时，会用到y的数据，所以不能在cv外面做这个，只能在里面；而pca只用到x数据，不会用到y 可以在cv外面做。**所以处理数据时只要不事先用到y的数据、信息就可以**】
   - 根据自变量和因变量的相关性，选取自变量进行回归的话，即使本来因变量自变量是不相关的（ground true），但这个操作自然而然也会使原本不相干的因变量和自变量，变得相关【LDA不会有这个问题，logistic会
<br/>

2. Bootstrap：
   - 核心在于：**有放回的** 抽样
   - 即把数据样本当作总体分布

***
#### Mar.17th

1. LOOCV: Leaving one out cross validation
   - K-folder is better than loocv
   - 数学证明：每个i out的训练集残差，和对应的验证集残差具有线性关系
   - 在实际操作时，不用真的做i个for循环，可以直接拿所有的来跑，然后再用线性关系直接估计出LOOCV的残差
<br/>

2. Sheman-Morrison formula
   - Rank-one update
   - $(A + )$...
<br/>

3. Linear Model with regularization
   - 最优子集回归
   - 线性回归中 RSS和 log-likelihood的关系(相等)  用在AIC评估中
   - AIC/BIC谁更好：
      1. 必须要有大样本，来满足这些指标的假设
      2. AIC：会选出更小的training error，当需要用模型来更准确的预测时，用这个
      3. BIC：要选更有影响力的自变量时，用来推断哪个自变量更有效，用BIC，
      4. 实际中不怎用，更多用cv，因为这些评价指标依赖的假设太多了
   - 用CV选best subset回归：
      1.  重点是：每个切出来的K-folder去做回归时，都有个p参数的曲线
      2.  每个folder做p参数的回归后，有参数的loss曲线，然后对k个folder做平均，得到平均的变量个数参数曲线
      3.  选择最小的哪个p作为模型囊括的变量个数
      4.  **问题！！**：万一每个模型中筛选的变量个数一样，但是选出来的不同怎么办：
            - Remark: Best Subset回归，只告诉你最优的模型变量数量大小，并不会告诉你最优的
            - cv后选出来后，应该再用所有的training data 在做一遍best subset，但是这次只做最优参数个数的筛选，同样选出最优的，就是最终的
   - 怎么验证 cv是有效的：用signal-noise-ratio信噪比和真实模型去做
        - 当信噪比过低时，噪声更大，即使最优子集搜索，也无法找到最优的模型
  
4. Forward stepwise selection
   - 从单变量开始，逐步添加自变量回归，直到模型不会提升为止
   - 同样在cross validation里可以用
   - 当信噪比低时，用best subset时，只会搜寻到垃圾信息，对模型提升没帮助，这个时候forward stepwise更好，因为减少了过拟合的可能；
   - 但当信噪比高时，best subset更好，因为利用的信息更多
   - Forward stage wise：lasso\gradient boosting\res-net

5. SNR: signal noise ratio 信噪比十分重要！
   - 有时候做exhaustive searching 并不是很好，学到的垃圾信息过多，容易overfitting


6. Ridge Regression:
   - ...


***
### Mar.24

1. Ridge Regression:
   - 极小化的数学推导！ 矩阵形式 【 有截距 和 无截距 的不同： 无截距时 I 多一个0   **考试考！！自己证一下！**
   - 实际操作中 一般会对x做标准化处理  
   - > 当标准化时：\beta0 不会受\lambda 影响，只有在标准化时，β0才等于y的平均 
   - > 证明：将loss function对β0求导，令它=0，即求使损失函数最小的β0（截距项）的值， 然后利用标准化后x均值=0证明得到β0=y均值

2. 标准化处理对模型预测能力的影响：
   - 注意：用标准化后的数据做出来的模型，不能直接用在预测集上，预测集必须也做标准化？【也不行，当你的测试集样本太少时，你的均值不稳定，μ、σ跟训练集不同
   - 正确的做法：要对每个标准化后得到的β做处理，对变量Xi: βi = βi/σi； β0 = β0 - sum(βi*μi/σi)； 再用得到的新的β直接作用到测试集上
   - > 证明：直接将标准化的β带入minimize的损失函数，然后把y_pred写开来就行  **考试**

3. 怎么求岭回归中的系数β 
   - 直接求矩阵的逆  【 有点太麻烦了，尤其要遍历n多个λ时，要算n个矩阵的逆，太麻烦太耗时了，考虑一种方法能够很快的得到不同λ的值
   - Solve $ \hat{\beta\lambda} = (X^T X + \lambda I)^{-1}X^Ty$
   - 用LDL分解$X^TX$矩阵，就等于$L(D + \Lambda I)^{-1} U^T$，因此，每次λ变化，只用算中间对角阵的λ改变即可，不用再把λ加进去后算一遍整个的逆
   - 【**注意！** 但是当我们没有做标准化时，就不能上面这样，只能对原始XtX+λI直接用chelosky分解来算逆
   - 【这里是直接矩阵求ridge，后面用lasso递归算法求解就是用坐标梯度下降法

4. 怎么选岭回归中的lambda：
   - 最大：选取最大的特征根得100倍
   - 最小：0.01最大特征根   【一般不选0，为了防止XtX有多重共线性无法求逆得情况
   - 再将最小——最大投影到log空间上 -0.5\*log10（最小）—— 5\*log10（最大）
   - 根据CV选出来得最佳得lambda： 不能被直接使用到整个数据集上！！
      - 因为评估得lambda某种程度上受训练集得数量影响，因此你找到得最佳lambda是基于训练集得样本数得，因此不能直接用在整个样本集上去徐连
      - 所以必须再乘一个参数将lambda变到整体样本上，参数就是 整体样本数 / 训练集样本数  【 可以带入loss function检验一下

5. 如何估计岭回归中得自由度：
   - 找H得trace
   - $tr(H) = tr(X(X^TX + \lambda I)^{-1}X^T)$ 作LDL分解，
   - 惩罚项越大，自由度越小
   - Solution Path:
     - 当lambda增加时，回归系数beta减小，不同得系数beta随着lambda得变化得曲线构成这个solution path

**Lasso**
1. **对周频数据来讲，ridge可能比lasso更好**，但也要具体情况具体分析

2. Ridge回归在信噪比低时，更加稳定

3. 虽然Lasso能够选取变量，但是也要考虑自变量之间的相关性，即相关性强时，lasso也有可能选错

4. Lasso 的求解方法： Coordinate descent 坐标下降法
   - 每次只更新一个坐标方向的数据，前后更新的坐标不同
   - 步长 = 学习率 * 该方向的斜率（导数）
   - 当损失函数f不可求导时，坐标下降法不能保证递归到极小值 【因为在不可导点时，更改任意一个坐标都只能增大你的f，只有同时更改两个坐标才可能减小f，所以此时不会迭代了】
   - 应用： **线性回归更新**  **考试！！**

5. Soft Shrinkage Operator
   - 用在lasso 里面，看ppt！！
   - 要分析sparsity?
      - 在这个算子里面，要做循环，如果数据是很稀疏的话，就浪费计算时间（因为有很多0
      - 这时候用 Active-Set Coordinate Descent 对lasso来回归
        - 1.每次循环的时候 先做一遍总的，得到非0的active集合；
        - 2.然后在集合内做coordinate descent的循环，直到收敛；
        - 3.然后回到1，再做一次总的；
        - 4.知道你的active set不变为止
        - 【注意：外层的循环，并不用做到收敛，只用更新beta即可

***
### Mar.31
1. Coordinate Descent for LASSO:
   - 存在Sparsity的问题，计算效率低
   - 提升1： Active Set Approach 
     - 1.每次循环的时候 先做一遍总的，得到非0的active集合；
     - 2.然后在集合内做coordinate descent的循环，直到收敛；
     - 3.然后回到1，再做一次总的；
     - 4.知道你的active set不变为止
     - 【注意：外层的循环，并不用做到收敛，只用更新beta即可
   - 提升2：Warm Start
     - 由不等式推来，一开始直接设置lambda = max(XjY)
     - 实际就是：对 待估参数beta 的初始化，设定成上一个lambda的收敛的系数beta
     - 【即从上一个lambda的收敛的beta 开始做下降递归；注意这里的两次lambda的变化应该越小越好

2. Proximal Gradient Descent
   - gradient method with momentum update

3. LASSO:
   - elastic net
   - group lasso
   - ridge regression
   - 怎么找超参数lambda：
     - 要有一个sequence，
     - lamda = XTy的max j; 然后*0.95递归下去
     - 然后对整个序列用CV去找最优的

4. Relaxed Lasso
   - 先做Lasso；得到beta，并找到非零beta的集合 active set A； 然后只用集合A内的做线性回归，得到A上的OLS beta； 用0.5权重把LASSO和OLS的加起来
   - LASSO用bias，但减少了var，并且对sparsity有改善，但用relax后，减少了lasso的bias。

5. PVE： percentage of variance explained = Var(x*beta) / Var(y)
6. SNR: signal noise ratio  = Var(x*beta) / Var(epsilon)

7. Degrees of Freedom:
   - 当假设是noise高斯正态分布时：
   - $df = \sum_{i=1}^n=  \frac{1}{\sigma^2} \sum_i{Cov(\hat{y_i}y_i)}$
   - 有啥用呢？

8. Ridge regression：
   - data augment
   - [Ridge Regularizaton: an Essential Concept in Data Science](https://arxiv.org/abs/2006.00371)

9. Tree Method:
    - [Why do tree method still outperform deep-learning](https://arxiv.org/abs/2207.08815)

***
### April. 14

1. Tree Model
   - How to Grow a Tree:
     - Loss Function: ...
     - 在每个维度上都要把所有的spilt point（即样本数的中间值）测试一遍，然后看那个对损失函数的减少最多（最小）选哪个
   - Why we need Tree method:
     - 为了解决interaction问题
   - Fit Tree: 1.找分割点（用loss function） 2.找预测值（Tree 是piecewise constant function）
   - When to stop:
     - 如果仅用生成tree前后的loss function变化是否充分，用可能这一步停止了，但其实下一步的变化还是很大，因此
     - 为了避免grow不充分的情况，因此先尽量生成足够多的tree，然后来剪枝！！
     - 怎么剪： min LOSS + Penalty 【 LOSS + alpha * |T|  T:中间结点的数量（子树的数量）】
     - 怎么判断充分enough： 每个叶子结点下有不少于threshold的样本数
   - 现在实际不剪枝了：
     - 因为单一的树是不精准的，因此此时用 ensemble 的tree: 
       - 1.Random forest: Large Tree  
       - 2.Gradient boost: Small Tree  用深度为1、2（最正常用4、6）的tree做迭代
         - Gradient Tree中，利用深度为1的tree具有可加性的特点，去拟合additive model
   - **Training Error总是在降低**，当你的**Tree越来越大的**时候   【跟Linear Regression中变量越多同理

2. Classification Tree:
   - 和regression tree唯一的不同就是Loss function
   - Regression Tree： 用的是square loss function  $$
   - Classification Tree:
     - 1.Classification error  *NOT GOOD* 【因为有max所以并不连续
     - 2.Gini index 
     - 3.cross-entropy
   - 分类的时候：用叶子结点中最多的分类当作该结点的预测分类

3. 什么时候用Tree什么时候用Linear:
   - 用cv做一下就好了

4. Tree Model的优缺点：
   - 优点
     - 1.运行速度  O(p nlogn) 是p*n的线性，运行时间少
     - 2.能拟合非线性的问题
     - 3.可解释性强（也就那样吧
   - 缺点：
     - 1.精准度低  
       - 单一树的预测精准度不高，因为tree结构不稳定，太吃training dataset了
       - 当拿出一两个数据时，整个树结构都有可能变化，不像LR，少一两个数据也很稳定
       - 并且在CV的时候，每个folder做出来的tree都可能不一样
       - 因此 这也影响了可解释性；
       - 并且增加了Variance

5. Tree piece-wise constant
   - 为什么一定要piece-wise而不在分类中再用个linear regression？
     - 因为树模型中，真正起作用的是你的分类点，你的split point，因此每个分类下加个LR并不会改善你的tree
     - 并且加个LR只会增加你的variance，但并不会减少你的bias（取决于你的分类点），因此总的误差还会增大

6. Bagging： Bootstrap aggregating
   - 就是用每个cv的训练集来单独预测，然后取个均值，在tuning hyper-parameter的时候也能用
   - **regression**：每个模型的预测均值
   - **classification**：最多的那个投票
   - 可以用在forward stepwise regression：
   - 主要能大量减少Variance
   - 但不能减少lasso和bridge的RSS：
     - 因为penalty已经起到减少var的作用了，所以这里不起作用
     - 因此必须用能够减少bias的方法来做  【soft lasso
   - **想要控制你的模型的方差是，bagging是个很常用的想法**

7. Out of Bagging Error:
   - Amazing!!
   - 看PPT
   - 用bagging中没有取到的样本集来做测试集的误差估计

8. 普通平均Bagging 的不足：
   - 做Bootstrap的时候，抽出来的集合之间并不是独立的，存在相关关系，【更重要的是产生的Tree模型是相关的】，因此不能保证bagging的方法一定能降低估计的方差
   - 考虑克服这样的问题：De-correlation
   - 注意：De-corr的时候不能同时增加Bias，不然白费

9. Random Forest：
   - 为模型添加随机性，从而减小相关性
   - 在树模型生成的时候，不在对每个维度都检查最优的分界点，然后选最优的
   - 而是先随机选一些维度，检查最优分界点，然后再来选最优的  【注意：每一次生成子树的时候都要这样做，而不是用同样的随机维度集去做split】
   - 这样之后，每个树的生成时，相关性就会减小
   - 你每一步的active set大小应该是 sqrt（p）【即mtry】
   - 但如何保证你这样做bias不会增加呢？：要用足够大的tree，这样你的bias就不会增加太多
   - **减少你的tree的相关性的一种重要方法！！！**
   - 这样即使你的数据集相关性大，但通过这样的方式你做出来的树 
   - 从每个bootstrap抽出来的样本中，再随机抽一些集合 出来做aggregate

10. 用Random Forest的方式做 Forward stepwise regression:
    - 在每次添加一个变量时，不从全局选，而是从一个随机的子集里面选

11. 将这种randomize的工作，看成正则化的一种，都是对你的模型施加一种约束【同样的 Drop-out层，可以写成Ridge-regression的形式
    - degrees of freedom： bagging的应该大于random；因为bagging对所有维度搜查，而random的只选取部分
    - 此时你的 mtry就像超参数一样 需要tuning；随着mtry增大，自由度也增大
    - randomize的工作 就是一种隐式的正则化

12. Final exam： 什么时候bagging会比random forest更好
    - 即自由度越高的越好：  
      - 类似什么时候需要正则化
      - 数据信噪比低，噪声多，容易过拟合，因此此时需要正则化
      - 但当信噪比高时，信息多，此时不需要正则化，那么这个时候 bagging会比random forest更好
      - 同样这时候的forward stepwise的方法会比lasso、ridge更好

13. Regularization：
    - Explicit: Lasso\Ridge
    - Implicit: Random Forest\Drop-out\Early Stopping in gradient

14. 怎么用Random Forest 去做**重要性排名**：
    - 定义一个变量的重要性： 加了这个结点后损失函数的变化  【每次用到这个变量的时候把之前的叠加起来
    - 有randomization后，在衡量变量重要性方面，比传统的更好，因为
      - 1.可以衡量有相关关系的变量的重要性【因为有可能把其中一个筛选出去
      - 2.可以做很多树的平均
    - **也有问题**：
      - 在**不同种类**的数据上做重要性排名时，会更加侧重连续型的变量，因为此时的分段点更多
      - 不同种类指：有的是连续的、有的是分类型变量
      - paper：[Bias in random forest variable importance measures: Illustrations, sources and a solution](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25)


***
## April. 21

1. random forest不会产生过拟合的现象，因为不同的tree模型做平均后效果并不会更好，因此不会产生过拟合现象。

2. Boosting
   - 为了Classification研究出来的
   - Intuition:
      - 当我有很多个弱分类器时，例如：stump 【即单层树，一个root两个叶子结点
      - 考虑用很多个弱分类器，得到一个强分类：Adaboost
   - Adaboost：
      - 只能做2class的classification，并且y必须是-1、1  【当然后面可以改进成0、1  但这里的算法是针对-1、1的
      - 对每个模型f做alpha加权平均，再取sign
      - $sign(\sum\alpha_mf_m(x))$ 
      - 因此问题在于优化得到每个模型的权重alpha  【很自然的，当模型数量f越多，你的优化参数alpha就越多，此时越容易过拟合
      - 具体的algorithm看ppt

3. Adaboost：
   - 并不是对误分类的误差做优化！！
   - boosting idea 其实是 additive logistic regression !!!

4. Statistical view of Adaboost:
   - Loss function: 
   - 看ppt
   - 能和logistics regression联系起来  【真的nb
  
5. margin & Loss function for classification:
   - ppt
   - 推导一下Logistic到binomial的形式  【考试！！！】
   - margin：y*f(x)  在分类任务的Loss function设计中经常会用到的
   - y - f(x)  在回归任务中经常用到的

6. Change Adaboost into regression model
   - 只用将损失函数 改成 线性回归的 均方误差形式就行了
   - 相当于 用上一个模型的residual 当作y 放进新的fm(x)的模型中，去fit 新模型
   - 这就是gradient boosting 【GBDT】了 ！！！  

7. 考虑在每次更新时，做一些shrinkage在新的模型上，这时候变成了真正的gradient boosting
   - shrinkage 也可以叫learning rate
   - 注意GBDT中的这个，并不是hyper 而是真正需要的
   - 加模型这个步骤，可以看成forward stepwise模型，是一种greedy的操作，跟L0—norm差不多，只有在信噪比强时能够使用  【跟之前提到的一样



***
April. 28

1. Gradient Boosting 做分类任务
   - 首先记住：Gradient Boosting中所有的子树，都必须是回归树  【因为当使用分类树时，无法做shrinkage的操作
   - Generalized Residual(working response): negative gradient of loss function
   - 当你的loss function是可导的时候：
     - 做回归任务【用square loss】
     - 做分类任务时：要找一个函数形式 去接近广义的负梯度
   - 更广义来看：GBDT做的是，用一个函数的梯度，更新一个函数，有点像对函数的梯度下降更新


2. Stochastic Gradient Boosting Tree:
   - 有点像SGD、MBGD和BGD的区别
   - 即在Gradient Boosting的每步更新中，更新方式是对原始估计F，加一个Loss function对F的导数，即上面1说的，每一步更新估计的函数并且用的是梯度去更新
   - 这就像SGD\BGD\MBGD
   - > 关于SGB\MBGD\BGD的一个知识点：！！
     - 实际上在从BGD转到MBGD时，一般理解成 1.对计算加速； 2.但同时会减少精度
     - 但实际上2是不对的，可以从RF中的randomization来看，MBGD中，用来计算减少更新每次参数的梯度的batch适当的减小，可以起到类似正则化的效果
     - 因此，只要采取的batch得当，不仅不会减少，还是可以提升精度的
   - 回到GBDT： 看ppt p73
     - 同样可以引用batch更新的思想，用部分的data去计算用来更新的参数
   - 与传统的不同：
     - 1. 不是用全部的数据去fit，而是随机抽
     - 2. 还要做一次argmin 即用这部分的data去fit一个tree 然后当作新的

3. Example ： Logistic form of SGBT
   - 用Gradient Boosting的形式做逻辑回归
   - 首先要对f0估计  看ppt p74
   - **期末！！！  ppt p74**
     - 把loss function 变成别的形式，construct 一个gbdt的方式，
   - 记得还要做constraint
   - 精髓在于：可以用GBDT的形式，来更新任何 只要能求导的loss function

4. SGD

5. Shrinkage & Overfitting


6. adaboost 并不是不会过拟合，只是选取的loss function不对，不能用miss classification function
   - 因为这个loss function不平滑

7. 用Boosting Idea 做Linear regression
   - 具体算法看ppt
   - 效果跟lasso很像，因此，可以将Boosting 看成做正则化的一种

8. Boosting优点：
   - 1. 
   - 2. 
   - 3. Function 
   - 4.


9. EM算法
    - EM的9层境界  https://blog.csdn.net/weixin_45962068/article/details/118251318
    - 本质： goal永远是做极大似然函数的事情，并且是最大化 不完整变量下的  log-likelihood
    - 挑战：
      - 1.由于存在缺省变量、因此log-likelihood的maximum不能直接做出来
      - 2. 即使能写出来边缘密度函数的形式，还要求导，但你并不知道p(x)的表达式
    - 为了克服以上的问题：提出EM算法
    - 分解：
      - E
      - M

10. 用EM来做logistics regression
    - 误分类问题，你训练模型的数据本身，就包含miss label的数据
    - 这时候就要引入EM，引入误分类的变量，然后对误分类变量做概率形式估计，从而能够用EM算法，得到最终的解估计
    - 相当于EM和logistics结合
    - 事实上这种想法 / 工具 能用到很多算法上



***
## May. 5th

EM algorithm

1. Missing data: 不一定是缺失值，只要观测不到就算 （模型参数。。）

2. E-step：
- take mean of compelete-data log-likelihood
- 引入潜变量和对潜变量的先验估计，然后取mean $p(z|x, \theta^{old})$，得到前一个参数和更新参数的theta联合max-likelihood 当作Q-function
- **永远只对潜变量（遗失变量）做期望**
- 

3. M-step:
- Maximize Q- function

4. 重复E-M，得到


5. Final: ppt习题 p10页

6. theta 就是 Πk（ppt里的推导）

7. K-means clustering： specail case of gausse mixture
- 是EM算法的一种特殊情况，可以用高斯分布近似

8.如何初始化EM算法：
- 用Kmeans 选取你的潜变量分组，做one-hot

9. EM算法第四层： 本质在不断提升θ的下界
- 把log-likelihood拆成 lower-bound + KL(q||p)
- EM算法的log总是在光滑提升的，不可能下降

10. 第六层：
- E step 在更新先验分布
- M step 在更新分布的参数
- 可以用不同的概率分布  去估计，那个条件概率分布


coordinate-descent
l0-norm



***
# Final
Proof of LOOCV in Linear model:
https://robjhyndman.com/hyndsight/loocv-linear-models/#:~:text=The%20leave%2Done%2Dout%20cross,y%5E%5Bi%5D%20is%20the
[岭回归-Loocv通解公式推导 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/607797639)

[岭回归直接得到最优解的公式推导_岭回归推导_liweiwei1419的博客-CSDN博客](https://blog.csdn.net/lw_power/article/details/82953337)

[线性回归大结局(岭(Ridge)、 Lasso回归原理、公式推导)，你想要的这里都有 - 一无是处的研究僧 - 博客园 (cnblogs.com)](https://www.cnblogs.com/Chang-LeHung/p/16732520.html)

https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/em_algorithm_slides.pdf

