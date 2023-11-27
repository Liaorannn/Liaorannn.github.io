---
layout: post
title: Reinforcement Learning Notes
subtitle: Reinforcement Learning study notes
author: Liaoran
categories: Notes
tags:
  - RL
  - notes
---

## Reinforcement Learning
***

**Q：解决的目标问题？基础要素有哪些？核心是什么？**
A：相比于有监督和无监督学习，强化学习目标是求的代理人在不同环境下的最优决策。通过机器学习得到不同环境下代理人应该做出的最优决策，因此学习目标是一个决策，或者action，一种行动策略，而不是单一的数字。

基础要素主要有state, value, environment, reward, policy, return等。核心是MDP马尔可夫决策过程和Bellman equation。即agent在不同的state下，做出不同的action和environment进行互动，得到不同的return并且进入下一个state。

Bellman equation用来描述state value，当然也可以写成action value的形式。
state value = 当前state所有action的immediate reward期望和 + 所有未来可能达到的state的state value的期望和。  【可以用action value来表示，互相可转化】
bellman equation给出了评价policy的一种方式。
bellman optimal equation 给出了最佳state value的存在性。解贝尔曼最优方程即等价于，求解最佳的policy。

### RL 的经典算法
***
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126222441221.png)


#### Model Based
> 各个state下的转移状态矩阵已知，reward已知，agent了解所有信息。此时用动态规划，迭代求解bellman optimal function。
> 下面两种办法本质上都是迭代求解贝尔曼最优方程。
##### Value Iteration
	核心理解就是：
	- 本质上就是用贝尔曼公式，求解整个系统的State Value然后根据state value去进行最后的策略选择，整个计算过程中就算把pi的更新给拿掉，只在收敛后再取，也没有任何问题！
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126223454823.png)

##### Policy Iteration
	初始化Pi0，同样先做policy evaluation(这一步先迭代到最优的state value)，然后更新最优策略pi，然后再做policy evaluation 做iteration
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126223506754.png)

#### Model Free
> 现实状态，不知道转移状态矩阵，并且不知道reward函数，通过与环境互动得到的信息进行学习，更新策略。 

##### Valued Based
> 通过学习各个state的价值函数，得到最优policy。

###### Monte Carlo Learning
> 用样本频率估计模型中的转移概率等。
> 本质上：  
	- 就是把Policy Iteration中的policy evaluation，从具体计算换成了抽样求平均估计而已  
	-【注意】这里的pseudocode并没有给出得到episode和return的算法！  
	-【注意】这里对每个state、action pair都要做N个episode进行return估计，因此太麻烦了【当然如果policy和env都是deterministic的话，就可以只取一个trajectory因为确定】  
	每步episode长度需要注意，尽量足够长
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126223847002.png)

坏处：方法太过耗时。需要采样很多遍。
改进：exploting start方法， 策略每步迭代方法。

###### TD-Learning
***
>  本质：用类似RM算法的迭代的思想，从单一trajectory中 直接对state value、action value等进行估计。  
>  本质上：用RM算法解决bellman equation。通过t+1时刻的reward和state value来更新 t时刻的状态价值函数。

![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126224845091.png)


###### On-Policy
> 用同一套策略，即用来产生state上的各种数据（s，a，r，s，a），又用产生的数据更新原始策略。
> （显然MC也是on-policy的）

**SARSA:**
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126225122263.png)

【后续】：有N-STEP SARSA 和 Expected SARSA

###### Off-Policy
> 用两套策略，一套策略用来和环境交互，产生信息记录；另一套策略使用产生的信息进行更新迭代。

**Q-Learning：**
>  本质上是直接解决贝尔曼的最优问题，因为存在max

**on-policy版本：**
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126225532649.png)

**off-policy版本：**
![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126225542502.png)



### Continuous Space
>  以上都是离散的状态-动作空间，类似于state-action的表格。在实际过程中，面对的大多是离散空间，因此，需要引入神经网络对value进行估计。


#### DQN
>  本质：
>  在DQN中：由于神经网络本身已经能够进行迭代找到最优参数，因此不需要再手动进行求导参数跟新。但问题在于神经网络拟合value function需要用到大量数据，这个数据本身是不能具有顺序和特殊分布的，也因此，在DQN中，引入了replay buffer和experience replay的想法，通过对已有策略不断产生样本，然后平均抽样放进网络模型中，这样能使神经网络估计的value function收敛，但这样的收敛只在一次数据的抽样中，并且目标函数（Yt）也是仅仅针对一次的数据产生的。由于每次数据抽样可能存在误差，因此要不断的进行迭代抽样，并且每隔一段时间用新的w替代目标函数中的参数w，这就是DQN“两个网络”设计的原因。整个不断抽样，并且定期更新参数的过程可以理解为，减少value approximation function **的** 目标函数（损失函数）的bias，即让目标函数贴近真实的目标函数的过程。
>  
>  **最屌的设计：**
> 	 - Replay buffer
> 	 - Double Net

**DQN和普通Q-table的区别：**  
重点是：为什么Q-learning就要有**experience replay**，而Q-table没有

答： 因为  
- Q-table实际上在直接解决一个bellman optimistic problem，实际不需要用到S，A的分布【实际上是类似于对每个（s,a）的点，求BOE得到q-value，只是求解的方法，用的是类似EM（SGD）算法，即对每个点都要遍历多次采样，然后公式的长相和这边functional approximation类似】；  
- 但Q-functional approximation中，是用最优化的方法求解一个函数估计、最小化损失函数的问题，因此不是针对某一个（s，a）点的，而是对一整个action-value function的参数估计，因此需要进行样本采样，这个时候采样 用到的就是 experience replay

![](_data/img/2023-11-26-Reinforcement-Learning-Notes/image-20231126233134971.png)


