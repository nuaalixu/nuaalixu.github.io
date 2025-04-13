---
layout: post
category: Deep Learning
tag: Positional Encoding
---
# Positional Encoding in Transformer
# 位置编码的意义
对于Transformer模型来说，因为纯粹的Attention模块无法捕捉输入顺序的，即无法区分不同位置的Token，所以位置编码的加入是必不可少的。
# 位置编码基本特征
1. 唯一性：不同位置的token的编码唯一；
2. 相对性：任何长度的序列中，不同位置的token之间相对位置保持一致；
3. 外推有界性：容易推广到未见过的长句，因此其值域应有界限。

# 位置编码分类
位置编码有规则定义的，如Transformer利用三角函数公司去计算每个位置向量。也有通过参数学习的，如Bert给每个位置都用可学习的向量在预训练数据上训练，最后存储在位置词表中。
但通常不以如何得到位置编码进行分类，而是以模型如何使用位置编码，分成两大类：
* 绝对位置编码：建模每个输入的位置信息，然后将位置信息融入到输入中。
* 相对位置编码：在算Attention的时候考虑当前位置与被Attention的位置的相对距离，使得它有能力分辨不同位置的Token。

例如，对于最大长度为T的多个输入序列，如果使用序列中每个token的位置序号作为位置编码，那么最多存在T个位置编码，即位置编码形如$(T,D)$，D是位置编码的维度。将原来的输入和位置编码相加送入Attention模块，此时位置编码和token间的相对位置无关，属于绝对位置编码。

如果对某个token相对于其他token的相对位置建模，每个token前面最多有T-1个token，后面最多也有T-1个token，再包括自身，最多共2T-1个相对位置，得到位置编码形如$(2T-1, D)$。在Attention计算时，根据$q_i$和$k_j$的相对位置，给注意力得分加上对应的位置编码，这种方式属于相对位置编码。

# Sinusoidal Positional Encodings
第t个Token的第i个维度的位置编码公式如下：

$$
PE_t^{(i)} =
\left\{
\begin{align*}
&sin(\frac{1}{10000^\frac{2k}{d_{model}-1}} t), i=2k\\
&cos(\frac{1}{10000^\frac{2k+1}{d_{model}-1}} t), i=2k+1
\end{align*}
\right.
$$

特征一：sin与cos的交替。

计算公式中偶数维度和奇数维度分别采用不同的正弦和余弦函数，这样使得任意位置的编码只需通过一个相对值的变换即能转换到距离当前同样相对值的位置编码表示的形式，即：

$$
PE_{t+\Delta t}=T_{\Delta t}*PE_t
$$

通过三角函数中和差化积公式可证，例如以2维的位置编码举例：

$$
\begin{pmatrix}

sin(t+\Delta t) \\
cos(t + \Delta t)
\end{pmatrix}
=
\begin{pmatrix}

cos(\Delta t) & sin(\Delta t) \\
-sin(\Delta t) & cos(\Delta t)
\end{pmatrix}

\begin{pmatrix}
sin(t) \\
cos(t)
\end{pmatrix}
$$

它形式和二维向量空间的旋转矩阵一模一样。扩展到多维也适用。

此外，sin与cos的交替，使得两个位置编码的点积仅取决于偏移量，保证了相对性。
证明如下：

$$
\begin{aligned}
PE^T_t*PE_{t+\Delta t}&=\sum_0^{\frac{d_{model}}{2}-1}(sin(w_i t)sin(w_i (t+\Delta t))+cos(w_i t)sin(w_i(t+\Delta t)) \\
&=\sum_0^{\frac{d_{model}}{2}-1}cos(w_i(t-(t+\Delta t))) \\
&= \sum_0^{\frac{d_{model}}{2}-1}cos(w_i \Delta t)
\end{aligned}
$$

特征二：频率随着维度上升而变小。

三角函数位置编码的频率$w_i$计算如下：

$$
w_i = \frac{1}{10000^\frac{i}{d_{model}-1}}
$$

已知频率越小，周期越大，相邻位置数值变化越小，但可覆盖的位置范围越大。频率越大，周期越短，相邻位置数值变化大，但可覆盖的位置范围就越小。

为了保证每个位置编码不重复，同时足够具有区分性，将三角函数的频率参数沿着维度设置小。

三角函数位置编码的性质：

1. 两个位置编码的点积(dot product)仅取决于偏移量；

2. 三角函数位置编码无法学习到方向，即从位置$t$到$t+\Delta t$，和从位置$t$到$t-\Delta t$是一样的。而且还是对称的。

3. 三角函数位置编码在较远距离（较大偏移量）位置信息较差，三角函数编码进入Attention层时乘上参数矩阵后位置信息会被损坏。

在使用时，直接将三角函数位置编码和token的embedding向量相加，送入注意力模块进行计算。

# Relative Positional Encodings
广义上，在建模时考虑两个token间相对位置信息的编码，都属于相对位置编码。本章介绍两种具体的实现，分别来自于论文1 [Self-Attention with Relative Position Representation](https://arxiv.org/pdf/1803.02155v2)和论文2 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)。

## Self-Attention with Relative Position Representation

论文1使用可学习的向量表示每个相对位置编码，在使用时分别和向量$k$和向量$v$相加，并且$k$和$v$分别相加不同的相对位置编码，也即存在两套可学习的相对位置编码。

例如对于注意力权重，使用相对位置编码后，计算如下：

$$
e_{ij}=\frac{x_i W^Q(x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}
$$

注意力模块输入向量计算如下：

$$
z_i=\sum^n_{j=1}\alpha_{ij}(x_jW^V+a_{ij}^V)
$$

其中$a_{ij}^K$和$a_{ij}^V$均是相对位置编码向量，由参数学习而来。

论文限制了相对位置的数值最大为k，即相对位置编码仅考虑[-k+1, k-1]，超过范围的沿用边界值，这样做原因如下：
1. 超出一定距离的两个位置无需精确编码；
2. 训练时限制最大距离便于外推到没见过的距离。

因此相对位置编码矩阵的形状为$(2k-1, d_z)$，其中$d_z$是注意力模块的输出维度。

此外，相对编码在不同的注意力头之间共享，以降低空间复杂度。

## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
论文2采用三角函数和可学习参数相结合的方式表示相对位置编码。

当采用三角函数进行绝对位置编码后，注意力权重的计算，可以拆解如下：

$$
\begin{align*}
A^{abs}_{i,j} &= (E_{x_i} + U_i)W^T_q W_k (E_{x_j} + U_j)\\
&= E_{x_i}^T W^T_q W_k E_{x_j}+E_{x_i}^T W_q^T W_k U_j+U_i^T W_q^T W_k E_{x_j}+U^T_i W_q^T W_k U_j
\end{align*}
$$

如上所示，注意力权重的计算被拆解成四个子项。

论文2做出如下修改：

1. 上式中$U$表示绝对位置的三角函数编码，论文2将第二和第四子项中的$U_j$替换成相对位置的三角函数编码$R_{i-j}$。这一改动本质上体现了只有相对距离才对关注点有影响。
2. 将第三子项中的$U^T_i W_q^T$替换为可学习的向量$u$，旨在于取消query位置的影响。同样的考量，将第四项中的$U^T_i W_q^T$替换为可学习的向量$v$。
3. 将$W_k$矩阵替换为两个不同的矩阵$W_{k,E}$和$W_{k, R}$，分别用于产生内容相关的key和位置相关的key。

按照如上修改，新的注意力权重计算公式如下：

$$
\begin{align*}
A^{abs}_{i,j} &= E_{x_i}^T W^T_q W_{k,E} E_{x_j}+E_{x_i}^T W_q^T W_{k,R} R_{i-j}+u^T W_{k,E} E_{x_j}+v^T W_{k,R} R_{i-j}
\end{align*}
$$

子项1表示基于内容的寻址，子项2捕获内容相关的位置偏差，子项3控制全局内容偏差，子项4编码全局位置偏差。

和论文1相比，论文1仅拥有子项1和子项2，并且放弃了三角函数，也就放弃了由三角函数归纳偏差来带的泛化能力。

# RoPE
RoPE在分类上也属于相对位置编码。

未完待续...
