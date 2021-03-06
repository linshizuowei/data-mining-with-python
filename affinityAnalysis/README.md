该部分是有关挖掘数据关联规则的算法，主要是Apriori算法。

首先理解一个概念 -- 频繁项集，通俗来说就是一起出现次数多的数据集，
从频繁项集中我们可以归纳出集合的一些模式，从而有助于我们做一些决策。
这里有两个问题，首先是频繁项集的评估标准，即什么样的集合才能叫做频繁项集
，比如10条记录，里面A和B同时出现了三次，那么我们能不能说A和B一起构成频繁
项集呢？因此我们需要一个评估频繁项集的标准，常用的评估标准有三个：支持度，
置信度和提升度。关于这三个标准的解释这里不再赘述；第二个问题是当数据集规
模很大时，我们很难直接发现频繁项集，这催生了挖掘关联规则的相关算法，比如
Apriori, PrefixSpan, CBA，通过这些算法，我们可以从数据集中提取出频繁项集
，并发现频繁项集的关联规则。

对于Apriori算法，我们使用支持度来作为判断频繁项集的标准。Apriori算法
的目标是找到最大的K项频繁集。这里有两层意思，首先，我们要找到符合支持度
标准的频繁集；但是这样的频繁集可能有很多，第二层意思就是我们要找到最大个
数的频繁集。

Apriori算法采用了迭代的方法，先搜索出候选1项集及对应的支持度，剪枝去掉低
于支持度的1项集，得到频繁1项集。然后对剩下的频繁1项集进行连接，得到候选
的频繁2项集，筛选去掉低于支持度的候选频繁2项集，得到真正的频繁二项集，
以此类推，迭代下去，直到无法找到频繁k+1项集为止，对应的频繁k项集的集合即
为算法的输出结果。

Apriori算法的挖掘效率稍差一些，但算法本身不复杂，较好理解。

参考：http://www.cnblogs.com/pinard/p/6293298.html