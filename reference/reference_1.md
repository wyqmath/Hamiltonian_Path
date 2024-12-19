# An Efficient Algorithm for Hamiltonian Path Embedding of \( k \) -Ary \( n \) -Cubes Under the Partitioned Edge Fault Model

# 一种在划分边故障模型下对 \( k \) -元 \( n \) -立方体进行哈密顿路径嵌入的高效算法

Hongbin Zhuang ©, Xiao-Yan Li D, Jou-Ming Chang (C), and Dajin Wang (C)

郑红宾 ©, 李晓燕 D, 张觉明 (C), 以及王大金 (C)

Abstract-The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is one of the most important interconnection networks for building network-on-chips, data center networks, and parallel computing systems owing to its desirable properties. Since edge faults grow rapidly and the path structure plays a vital role in large-scale networks for parallel computing, fault-tolerant path embedding and its related problems have attracted extensive attention in the literature. However, the existing path embedding approaches usually only focus on the theoretical proofs and produce an \( n \) -related linear fault tolerance since they are based on the traditional fault model, which allows all faults to be adjacent to the same node. In this paper, we design an efficient fault-tolerant Hamiltonian path embedding algorithm for enhancing the fault-tolerant capacity of \( k \) -ary \( n \) -cubes. To facilitate the algorithm, we first introduce a new conditional fault model, named Partitioned Edge Fault model (PEF model). Based on this model,for the \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) with \( n \geq  2 \) and odd \( k \geq  3 \) ,we explore the existence of a Hamiltonian path in \( {Q}_{n}^{k} \) with large-scale edge faults. Then we give an \( O\left( N\right) \) algorithm,named HP-PEF, to embed the Hamiltonian path into \( {Q}_{n}^{k} \) under the PEF model, where \( N \) is the number of nodes in \( {Q}_{n}^{k} \) . The performance analysis of HP-PEF shows the average path length of adjacent node pairs in the Hamiltonian path constructed by HP-PEF. We also make comparisons to show that our result of edge fault tolerance has exponentially improved other known results. We further experimentally show that HP-PEF can support the dynamic degradation of average success rate of constructing Hamiltonian paths when increasing faulty edges exceed the fault tolerance.

摘要- \( k \) -元 \( n \) -立方体 \( {Q}_{n}^{k} \) 是构建网络芯片、数据中心网络和并行计算系统最重要的互联网络之一，这要归功于其理想的特性。由于边缘故障迅速增长，并且在并行计算的大规模网络中路径结构发挥着至关重要的作用，因此在文献中，容错路径嵌入及其相关问题已经引起了广泛的关注。然而，现有的路径嵌入方法通常只关注理论证明，并且由于它们基于传统的故障模型，该模型允许所有故障与同一节点相邻，因此它们产生了与 \( n \) 相关的线性故障容忍度。在本文中，我们设计了一种高效的容错哈密顿路径嵌入算法，以提高 \( k \) -元 \( n \) -立方体的故障容忍能力。为了便于算法的实现，我们首先引入了一种新的条件故障模型，名为分区边缘故障模型（PEF模型）。基于这个模型，对于具有 \( n \geq  2 \) 和奇数 \( k \geq  3 \) 的 \( k \) -元 \( n \) -立方体 \( {Q}_{n}^{k} \)，我们探讨了在存在大规模边缘故障的 \( {Q}_{n}^{k} \) 中哈密顿路径的存在性。然后我们给出了一种名为HP-PEF的算法，用于在PEF模型下将哈密顿路径嵌入到 \( {Q}_{n}^{k} \) 中，其中 \( N \) 是 \( {Q}_{n}^{k} \) 中的节点数。HP-PEF的性能分析显示了由HP-PEF构建的哈密顿路径中相邻节点对的平均路径长度。我们还进行了比较以显示我们的边缘故障容忍结果指数级地优于其他已知结果。我们进一步通过实验表明，当增加的故障边超过故障容忍度时，HP-PEF可以支持构建哈密顿路径的平均成功率动态下降。

Index Terms- \( k \) -ary \( n \) -cubes,algorithm,fault-tolerant embedding, Hamiltonian path, interconnection networks.

索引术语- \( k \) -元 \( n \) -立方体，算法，容错嵌入，哈密顿路径，互联网络。

## NOMENCLATURE

## 名词术语表

NoC Network-on-Chip

NoC 芯片上网络

TSV Through silicon via

TSV 硅微孔

IC Integrated circuit

IC 集成电路

VLSI Very large scale integration

VLSI 超大规模集成电路

TRC Tours routing chip

TRC 旅行路由芯片

DCN Data center network

DCN 数据中心网络

H-path Hamiltonian path

H-path 哈密顿路径

PEF Partitioned edge fault

PEF 分区边缘故障

\( {Q}_{n}^{k}\;k \) -Ary \( n \) -cube

\( {Q}_{n}^{k}\;k \) -Ary \( n \) -立方体

APL Average path length

APL 平均路径长度

SD Standard deviation

SD 标准差

FT Fault tolerance of \( {Q}_{n}^{k} \) when embedding the H-path under the traditional model

FT 在传统模型下嵌入 H 路径时的 \( {Q}_{n}^{k} \) 容错性

FP Fault tolerance of \( {Q}_{n}^{k} \) when embedding the H-path under the PEF model

FP 在 PEF 模型下嵌入 H 路径时的 \( {Q}_{n}^{k} \) 容错性

ASR Average success rate

ASR 平均成功率

## I. INTRODUCTION

## I. 引言

NETWORK-ON-CHIPS (NoCs) have emerged as a promis- ing fabric for supercomputers due to their reusability and scalability [1], [2]. This fabric allows a chip to include a large number of computing nodes and effectively turn it into a tiny supercomputer. Therefore, it alleviates the bottlenecks faced by the further development of supercomputers. With the rapidly increasing demand for computing capacity, the number of on-chip cores increases quickly, which results in a high average internode distance in two-dimensional NoCs (2D NoCs). Consequently, 2D NoCs exhibit high communication delay and power consumption as the scale of networks increases [3]. Hence, 3D \( \mathrm{{NoCs}} \) have been designed to solve the scalability problem of 2D NoCs. In 3D NoCs, the so-called through silicon via (TSV) links are used to connect various planes or layers. Though the fabrication cost of TSV links is quite high, 3D NoCs can reduce the probability of long-distance communication while still maintaining high integration density.

网络-芯片（NoCs）因其可重用性和可扩展性 [1], [2] 而成为超级计算机的有前景的架构。这种架构允许芯片包含大量的计算节点，并有效地将其转变为微型超级计算机。因此，它缓解了超级计算机进一步发展所面临的瓶颈。随着计算容量的需求迅速增长，芯片上的核心数量快速增加，这导致二维网络-芯片（2D NoCs）中的平均节点间距离较高。因此，随着网络规模的增加，2D NoCs 显示出较高的通信延迟和功耗 [3]。因此，三维 \( \mathrm{{NoCs}} \) 被设计出来以解决 2D NoCs 的可扩展性问题。在 3D NoCs 中，所谓的硅微孔（TSV）连接用于连接不同的平面或层。尽管 TSV 连接的制造成本相当高，但 3D NoCs 能够在保持高集成密度的同时减少远距离通信的概率。

However, since the additional expense is required for incorporating more processing nodes, NoCs demand a robust fault tolerance. For example, 3D integrated circuit (IC) fabrication technology improves the power density of modern chips, which results in a thermal-intensive environment for 3D NoCs. High core temperatures reduce chip lifetime and mean time to failure, as well as resulting in low reliability and high cooling costs. The faults in NoCs are mainly divided into two categories, namely, transient faults and permanent faults. Generally speaking, permanent fault deserves more attention [4] since it seriously affects the transmission of more packets. With the rapid increase in the number of processing nodes, NoCs may encounter many permanent faults problems [5], and more reliability threats accompanied by permanent faults will also appear [6]. Therefore, we mainly discuss the permanent faults in this paper.

然而，由于需要额外费用来集成更多的处理节点，NoCs（网络芯片）需要具备强大的容错能力。例如，3D集成电路（IC）制造技术提高了现代芯片的功率密度，导致3D NoCs面临热密集环境。高核心温度会缩短芯片寿命和平均故障间隔时间，同时导致可靠性降低和冷却成本增加。NoCs中的故障主要分为两类，即暂时性故障和永久性故障。一般来说，永久性故障更值得关注[4]，因为它严重影响更多数据包的传输。随着处理节点数量的快速增加，NoCs可能会遇到许多永久性故障问题[5]，并且伴随着永久性故障的可靠性威胁也会出现[6]。因此，本文主要讨论永久性故障。

---

<!-- Footnote -->

Manuscript received 23 July 2022; revised 4 February 2023; accepted 1 April 2023. Date of publication 5 April 2023; date of current version 8 May 2023. This work was supported by the National Natural Science Foundation of China under Grant 62002062 (X.-Y. Li), in part by the Ministry of Science and Technology of Taiwan under Grant MOST-111-2221-E-141-006 (J.-M. Chang), and in part by the Natural Science Foundation of Fujian Province under Grant 2022J05029 (X.-Y. Li). Recommended for acceptance by D. Yang. (Corresponding author: Xiao-Yan Li.)

手稿于2022年7月23日收到，于2023年2月4日修订，于2023年4月1日接受。发表日期为2023年4月5日；当前版本日期为2023年5月8日。这项工作得到了中国国家自然科学基金资助（项目编号62002062，李晓燕），部分得到了台湾科技部资助（项目编号MOST-111-2221-E-141-006，张俊明），以及部分得到了福建省自然科学基金资助（项目编号2022J05029，李晓燕）。由杨德推荐接受。（通讯作者：李晓燕）

Hongbin Zhuang and Xiao-Yan Li are with the College of Computer and Data Science, Fuzhou University, Fuzhou 350108, China (e-mail: hbzhuang476@gmail.com; xyli@fzu.edu.cn).

郑红宾和李晓燕均任职于福州大学计算机与数据科学学院，福州350108，中国（电子邮件：hbzhuang476@gmail.com; xyli@fzu.edu.cn）。

Jou-Ming Chang is with the Institute of Information and Decision Sciences, National Taipei University of Business, Taipei 10051, Taiwan (e-mail: spade@ntub.edu.tw).

张乔铭是台湾台北商业大学信息与决策科学研究所的成员，地址：台湾台北10051 (电子邮件：spade@ntub.edu.tw)。

Dajin Wang is with the Department of Computer Science, Montclair State University, Montclair, NJ 07043 USA (e-mail: wangd@montclair.edu).

汪大金是蒙特克莱尔州立大学计算机科学系的成员，地址：美国新泽西州蒙特克莱尔07043 (电子邮件：wangd@montclair.edu)。

Digital Object Identifier 10.1109/TPDS.2023.3264698

数字对象标识符 10.1109/TPDS.2023.3264698

<!-- Footnote -->

---

A well-designed fault-tolerant routing algorithm can address the fault tolerance challenges of NoCs by bypassing the faults when delivering packets. Routing algorithms should be not only fault-tolerant but also deadlock-free. The Hamiltonian path (H-path for short) strategy is a powerful tool for deadlock avoidance. Since the H-path traverses every node in the network exactly once and contains no cycle structure, the deadlock can be easily prevented by transmitting the packets along the H-path [7], [8], [9], [10], [11], [12], [13]. In recent years, this excellent strategy is utilized for designing fault-tolerant routing algorithms. For instance, the HamFA algorithm [7] is one of the most famous fault-tolerant routing algorithms using the H-path strategy. It constructs two directed subnetworks through the \( \mathrm{H} \) -path strategy and limits packets to be routed in a single subnetwork so that the deadlock can be avoided. Simultaneously, HamFA can tolerate almost all one-faulty links. The FHOE algorithm [8] is also a fault-tolerant routing algorithm based on the H-path strategy for 2D NoCs. It fully combines the advantages of traditional odd-even turn model and HamFA strategy, and consequently can provide higher adaptivity and more choices of minimal paths compared to HamFA. Considering the importance of fault tolerance and extensive applications of the H-path in NoCs, it's natural to investigate the existence of the H-path in NoCs (i.e., the problem of embedding the H-path into NoCs), especially when faults occur (i.e., the fault-tolerant problem of embedding the H-path into NoCs). However, it is well-known that the problem of embedding an \( \mathrm{H} \) -path into a network is NP-complete, even when no fault exists.

设计良好的容错路由算法可以通过在传送数据包时绕过故障来解决NoCs的容错挑战。路由算法不仅应该是容错的，而且应该是无死锁的。哈密顿路径（简称H路径）策略是避免死锁的有力工具。由于H路径在网络中的每个节点恰好经过一次且不包含循环结构，通过沿H路径传输数据包可以轻松防止死锁[7]、[8]、[9]、[10]、[11]、[12]、[13]。近年来，这种优秀的策略被用于设计容错路由算法。例如，HamFA算法[7]是最著名的采用H路径策略的容错路由算法之一。它通过\( \mathrm{H} \)路径策略构建了两个有向子网，并限制数据包在单个子网中路由，从而避免了死锁。同时，HamFA几乎可以容忍所有单链路故障。FHOE算法[8]也是基于H路径策略的2D NoCs的容错路由算法。它充分结合了传统奇偶转向模型的优点和HamFA策略，因此相比HamFA，可以提供更高的适应性和更多的最短路径选择。考虑到容错的重要性以及H路径在NoCs中的广泛应用，研究NoCs中H路径的存在（即，将H路径嵌入NoCs的问题）是很自然的事情，特别是在故障发生时（即，将H路径嵌入NoCs的容错问题）。然而，众所周知，即使在无故障的情况下，将H路径嵌入网络的问题也是NP完全的。

NoCs usually take interconnection networks as their underlying topology, which inherently affects the performance of NoCs. The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is one of the most important interconnection networks for building NoCs owing to its desirable properties, such as regularity, recursive structure, node symmetry, edge symmetry, low-latency, and ease of implementation [14]. The two associated parameters \( k \) and \( n \) in \( {Q}_{n}^{k} \) provide it the ability to satisfy structural needs in a variety of circumstances. A commercial VLSI chip named the Tours Routing Chip (TRC) was designed early to perform wormhole routing in an arbitrary \( k \) -ary \( n \) -cube [15]. Furthermore,many fault-tolerant deadlock-free routing algorithms have been developed in \( {Q}_{n}^{k} \) -based NoCs [16], [17],[18]. The desirable properties of \( {Q}_{n}^{k} \) have even attracted a lot of research actually to build data center networks (DCNs), such as CamCube [19], NovaCube [20], CLOT [21], and Wave-Cube [22]. Though the scale of DCNs is much larger than that of NoCs, \( k \) -ary \( n \) -cubes can easily cope with it. It’s worth pointing out that stronger fault tolerance is necessary for the DCN since it possesses a lot of servers. Moreover, a lot of well-known parallel computing systems like iWarp [23], J-machine [24], Cray T3D [25], Cray T3E [26], and IBM Blue Gene/L [27] all have adopted \( k \) -ary \( n \) -cubes as their underlying topologies. These \( {Q}_{n}^{k} \) - based architectures usually have high bisection width, high path diversity, high scalability, and affordable implementation cost.

NoCs 通常采用互联网络作为其底层拓扑结构，这本质上影响了 NoCs 的性能。 \( k \) -叉 \( n \) -立方体 \( {Q}_{n}^{k} \) 是构建 NoCs 的最重要的互联网络之一，这是由于它具有一系列令人期待的特性，如规律性、递归结构、节点对称性、边缘对称性、低延迟和易于实现 [14]。\( {Q}_{n}^{k} \) 中的两个相关参数 \( k \) 和 \( n \) 使其能够满足各种情况下的结构需求。早期设计的一款商业 VLSI 芯片，名为 Tours 路由芯片（TRC），用于在任意的 \( k \) -叉 \( n \) -立方体中执行虫洞路由 [15]。此外，基于 \( {Q}_{n}^{k} \) 的 NoCs 已经开发出了许多容错无死锁路由算法 [16]、[17]、[18]。\( {Q}_{n}^{k} \) 的这些理想特性甚至吸引了许多研究实际构建数据中心网络（DCNs），如 CamCube [19]、NovaCube [20]、CLOT [21] 和 Wave-Cube [22]。尽管 DCNs 的规模远大于 NoCs，\( k \) -叉 \( n \) -立方体可以轻松应对。值得注意的是，由于 DCN 拥有大量服务器，因此它需要更强的容错能力。此外，许多知名的并行计算系统，如 iWarp [23]、J-machine [24]、Cray T3D [25]、Cray T3E [26] 和 IBM Blue Gene/L [27]，都采用了 \( k \) -叉 \( n \) -立方体作为其底层拓扑结构。这些基于 \( {Q}_{n}^{k} \) 的架构通常具有高分割宽度、高路径多样性、高可扩展性和可承受的实现成本。

In order to apply the attractive H-path structure in \( {Q}_{n}^{k} \) with as many faults as possible, the fault-tolerant problem of embedding the H-path into \( {Q}_{n}^{k} \) has been extensively investigated in [28], [29],[30],[31],[32],[33],[34]. A network \( G \) is Hamiltonian-connected if an H-path exists between any two nodes in \( G \) . Also, \( G \) is \( f \) -edge fault-tolerant Hamiltonian-connected provided it is Hamiltonian-connected after removing arbitrary \( f \) edges in \( G \) . Yang et al. [32] proved that for any odd integer \( k \geq  3,{Q}_{n}^{k} \) is(2n - 3)-edge fault-tolerant Hamiltonian-connected. Stewart and Xiang [33] proved that for any even integer \( k \geq  4 \) ,there is an \( \mathrm{H} \) -path between any two nodes in different partite sets in \( {Q}_{n}^{k} \) with at most \( {2n} - 2 \) faulty edges. Yang and Zhang [34] recently showed that for every odd integer \( k,{Q}_{n}^{k} \) admits an \( \mathrm{H} \) -path between any two nodes that avoids a set \( F \) of faulty edges and passes through a set \( L \) of prescribed linear forests when \( \left| {E\left( L\right) }\right|  + \left| F\right|  \leq  {2n} - 3 \) . All the above results are obtained under the traditional fault model, which doesn't exert any restriction on the distribution of faulty edges. However, Yuan et al. [35] and \( \mathrm{{Xu}} \) et al. [36] respectively pointed out that this model has many flaws in the realistic situation since it ignores the fact that it's almost impossible for all faulty nodes (resp. faulty edges) to be adjacent to the same node simultaneously (unless that the node fails). In other words, the fault tolerance assessment approaches under the traditional fault model seriously underestimate the fault tolerance potential of \( {Q}_{n}^{k} \) .

为了尽可能多地应用 \( {Q}_{n}^{k} \) 中的吸引性 H-路径结构，研究者们已经在 [28]、[29]、[30]、[31]、[32]、[33]、[34] 中广泛研究了将 H-路径嵌入 \( {Q}_{n}^{k} \) 的容错问题。一个网络 \( G \) 如果任意两个节点之间存在 H-路径，则称为哈密尔顿连通。此外，如果移除 \( G \) 中的任意 \( f \) 条边后仍然哈密尔顿连通，则 \( G \) 是 \( f \) -边容错哈密尔顿连通的。杨等人 [32] 证明了对于任意奇数 \( k \geq  3,{Q}_{n}^{k} \) ，它是 (2n - 3)-边容错哈密尔顿连通的。斯图尔特和向 [33] 证明了对于任意偶数 \( k \geq  4 \) ，在 \( {Q}_{n}^{k} \) 中不同的分部集之间的任意两个节点之间存在一条至多包含 \( {2n} - 2 \) 条故障边的 \( \mathrm{H} \) -路径。杨和张 [34] 最近表明，对于每个奇数 \( k,{Q}_{n}^{k} \) ，当 \( \left| {E\left( L\right) }\right|  + \left| F\right|  \leq  {2n} - 3 \) 时，它允许任意两个节点之间存在一条避开故障边集合 \( F \) 并通过指定的线性森林集合 \( L \) 的 \( \mathrm{H} \) -路径。以上所有结果都是在传统故障模型下获得的，该模型对故障边的分布没有施加任何限制。然而，袁等人 [35] 和 \( \mathrm{{Xu}} \) 等人 [36] 分别指出，由于该模型忽略了几乎不可能所有故障节点（或故障边）同时与同一节点相邻的事实（除非该节点故障），在现实情况下该模型存在许多缺陷。换句话说，传统故障模型下的容错评估方法严重低估了 \( {Q}_{n}^{k} \) 的容错潜力。

The conditional fault model was proposed for tolerating more faulty edges by restricting each node to be adjacent to at least two fault-free edges. Under this model, Wang et al. [29] proved that for any even integer \( k \geq  4 \) ,there is an \( \mathrm{H} \) -path between any two nodes in different partite sets in \( {Q}_{n}^{k} \) with at most \( {4n} - 5 \) conditional faulty edges. Though the fault tolerance they obtained is about twice that under the traditional fault model, it remains linearly correlated with \( n \) . In addition,all the literature mentioned above only provides theoretical proofs about the existence of the \( \mathrm{H} \) -path in \( {Q}_{n}^{k} \) ,while executable fault-tolerant \( \mathrm{H} \) -path embedding algorithms and their performance analysis are missing. Thus, this may hinder the practical application of the \( \mathrm{H} \) -path on \( {Q}_{n}^{k} \) .

条件故障模型被提出，以通过限制每个节点至少与两条无故障边相邻，从而容忍更多的故障边。在这个模型下，王等人 [29] 证明了对于任何偶数 \( k \geq  4 \)，在 \( {Q}_{n}^{k} \) 中，不同分部集的任意两个节点之间存在一条 \( \mathrm{H} \) 路径，且最多有 \( {4n} - 5 \) 条条件故障边。尽管他们获得的容错能力是传统故障模型下的大约两倍，但它仍然与 \( n \) 线性相关。此外，上述所有文献仅提供了关于 \( {Q}_{n}^{k} \) 中 \( \mathrm{H} \) 路径存在的理论证明，而可执行的容错 \( \mathrm{H} \) 路径嵌入算法及其性能分析却缺失。因此，这可能会阻碍 \( \mathrm{H} \) 路径在 \( {Q}_{n}^{k} \) 上的实际应用。

In this paper, we pay more attention to the distribution pattern of faulty edges in each dimension of \( {Q}_{n}^{k} \) . This consideration is based on the fact that various dimensions of \( {Q}_{n}^{k} \) usually possess different faulty features in practical fields. For example, to minimize the fabrication cost of TSV links, \( {Q}_{n}^{k} \) -based 3D \( \mathrm{{NoCs}} \) are often designed with only partial connection in the vertical dimension (i.e., partial TSVs) [37], [38]. Particularly, the TSV density of 3D NoCs was suggested for only 12.5% in [38]. It implies that only 12.5% of vertical links are available, and 87.5% of vertical links can be deemed faulty. In other words, many missed links exist in one dimension inherently when \( {Q}_{n}^{k} \) is utilized for building 3D NoCs. In this case, we can deem the vertically partially connected NoC topology as a \( {Q}_{n}^{k} \) with many faulty links concentrated at the same dimension.

在本文中，我们更加关注 \( {Q}_{n}^{k} \) 每个维度中故障边的分布模式。这种考虑基于这样一个事实：\( {Q}_{n}^{k} \) 的不同维度在实际领域中通常具有不同的故障特征。例如，为了最小化TSV链路的制造成本，基于 \( {Q}_{n}^{k} \) 的3D \( \mathrm{{NoCs}} \) 通常仅在垂直维度（即部分TSV）中设计部分连接 [37]、[38]。特别是，文献 [38] 中建议3D NoC的TSV密度仅为12.5%。这意味着只有12.5%的垂直链路是可用的，而87.5%的垂直链路可以被认为是故障的。换句话说，当使用 \( {Q}_{n}^{k} \) 构建三维NoC时，一个维度中固有无故障链路缺失。在这种情况下，我们可以认为垂直部分连接的NoC拓扑是一个在相同维度上集中了许多故障链路的 \( {Q}_{n}^{k} \)。

Based on the above concerns, for a class of networks that exhibit different faulty features in each dimension, we introduce another fault model, named the partitioned edge fault model (PEF model for short), to help such networks achieve a better fault-tolerant capacity. In essence, this model imposes different restrictions on faulty edges in each dimension according to flawed features. In fact, these restrictions are similar to the concept recently proposed by Zhang et al. [39], which pointed out that restricting the number of faulty edges in each dimension is quite important for reflecting the actual fault-tolerant capacity of a network and can be utilized to improve network edge connectivity. Thus, we utilize the PEF model to explore the fault tolerance potential of \( {Q}_{n}^{k} \) when embedding an H-path into \( {Q}_{n}^{k} \) with \( n \geq  2 \) and odd \( k \geq  3 \) and evaluate the performance of our approach. Our contributions are presented as follows:

基于以上关切，针对在每一维度展现出不同故障特征的一类网络，我们引入了另一种故障模型，命名为分区边故障模型（简称PEF模型），以帮助这类网络实现更好的故障容忍能力。本质上，该模型根据故障特征对每一维度的故障边施加不同的限制。实际上，这些限制与张等人最近提出的概念相似[39]，他们指出限制每一维度中故障边的数量对于反映网络的实际故障容忍能力非常重要，并且可以用来提高网络边连通性。因此，我们利用PEF模型来探索在将H路径嵌入\( {Q}_{n}^{k} \)时\( {Q}_{n}^{k} \)的故障容忍潜力，并评估我们方法的表现。我们的贡献如下所示：

1) We propose a new fault model, the PEF model, to improve the fault tolerance of \( {Q}_{n}^{k} \) when we embed an H-path into the faulty \( {Q}_{n}^{k} \) .

1) 我们提出了一个新的故障模型，PEF模型，以改善在将H路径嵌入故障\( {Q}_{n}^{k} \)时\( {Q}_{n}^{k} \)的故障容忍性。

2) Under the PEF model, we provide a theoretical analysis for proving the existence of the \( \mathrm{H} \) -path in \( {Q}_{n}^{k} - F \) ,where \( F \) is a PEF set (defined in Section III) such that \( \left| F\right|  \leq \) \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \)

2) 在PEF模型下，我们提供了理论分析，以证明在\( {Q}_{n}^{k} - F \)中存在\( \mathrm{H} \)路径，其中\( F \)是一个PEF集合（在第III节中定义），使得\( \left| F\right|  \leq \) \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \)。

3) Based on the obtained theoretical results, we design an \( O\left( N\right) \) algorithm,named \( {HP} - {PEF} \) ,for embedding the \( \mathrm{H} \) - path into \( {Q}_{n}^{k} \) under the PEF model,where \( N \) is the number of nodes in \( {Q}_{n}^{k} \) . To our knowledge,this is the first time that an algorithm is not only proposed and proved correct, but also actually implemented, for H-path embedding into an edge-faulty \( {Q}_{n}^{k} \) .

3) 基于获得的理论结果，我们设计了一个名为\( {HP} - {PEF} \)的\( O\left( N\right) \)算法，用于在PEF模型下将\( \mathrm{H} \)路径嵌入\( {Q}_{n}^{k} \)，其中\( N \)是\( {Q}_{n}^{k} \)中的节点数。据我们所知，这是第一次提出并证明正确，且实际实现的算法，用于将H路径嵌入边故障\( {Q}_{n}^{k} \)。

The implementation of the algorithm afforded us the ability to observe some features of the generated \( \mathrm{H} \) -paths. For example,if an edge connecting nodes \( u \) and \( v \) became faulty,then the path length of \( u \) and \( v \) in the generated \( \mathrm{H} \) -path can be an indicator of how important the missed edge is. By experimenting with the algorithm, we gather the data of average path lengths for all edges in the generated \( \mathrm{H} \) -path.

算法的实现使我们能够观察到生成的 \( \mathrm{H} \) -路径的一些特征。例如，如果连接节点 \( u \) 和 \( v \) 的边出现故障，那么在生成的 \( \mathrm{H} \) -路径中 \( u \) 和 \( v \) 的路径长度可以作为指示丢失边的重要性的指标。通过实验该算法，我们收集了生成的 \( \mathrm{H} \) -路径中所有边的平均路径长度数据。

Our algorithm is shown to outperform all known similar works in terms of tolerated faulty-edges. In particular, compared to [32],the improvement is from linear(2n - 3) to exponential \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) . We also show that HP-PEF can support the dynamic degradation of average success rate of constructing required \( \mathrm{H} \) -paths even when increasing faulty edges exceed \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) .

我们的算法在容忍故障边方面优于所有已知类似工作。特别是，与 [32] 相比，改进是从线性 (2n - 3) 到指数 \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) 的。我们还展示了 HP-PEF 可以支持在故障边增加时，构建所需的 \( \mathrm{H} \) -路径的平均成功率动态降低。

Organization: The rest of this paper is organized as follows. In Section II, we provide the preliminaries used throughout this paper. In Section III, we first present the definition of the PEF model, and then give the theoretical proof related to the existence of the H-path in \( {Q}_{n}^{k} \) under the PEF model. In Section IV,we design the fault-tolerant H-path embedding algorithm HP-PEF based on the theoretical basis in Section III and offer a detailed example for illuminating the execution process of HP-PEF. In Section V, we evaluate the performance of our method by implementing computer programs. Section VI concludes this paper.

组织结构：本文的其余部分组织如下。在第二部分，我们提供了本文中使用的预备知识。在第三部分，我们首先给出了 PEF 模型的定义，然后给出了在 PEF 模型下 \( {Q}_{n}^{k} \) 中 H-路径存在的理论证明。在第四部分，我们基于第三部分的理论基础设计了容错 H-路径嵌入算法 HP-PEF，并给出了一个详细的例子来阐明 HP-PEF 的执行过程。在第五部分，我们通过实现计算机程序来评估我们的方法的性能。第六部分总结本文。

## II. Preliminaries

## II. 预备知识

## A. Terminologies and Notations

## A. 术语和符号

For terminologies and notations not defined in this subsection, please refer to the reference [40]. An interconnection network can be modeled as a graph \( G = \left( {V\left( G\right) ,E\left( G\right) }\right) \) ,where \( V\left( G\right) \) represents its node set and \( E\left( G\right) \) represents its edge set. The notations \( \left| {V\left( G\right) }\right| \) and \( \left| {E\left( G\right) }\right| \) denote the size of \( V\left( G\right) \) and \( E\left( G\right) \) , respectively. Given a graph \( S \) ,if \( V\left( S\right)  \subseteq  V\left( G\right) \) and \( E\left( S\right)  \subseteq \) \( E\left( G\right) \) ,then \( S \) is a subgraph of \( G \) . Given a node set \( M \subseteq  V\left( G\right) \) , the subgraph of \( G \) induced by \( M \) ,denoted by \( G\left\lbrack  M\right\rbrack \) ,is a graph with the node set \( M \) and edge set \( \{ \left( {u,v}\right)  \in  E\left( G\right)  \mid  u,v \in  M\} \) . Let \( F \) be a faulty edge set of \( G \) ,and \( G - F \) be the graph with the node set \( V\left( G\right) \) and the edge set \( E\left( G\right)  - F \) . Given a positive integer \( n \) ,we denote the set \( \{ 1,2,\ldots ,n\} \) as \( \left\lbrack  n\right\rbrack \) . Moreover,let \( {\mathbb{Z}}_{n} = \left\lbrack  {n - 1}\right\rbrack   \cup  \{ 0\} \) when \( n \geq  2 \) and \( {\mathbb{Z}}_{1} = \{ 0\} \) . A graph \( P = \) \( \left( {{v}_{0},{v}_{1},\ldots ,{v}_{p}}\right) \) is called a path if \( p + 1 \) nodes \( {v}_{0},{v}_{1},\ldots ,{v}_{p} \) are distinct and \( \left( {{v}_{i},{v}_{i + 1}}\right) \) is an edge of \( P \) with \( i \in  {\mathbb{Z}}_{p} \) . The length of \( P \) is the number of the edges in \( P \) . If \( V\left( P\right)  = V\left( G\right) \) ,then \( P \) is a Hamiltonian path (H-path for short) of \( G \) .

对于本节未定义的术语和符号，请参考参考文献 [40]。互联网络可以被建模为一个图 \( G = \left( {V\left( G\right) ,E\left( G\right) }\right) \)，其中 \( V\left( G\right) \) 表示它的节点集合，\( E\left( G\right) \) 表示它的边集合。\( \left| {V\left( G\right) }\right| \) 和 \( \left| {E\left( G\right) }\right| \) 分别表示 \( V\left( G\right) \) 和 \( E\left( G\right) \) 的大小。给定一个图 \( S \)，如果 \( V\left( S\right)  \subseteq  V\left( G\right) \) 和 \( E\left( S\right)  \subseteq \) \( E\left( G\right) \)，那么 \( S \) 是 \( G \) 的子图。给定一个节点集合 \( M \subseteq  V\left( G\right) \)，由 \( M \) 引发的 \( G \) 的子图，记作 \( G\left\lbrack  M\right\rbrack \)，是一个具有节点集合 \( M \) 和边集合 \( \{ \left( {u,v}\right)  \in  E\left( G\right)  \mid  u,v \in  M\} \) 的图。设 \( F \) 为 \( G \) 的故障边集合，\( G - F \) 为具有节点集合 \( V\left( G\right) \) 和边集合 \( E\left( G\right)  - F \) 的图。给定一个正整数 \( n \)，我们记集合 \( \{ 1,2,\ldots ,n\} \) 为 \( \left\lbrack  n\right\rbrack \)。此外，当 \( {\mathbb{Z}}_{n} = \left\lbrack  {n - 1}\right\rbrack   \cup  \{ 0\} \) 时，设 \( n \geq  2 \) 和 \( {\mathbb{Z}}_{1} = \{ 0\} \)。一个图 \( P = \) \( \left( {{v}_{0},{v}_{1},\ldots ,{v}_{p}}\right) \) 被称为路径，如果 \( p + 1 \) 节点 \( {v}_{0},{v}_{1},\ldots ,{v}_{p} \) 是不同的，并且 \( \left( {{v}_{i},{v}_{i + 1}}\right) \) 是 \( P \) 的一条边，且 \( i \in  {\mathbb{Z}}_{p} \)。\( P \) 的长度是 \( P \) 中的边数。如果 \( V\left( P\right)  = V\left( G\right) \)，那么 \( P \) 是 \( G \) 的一条哈密顿路径（简称 H-路径）。

## B. \( k \) -Ary \( n \) -Cube \( {Q}_{n}^{k} \)

## B. \( k \) -元 \( n \) -立方体 \( {Q}_{n}^{k} \)

Definition II.1. (See [33]). The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is a graph with the node set \( V\left( {Q}_{n}^{k}\right)  = \{ 0,1,\ldots ,k - 1{\} }^{n} \) such that two nodes \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{0} \) and \( v = {v}_{n - 1}{v}_{n - 2}\cdots {v}_{0} \) are adjacent in \( {Q}_{n}^{k} \) if and only if there is an integer \( i \in  {\mathbb{Z}}_{n} \) satisfying \( {u}_{i} = {v}_{i} \pm  1\left( {\;\operatorname{mod}\;k}\right) \) and \( {u}_{j} = {v}_{j} \) for every \( j \in  {\mathbb{Z}}_{n} - \{ i\} \) . In this case,such an edge(u,v)is called an \( i \) -dimensional edge for \( i \in  {\mathbb{Z}}_{n} \) ,and the set of all \( i \) -dimensional edges of \( {Q}_{n}^{k} \) is denoted by \( {E}_{i}\left( {Q}_{n}^{k}\right) \) ,or \( {E}_{i} \) for short.

定义II.1（见[33]）。\( k \) -元 \( n \) -立方体 \( {Q}_{n}^{k} \) 是一个图，其节点集为 \( V\left( {Q}_{n}^{k}\right)  = \{ 0,1,\ldots ,k - 1{\} }^{n} \)，当且仅当存在一个整数 \( i \in  {\mathbb{Z}}_{n} \) 满足 \( {u}_{i} = {v}_{i} \pm  1\left( {\;\operatorname{mod}\;k}\right) \) 和 \( {u}_{j} = {v}_{j} \) 对于每一个 \( j \in  {\mathbb{Z}}_{n} - \{ i\} \) 成立时，两个节点 \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{0} \) 和 \( v = {v}_{n - 1}{v}_{n - 2}\cdots {v}_{0} \) 在 \( {Q}_{n}^{k} \) 中是相邻的。在这种情况下，这样的边 (u,v) 被称为 \( i \) -维边对于 \( i \in  {\mathbb{Z}}_{n} \)，而 \( {Q}_{n}^{k} \) 的所有 \( i \) -维边的集合表示为 \( {E}_{i}\left( {Q}_{n}^{k}\right) \)，简称为 \( {E}_{i} \)。

Hereafter,for brevity,we will omit "(mod \( k \) )" in a situation similar to the above definition. By Definition II.1, \( {Q}_{n}^{k} \) is \( {2n} \) -regular and contains \( {k}^{n} \) nodes. In addition, \( {Q}_{n}^{k} \) is edge symmetric [41]. The \( {Q}_{n}^{k} \) can be partitioned into \( k \) disjoint subgraphs \( {Q}_{n}^{k}\left\lbrack  0\right\rbrack  ,{Q}_{n}^{k}\left\lbrack  1\right\rbrack  ,\ldots ,{Q}_{n}^{k}\left\lbrack  {k - 1}\right\rbrack \) (abbreviated as \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack  ) \) along the \( i \) -dimension for \( i \in \) \( {\mathbb{Z}}_{n} \) . All these \( k \) subgraphs are isomorphic to \( {Q}_{n - 1}^{k} \) . Given a faulty edge set \( F \) ,let \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack   = \{ \left( {u,v}\right)  \mid  u \in  V\left( {Q\left\lbrack  l\right\rbrack  }\right) ,v \in \) \( \left. {V\left( {Q\left\lbrack  {l + 1}\right\rbrack  }\right) \text{and}\left( {u,v}\right)  \in  F \cap  {E}_{i}}\right\} \) . If there is no ambiguity, we abbreviate \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack \) to \( F\left\lbrack  {l,l + 1}\right\rbrack \) . Each node of \( Q\left\lbrack  l\right\rbrack \) has the form \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}l{u}_{i - 1}\cdots {u}_{0} \) . The node \( v = \) \( {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}{l}^{\prime }{u}_{i - 1}\cdots {u}_{0} \) is the neighbor of \( u \) in \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) if \( {l}^{\prime } = l \pm  1 \) ,which is denoted by \( {n}^{{l}^{\prime }}\left( u\right) \) . To distinguish the positions of the subgraphs where different nodes are located, let \( {l}_{u} = {u}_{i} = l \) . That is, \( {l}_{v} = {l}_{u} \pm  1 \) . Although the values of \( {l}_{u} \) and \( {u}_{i} \) are equal,the notation \( {l}_{u} \) mainly focuses on the position \( l \) rather than the dimension \( i \) of node \( u \) . Let \( {Q}_{n}^{k}\left\lbrack  {\ell ,h}\right\rbrack \) (abbreviated as \( Q\left\lbrack  {\ell ,h}\right\rbrack \) ) be the subgraph induced by node set \( \{ u \mid  u \in \) \( V\left( {Q\left\lbrack  j\right\rbrack  }\right) \) with \( j = \ell ,\ell  + 1,\ldots ,h - 1,h\} \) . Moreover,we have \( Q\left\lbrack  {\ell ,h}\right\rbrack   = Q\left\lbrack  \ell \right\rbrack \) when \( \ell  = h \) ,and taken modulo \( k \) ,we have \( Q\left\lbrack  {\ell ,h}\right\rbrack   = {Q}_{n}^{k} \) when \( h = \ell  - 1 \) . A path in \( Q\left\lbrack  {\ell ,h}\right\rbrack \) is denoted by \( {P}_{\ell ,h} \) with \( V\left( {P}_{\ell ,h}\right)  \subseteq  V\left( {Q\left\lbrack  {\ell ,h}\right\rbrack  }\right) \) . For convenience,we abbreviate \( {P}_{\ell ,h} \) to \( {P}_{\ell } \) when \( \ell  = h \) ,and to \( P \) when \( h = \ell  - 1 \) by taking modulo \( k \) .

此后，为了简洁，我们将在类似于上述定义的情况下省略 "(mod \( k \))"。根据定义II.1，\( {Q}_{n}^{k} \) 是 \( {2n} \) -正则的，并且包含 \( {k}^{n} \) 个节点。此外，\( {Q}_{n}^{k} \) 是边对称的 [41]。\( {Q}_{n}^{k} \) 可以被划分为 \( k \) 个不相交的子图 \( {Q}_{n}^{k}\left\lbrack  0\right\rbrack  ,{Q}_{n}^{k}\left\lbrack  1\right\rbrack  ,\ldots ,{Q}_{n}^{k}\left\lbrack  {k - 1}\right\rbrack \)（简写为 \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack  ) \) 沿着 \( i \) 维度对于 \( i \in \) \( {\mathbb{Z}}_{n} \)）。所有这些 \( k \) 个子图都与 \( {Q}_{n - 1}^{k} \) 同构。给定一个故障边集合 \( F \)，设 \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack   = \{ \left( {u,v}\right)  \mid  u \in  V\left( {Q\left\lbrack  l\right\rbrack  }\right) ,v \in \) \( \left. {V\left( {Q\left\lbrack  {l + 1}\right\rbrack  }\right) \text{and}\left( {u,v}\right)  \in  F \cap  {E}_{i}}\right\} \)。如果没有歧义，我们简写 \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack \) 为 \( F\left\lbrack  {l,l + 1}\right\rbrack \)。\( Q\left\lbrack  l\right\rbrack \) 的每个节点形式为 \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}l{u}_{i - 1}\cdots {u}_{0} \)。如果 \( {l}^{\prime } = l \pm  1 \)，则节点 \( v = \) \( {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}{l}^{\prime }{u}_{i - 1}\cdots {u}_{0} \) 是 \( u \) 在 \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) 中的邻居，表示为 \( {n}^{{l}^{\prime }}\left( u\right) \)。为了区分不同节点所在的子图位置，设 \( {l}_{u} = {u}_{i} = l \)。即，\( {l}_{v} = {l}_{u} \pm  1 \)。尽管 \( {l}_{u} \) 和 \( {u}_{i} \) 的值相等，但 \( {l}_{u} \) 的符号主要关注位置 \( l \) 而不是节点 \( u \) 的维度 \( i \)。设 \( {Q}_{n}^{k}\left\lbrack  {\ell ,h}\right\rbrack \)（简写为 \( Q\left\lbrack  {\ell ,h}\right\rbrack \)）是由节点集 \( \{ u \mid  u \in \) \( V\left( {Q\left\lbrack  j\right\rbrack  }\right) \) 诱导的子图，并带有 \( j = \ell ,\ell  + 1,\ldots ,h - 1,h\} \)。此外，当 \( \ell  = h \) 时，我们有 \( Q\left\lbrack  {\ell ,h}\right\rbrack   = Q\left\lbrack  \ell \right\rbrack \)，并且取模 \( k \)，当 \( h = \ell  - 1 \) 时，我们有 \( Q\left\lbrack  {\ell ,h}\right\rbrack   = {Q}_{n}^{k} \)。在 \( Q\left\lbrack  {\ell ,h}\right\rbrack \) 中的路径表示为 \( {P}_{\ell ,h} \) \( V\left( {P}_{\ell ,h}\right)  \subseteq  V\left( {Q\left\lbrack  {\ell ,h}\right\rbrack  }\right) \)。为了方便，当 \( \ell  = h \) 时，我们简写 \( {P}_{\ell ,h} \) 为 \( {P}_{\ell } \)，当取模 \( k \) 且 \( h = \ell  - 1 \) 时，简写为 \( P \)。

Fig. 1 shows \( {Q}_{1}^{3},{Q}_{2}^{3} \) ,and \( {Q}_{3}^{3} \) . We color each edge according to its dimension. In Fig. 1(c),we partition \( {Q}_{3}^{3} \) into 3 disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ,and \( Q\left\lbrack  2\right\rbrack \) along the 2-dimension. The subgraph \( Q\left\lbrack  {0,1}\right\rbrack \) is induced by node set \( \{ u \mid  u \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right)  \cup \) \( V\left( {Q\left\lbrack  1\right\rbrack  }\right) \} \) . The node \( u = {110} \in  V\left( {Q\left\lbrack  1\right\rbrack  }\right) \) is adjacent to two nodes \( v = {010} \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) and \( w = {210} \in  V\left( {Q\left\lbrack  2\right\rbrack  }\right) \) . In addition, \( v = {n}^{0}\left( u\right) \) and \( w = {n}^{2}\left( u\right) \) . It’s easy to see that \( {l}_{v} = \) \( {l}_{u} - 1 = 0 \) and \( {l}_{w} = {l}_{u} + 1 = 2 \) .

图 1 显示了 \( {Q}_{1}^{3},{Q}_{2}^{3} \) 和 \( {Q}_{3}^{3} \) 。我们根据边的维度为每条边着色。在图 1(c) 中，我们将 \( {Q}_{3}^{3} \) 划分为 3 个不相交的子图 \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ，并沿着 2 维进行 \( Q\left\lbrack  2\right\rbrack \) 。子图 \( Q\left\lbrack  {0,1}\right\rbrack \) 由节点集 \( \{ u \mid  u \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right)  \cup \) \( V\left( {Q\left\lbrack  1\right\rbrack  }\right) \} \) 诱导。节点 \( u = {110} \in  V\left( {Q\left\lbrack  1\right\rbrack  }\right) \) 与两个节点 \( v = {010} \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) 和 \( w = {210} \in  V\left( {Q\left\lbrack  2\right\rbrack  }\right) \) 相邻。此外，\( v = {n}^{0}\left( u\right) \) 和 \( w = {n}^{2}\left( u\right) \) 。很容易看出 \( {l}_{v} = \) \( {l}_{u} - 1 = 0 \) 和 \( {l}_{w} = {l}_{u} + 1 = 2 \) 。

In particular,the \( {Q}_{2}^{k} \) can be deemed a \( k \times  k \) grid with wraparound edges,where a node \( {u}_{i,j} = {ij} \) is indexed by its row \( i \) and column \( j \) . Let \( p,q \in  {\mathbb{Z}}_{k} \) be two row indices with \( p \neq  q \) . If \( p < q \) ,we define the row torus \( {rt}\left( {p,q}\right) \) to be the subgraph of \( {Q}_{2}^{k} \) induced by the nodes on rows \( p,p + 1,\ldots ,q \) ,and particularly, all column edges between nodes on row \( p \) and row \( q \) are removed when \( p = 0 \) and \( q = k - 1 \) . Otherwise,if \( p > q \) ,we define the row torus \( {rt}\left( {p,q}\right) \) to be the subgraph of \( {Q}_{2}^{k} \) induced by the nodes on rows \( p,p + 1,\ldots ,k - 1,0,\ldots ,q \) ,and particularly,all column edges between nodes on row \( p \) and row \( q \) are removed when \( p = q + 1 \) . Fig. 2 depicts the \( {rt}\left( {0,4}\right) \) ,which is obtained by removing the column edges between nodes on row 0 and row 4 from \( {Q}_{2}^{5} \) . Throughout,we assume that the addition of row or column indices is modulo \( k \) .

特别是，\( {Q}_{2}^{k} \) 可以被认为是一个带有环绕边的 \( k \times  k \) 网格，其中节点 \( {u}_{i,j} = {ij} \) 通过其行 \( i \) 和列 \( j \) 索引。设 \( p,q \in  {\mathbb{Z}}_{k} \) 为两个行索引，满足 \( p \neq  q \) 。如果 \( p < q \) ，我们定义行环面 \( {rt}\left( {p,q}\right) \) 为由 \( {Q}_{2}^{k} \) 上的行节点诱导的子图，特别是，当 \( p = 0 \) 和 \( q = k - 1 \) 时，移除行 \( p \) 和行 \( q \) 之间的所有列边。否则，如果 \( p > q \) ，我们定义行环面 \( {rt}\left( {p,q}\right) \) 为由 \( {Q}_{2}^{k} \) 上的行节点诱导的子图，特别是，当 \( p = q + 1 \) 时，移除行 \( p \) 和行 \( q \) 之间的所有列边。图 2 描述了 \( {rt}\left( {0,4}\right) \) ，这是通过从 \( {Q}_{2}^{5} \) 中移除行 0 和行 4 之间的列边得到的。在整个过程中，我们假设行或列索引的加法是模 \( k \) 的。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_3.jpg?x=123&y=184&w=1463&h=571"/>

Fig. 1. The structures of (a) \( {Q}_{1}^{3} \) ; (b) \( {Q}_{2}^{3} \) ; (c) \( {Q}_{3}^{3} \) .

图 1. (a) \( {Q}_{1}^{3} \) 的结构；（b）\( {Q}_{2}^{3} \) 的结构；（c）\( {Q}_{3}^{3} \) 的结构。

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_3.jpg?x=162&y=885&w=635&h=409"/>

Fig. 2. The structure of \( {rt}\left( {0,4}\right) \) .

图 2. \( {rt}\left( {0,4}\right) \) 的结构。

<!-- Media -->

For the row torus \( {rt}\left( {p,q}\right) \) with \( q - p = 1 \) ,we define the four types of paths in \( {rt}\left( {p,q}\right) \) as follows. The notations of these paths are derived from the shape of their pictorial representations, where \( \bar{i} = q + p - i \) .

对于具有 \( q - p = 1 \) 的行环面 \( {rt}\left( {p,q}\right) \)，我们定义 \( {rt}\left( {p,q}\right) \) 中的四种路径如下。这些路径的表示法来源于它们图形表示的形状，其中 \( \bar{i} = q + p - i \)。

\[{N}^{ + }\left( {{u}_{i,j},{u}_{i,{j}^{\prime }}}\right)  = \left( {{u}_{i,j},{u}_{\bar{i},j},{u}_{\bar{i},j + 1},{u}_{i,j + 1},{u}_{i,j + 2},{u}_{\bar{i},j + 2},}\right. \]

\[\left. {{u}_{\bar{i},j + 3},{u}_{i,j + 3},{u}_{i,j + 4},\ldots ,{u}_{i,{j}^{\prime } - 1},{u}_{i,{j}^{\prime }}}\right) \]

\[\text{where}i \in  \{ p,q\} ,0 \leq  j \neq  {j}^{\prime } \leq  k - 1\text{,}\]

\[\text{and}\left| {j - {j}^{\prime }}\right| \text{is even.}\]

\[{N}^{ - }\left( {{u}_{i,j},{u}_{i,{j}^{\prime }}}\right)  = \left( {{u}_{i,j},{u}_{\bar{i},j},{u}_{\bar{i},j - 1},{u}_{i,j - 1},{u}_{i,j - 2},{u}_{\bar{i},j - 2},}\right. \]

\[\left. {{u}_{\bar{i},j - 3},{u}_{i,j - 3},{u}_{i,j - 4},\ldots ,{u}_{i,{j}^{\prime } + 1},{u}_{i,{j}^{\prime }}}\right) \]

\[\text{where}i \in  \{ p,q\} ,0 \leq  j \neq  {j}^{\prime } \leq  k - 1\text{,}\]

\[\text{and}\left| {j - {j}^{\prime }}\right| \text{is even.}\]

\[{C}_{m}^{ + }\left( {{u}_{i,j},{u}_{\bar{i},j}}\right)  = \left( {{u}_{i,j},{u}_{i,j + 1},{u}_{i,j + 2},\ldots ,{u}_{i,m - 1},{u}_{i,m},}\right. \]

\[\left. {{u}_{\bar{i},m},{u}_{\bar{i},m - 1},{u}_{\bar{i},m - 2},\ldots ,{u}_{\bar{i},j + 1},{u}_{\bar{i},j}}\right) \]

\[\text{where}i \in  \{ p,q\} ,0 \leq  j \leq  k - 1\text{,and}\]

\[0 \leq  m \leq  k - 1\]

\[{C}_{m}^{ - }\left( {{u}_{i,j},{u}_{i,j}}\right)  = \left( {{u}_{i,j},{u}_{i,j - 1},{u}_{i,j - 2},\ldots ,{u}_{i,m + 1},{u}_{i,m},}\right. \]

\[\left. {{u}_{\bar{i},m},{u}_{\bar{i},m + 1},{u}_{\bar{i},m + 2},\ldots ,{u}_{\bar{i},j - 1},{u}_{\bar{i},j}}\right) \]

\[\text{where}i \in  \{ p,q\} ,0 \leq  j \leq  k - 1\text{,and}\]

\[0 \leq  m \leq  k - 1\]

In particular,if \( m = j \) ,then \( {C}_{m}^{ + }\left( {{u}_{i,j},{u}_{\bar{i},j}}\right)  = \) \( {C}_{m}^{ - }\left( {{u}_{i,j},{u}_{\bar{i},j}}\right)  = \left( {{u}_{i,j},{u}_{\bar{i},j}}\right) . \)

特别地，如果 \( m = j \)，那么 \( {C}_{m}^{ + }\left( {{u}_{i,j},{u}_{\bar{i},j}}\right)  = \) \( {C}_{m}^{ - }\left( {{u}_{i,j},{u}_{\bar{i},j}}\right)  = \left( {{u}_{i,j},{u}_{\bar{i},j}}\right) . \)。

## III. THEORETICAL BASIS FOR EMBEDDING THE HAMILTONIAN PATH INTO \( k \) -ARY \( n \) -CUBES

## III. 将 Hamiltonian 路径嵌入 \( k \) 元 \( n \) 立方体的理论依据

In this section, we establish the theoretical basis for embedding the H-path into \( k \) -ary \( n \) -cubes under the PEF model. That is,we will prove that an \( \mathrm{H} \) -path can be found in a \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) in the presence of a partitioned edge fault set,described below.

在这一节中，我们建立了在PEF模型下将H路径嵌入 \( k \) 元 \( n \) 立方体的理论依据。也就是说，我们将证明在存在如下描述的分隔边故障集的情况下，可以在 \( k \) 元 \( n \) 立方体 \( {Q}_{n}^{k} \) 中找到一个 \( \mathrm{H} \) 路径。

Let \( F \) be a faulty edge set in \( {Q}_{n}^{k} \) ,and let \( {F}_{i} = F \cap  {E}_{i} \) with \( i \in  {\mathbb{Z}}_{n} \) . We set \( \left\{  {{e}_{0},{e}_{1},\ldots ,{e}_{n - 1}}\right\}   = \left\{  {\left| {F}_{i}\right|  \mid  i \in  {\mathbb{Z}}_{n}}\right\} \) such that \( {e}_{n - 1} \geq  {e}_{n - 2} \geq  \cdots  \geq  {e}_{0} \) . The faulty edge set \( F \) is a partitioned edge fault set (PEF set for short) if and only if \( {e}_{i} \leq  f\left( i\right)  < \) \( \frac{\left| E\left( {Q}_{n}^{k}\right) \right| }{n} = {k}^{n} \) for each \( i \in  {\mathbb{Z}}_{n} \) ,where \( f\left( i\right) \) is a function of \( i \) or a fixed value.

设 \( F \) 是 \( {Q}_{n}^{k} \) 中的一个故障边集，且设 \( {F}_{i} = F \cap  {E}_{i} \) 满足 \( i \in  {\mathbb{Z}}_{n} \)。我们设定 \( \left\{  {{e}_{0},{e}_{1},\ldots ,{e}_{n - 1}}\right\}   = \left\{  {\left| {F}_{i}\right|  \mid  i \in  {\mathbb{Z}}_{n}}\right\} \) 使得 \( {e}_{n - 1} \geq  {e}_{n - 2} \geq  \cdots  \geq  {e}_{0} \)。故障边集 \( F \) 是一个分隔边故障集（简称为PEF集），当且仅当对于每个 \( i \in  {\mathbb{Z}}_{n} \)，都有 \( {e}_{i} \leq  f\left( i\right)  < \) \( \frac{\left| E\left( {Q}_{n}^{k}\right) \right| }{n} = {k}^{n} \)，其中 \( f\left( i\right) \) 是 \( i \) 的函数或一个固定值。

For a PEF set \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) ,since \( {Q}_{n}^{k} \) is recursively constructed with edge symmetry, we can utilize the inductive method to analyze the Hamiltonian property of \( {Q}_{n}^{k} \) by partitioning it into \( k \) subgraphs along a dimension we expected. Consequently,provided that \( {Q}_{n}^{k} - F \) is Hamiltonian-connected, the exact values of \( f\left( i\right) \) can be derived by analyzing the number of faulty edges that can be tolerated when the H-path passes through two consecutive subgraphs.

对于一个PEF集合 \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) ，由于 \( {Q}_{n}^{k} \) 是通过边对称递归构造的，我们可以利用归纳法通过沿预期维度的 \( k \) 子图划分来分析 \( {Q}_{n}^{k} \) 的哈密顿性质。因此，如果 \( {Q}_{n}^{k} - F \) 是哈密顿连通的，那么通过分析当哈密顿路径经过两个连续子图时可以容忍的故障边数，可以推导出 \( f\left( i\right) \) 的确切值。

The following result is provided for dealing with the base case of our forthcoming inductive proof.

下面给出的结果是用来处理我们即将进行的归纳证明的基例。

Lemma III.1. (See [42]) For odd \( k \geq  3 \) ,let \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) with \( \left| F\right|  \leq  1 \) . Then \( {Q}_{2}^{k} - F \) is Hamiltonian-connected.

引理III.1。（见[42]）对于奇数 \( k \geq  3 \) ，设 \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) 满足 \( \left| F\right|  \leq  1 \) 。那么 \( {Q}_{2}^{k} - F \) 是哈密顿连通的。

Theorem III.2. For \( n \geq  2 \) and odd \( k \geq  3 \) ,let \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) be a PEF set satisfying the following conditions:

定理III.2。对于 \( n \geq  2 \) 和奇数 \( k \geq  3 \) ，设 \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) 是一个满足以下条件的PEF集合：

1) \( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) ;

2) \( {e}_{i} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ;

2) \( {e}_{i} \leq  {k}^{i} - 2 \) 对于每个 \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ；

3) \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) .

3) \( {e}_{0} = 0 \) 和 \( {e}_{1} \leq  1 \) 。

Then, \( {Q}_{n}^{k} - F \) is Hamiltonian-connected.

那么， \( {Q}_{n}^{k} - F \) 是哈密顿连通的。

Proof. The proof is by induction on \( n \) . When \( n = 2 \) ,by Lemma III.1,the theorem holds obviously. For \( n \geq  3 \) ,assume this theorem holds for all \( {Q}_{m}^{k} \) ’s with \( m < n \) . Therefore,what we need to prove is that this theorem holds for \( {Q}_{n}^{k} \) .

证明。证明采用对 \( n \) 的归纳法。当 \( n = 2 \) 时，由引理III.1可知，定理显然成立。对于 \( n \geq  3 \) ，假设定理对于所有 \( {Q}_{m}^{k} \) 且 \( m < n \) 成立。因此，我们需要证明的是定理对于 \( {Q}_{n}^{k} \) 也成立。

Since \( {Q}_{n}^{k} \) is edge symmetric,without loss of generality, let \( \left| {F}_{n - 1}\right|  = \max \left\{  {\left| {F}_{n - 1}\right| ,\left| {F}_{n - 2}\right| ,\ldots ,\left| {F}_{0}\right| }\right\} \) . That is, \( \left| {F}_{n - 1}\right|  = \) \( {e}_{n - 1} \leq  {k}^{n - 1} - 2 \) . Along the(n - 1)-dimension,we divide \( {Q}_{n}^{k} \) into \( k \) disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) ,all of which are isomorphic to \( {Q}_{n - 1}^{k} \) . Let \( s \) and \( t \) be arbitrary two vertices of \( {Q}_{n}^{k} \) with \( s \in  V\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) and \( t \in  V\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) . By the arbitrariness of \( s \) and \( t \) ,suppose that \( {l}_{s} \leq  {l}_{t} \) .

由于 \( {Q}_{n}^{k} \) 是边缘对称的，不失一般性，设 \( \left| {F}_{n - 1}\right|  = \max \left\{  {\left| {F}_{n - 1}\right| ,\left| {F}_{n - 2}\right| ,\ldots ,\left| {F}_{0}\right| }\right\} \) 。即 \( \left| {F}_{n - 1}\right|  = \) \( {e}_{n - 1} \leq  {k}^{n - 1} - 2 \) 。在 (n - 1) 维上，我们将 \( {Q}_{n}^{k} \) 分成 \( k \) 个不相交的子图 \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) ，它们都与 \( {Q}_{n - 1}^{k} \) 同构。设 \( s \) 和 \( t \) 是 \( {Q}_{n}^{k} \) 中任意两个顶点，满足 \( s \in  V\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) 和 \( t \in  V\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) 。由于 \( s \) 和 \( t \) 的任意性，假设 \( {l}_{s} \leq  {l}_{t} \) 。

Let \( {C}_{h}^{l} = E\left( {Q\left\lbrack  l\right\rbrack  }\right)  \cap  {F}_{h} \) with \( l \in  {\mathbb{Z}}_{k} \) and \( h \in  {\mathbb{Z}}_{n - 1} \) . Moreover,for each \( l \in  {\mathbb{Z}}_{k} \) ,let \( \left\{  {{e}_{n - 2}^{l},{e}_{n - 3}^{l},\ldots ,{e}_{0}^{l}}\right\}   = \) \( \left\{  {\left| {C}_{n - 2}^{l}\right| ,\left| {C}_{n - 3}^{l}\right| ,\ldots ,\left| {C}_{0}^{l}\right| }\right\} \) such that \( {e}_{n - 2}^{l} \geq  {e}_{n - 3}^{l} \geq  \cdots  \geq  {e}_{0}^{l} \) . According to the recursive nature and conditions 2) and 3), when \( n = 3 \) ,we have \( \left| F\right|  - \left| {F}_{n - 1}\right|  = \mathop{\sum }\limits_{{i = 0}}^{1}\left| {F}_{i}\right|  \leq  1,{e}_{0}^{l} = 0 \) , and \( {e}_{1}^{l} \leq  1 \) . When \( n \geq  4 \) ,we have

设 \( {C}_{h}^{l} = E\left( {Q\left\lbrack  l\right\rbrack  }\right)  \cap  {F}_{h} \) 满足 \( l \in  {\mathbb{Z}}_{k} \) 和 \( h \in  {\mathbb{Z}}_{n - 1} \) 。此外，对于每个 \( l \in  {\mathbb{Z}}_{k} \) ，设 \( \left\{  {{e}_{n - 2}^{l},{e}_{n - 3}^{l},\ldots ,{e}_{0}^{l}}\right\}   = \) \( \left\{  {\left| {C}_{n - 2}^{l}\right| ,\left| {C}_{n - 3}^{l}\right| ,\ldots ,\left| {C}_{0}^{l}\right| }\right\} \) 使得 \( {e}_{n - 2}^{l} \geq  {e}_{n - 3}^{l} \geq  \cdots  \geq  {e}_{0}^{l} \) 。根据递归性质以及条件 2) 和 3)，当 \( n = 3 \) 时，我们有 \( \left| F\right|  - \left| {F}_{n - 1}\right|  = \mathop{\sum }\limits_{{i = 0}}^{1}\left| {F}_{i}\right|  \leq  1,{e}_{0}^{l} = 0 \) ，以及 \( {e}_{1}^{l} \leq  1 \) 。当 \( n \geq  4 \) 时，我们有

\[\left| F\right|  - \left| {F}_{n - 1}\right|  = \mathop{\sum }\limits_{{i = 0}}^{{n - 2}}\left| {F}_{i}\right|  = \mathop{\sum }\limits_{{i = 2}}^{{n - 2}}{e}_{i} + \mathop{\sum }\limits_{{i = 0}}^{1}{e}_{i}\]

\[ \leq  \mathop{\sum }\limits_{{i = 2}}^{{n - 2}}\left( {{k}^{i} - 2}\right)  + 1\]

\[ = \frac{{k}^{n - 1} - {k}^{2}}{k - 1} - 2\left( {n - 1}\right)  + 5.\]

In addition, \( {e}_{0}^{l} = 0,{e}_{1}^{l} \leq  1 \) ,and \( {e}_{i}^{l} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n - 1} - {\mathbb{Z}}_{2} \) . Therefore,every \( Q\left\lbrack  l\right\rbrack   - F \) with \( l \in  {\mathbb{Z}}_{k} \) is Hamiltonian-connected. That is,when \( {l}_{s} = {l}_{t} \) ,there exists an \( \mathrm{H} \) -path in \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) between \( s \) and \( t \) .

此外，\( {e}_{0}^{l} = 0,{e}_{1}^{l} \leq  1 \) ，且对于每个 \( i \in  {\mathbb{Z}}_{n - 1} - {\mathbb{Z}}_{2} \) ，\( {e}_{i}^{l} \leq  {k}^{i} - 2 \) 。因此，每个 \( Q\left\lbrack  l\right\rbrack   - F \) 与 \( l \in  {\mathbb{Z}}_{k} \) 的图都是哈密顿连通的。即，当 \( {l}_{s} = {l}_{t} \) 时，在 \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) 中存在一条连接 \( s \) 和 \( t \) 的 \( \mathrm{H} \) -路径。

Without loss of generality,suppose that \( \left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right|  = \) \( \max \{ \left| {F\left\lbrack  {0,1}\right\rbrack  }\right| ,\ldots ,\left| {F\left\lbrack  {k - 2,k - 1}\right\rbrack  }\right| ,\left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right| \} .\; \) Since \( \mathop{\sum }\limits_{{l = 0}}^{{k - 1}}\left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  = \left| {F}_{n - 1}\right|  \leq  {k}^{n - 1} - 2 \) and \( k \geq  3 \) is odd, \( \left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  \leq  \left\lfloor  \frac{{k}^{n - 1} - 2}{2}\right\rfloor   = \frac{{k}^{n - 1} - 3}{2} \) for all \( l \in  {\mathbb{Z}}_{k - 1} \) .

不失一般性，假设 \( \left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right|  = \) \( \max \{ \left| {F\left\lbrack  {0,1}\right\rbrack  }\right| ,\ldots ,\left| {F\left\lbrack  {k - 2,k - 1}\right\rbrack  }\right| ,\left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right| \} .\; \) 。由于 \( \mathop{\sum }\limits_{{l = 0}}^{{k - 1}}\left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  = \left| {F}_{n - 1}\right|  \leq  {k}^{n - 1} - 2 \) 且 \( k \geq  3 \) 是奇数，因此 \( \left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  \leq  \left\lfloor  \frac{{k}^{n - 1} - 2}{2}\right\rfloor   = \frac{{k}^{n - 1} - 3}{2} \) 对所有 \( l \in  {\mathbb{Z}}_{k - 1} \) 成立。

Claim 1: Suppose that there exists an \( \mathrm{H} \) -path \( {P}_{q} \) in \( Q\left\lbrack  q\right\rbrack   - F \) between any two distinct nodes in \( Q\left\lbrack  q\right\rbrack \) . When \( 0 \leq  q \leq  k - \)

引理 1：假设在 \( Q\left\lbrack  q\right\rbrack \) 中任意两个不同节点之间存在一条 \( \mathrm{H} \) -路径 \( {P}_{q} \) 。当 \( 0 \leq  q \leq  k - \)

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_4.jpg?x=980&y=182&w=578&h=347"/>

Fig. 3. The constructions in Case 1.1 of Theorem III.2.

图 3。定理 III.2 中情形 1.1 的构造。

<!-- Media -->

2,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . And when \( 1 \leq  q \leq  k - \) 1,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) .

存在至少一条边 \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) 使得 \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。当 \( 1 \leq  q \leq  k - \) 1时，存在至少一条边 \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) 使得 \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。

The length of \( {P}_{q} \) is \( {k}^{n - 1} - 1 \) . Then there exist \( \frac{{k}^{n - 1} - 1}{2} \) mutually disjoint edges on \( {P}_{q} \) . When \( 0 \leq  q \leq  k - 2 \) ,since \( \frac{{k}^{n - 1} - 1}{2} - \) \( \left| {F\left\lbrack  {q,q + 1}\right\rbrack  }\right|  \geq  1 \) ,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Analogously, when \( 1 \leq  q \leq  k - 1 \) ,we can also find at least one edge \( \left( {x,{x}^{ * }}\right)  \in \) \( E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Then the claim holds.

\( {P}_{q} \) 的长度是 \( {k}^{n - 1} - 1 \) 。然后在 \( {P}_{q} \) 上存在 \( \frac{{k}^{n - 1} - 1}{2} \) 条互不相交的边。当 \( 0 \leq  q \leq  k - 2 \) 时，由于 \( \frac{{k}^{n - 1} - 1}{2} - \) \( \left| {F\left\lbrack  {q,q + 1}\right\rbrack  }\right|  \geq  1 \) ，存在至少一条边 \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) 使得 \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。类似地，当 \( 1 \leq  q \leq  k - 1 \) 时，我们也可以找到至少一条边 \( \left( {x,{x}^{ * }}\right)  \in \) \( E\left( {P}_{q}\right) \) 使得 \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。那么，该命题成立。

Next, we discuss the following cases separately.

接下来，我们分别讨论以下情况。

Case 1: \( {l}_{s} = 0 \) .

情况1：\( {l}_{s} = 0 \) 。

Case 1.1: \( {l}_{s} = {l}_{t} \) .

情况1.1：\( {l}_{s} = {l}_{t} \) 。

Since \( Q\left\lbrack  0\right\rbrack   - F \) is Hamiltonian-connected,there exists an H-path \( {P}_{0} \) in \( Q\left\lbrack  0\right\rbrack   - F \) between \( s \) and \( t \) . By Claim 1,there exists at least one edge \( \left( {x,{x}^{ * }}\right) \) of \( {P}_{0} \) such that \( \left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similarly,since \( Q\left\lbrack  1\right\rbrack   - F \) is Hamiltonian-connected,there exists an \( \mathrm{H} \) -path \( {P}_{1} \) in \( Q\left\lbrack  1\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( {n}^{1}\left( {x}^{ * }\right) \) . By Claim 1,there exists at least one edge \( \left( {y,{y}^{ * }}\right) \) of \( {P}_{1} \) such that \( \left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{2}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . By constantly iterating in this way, we can obtain an H-path \( {P}_{2,k - 1} \) in \( Q\left\lbrack  {2,k - 1}\right\rbrack   - F \) between \( {n}^{2}\left( y\right) \) and \( {n}^{2}\left( {y}^{ * }\right) \) . Thus, \( {P}_{0} \cup  {P}_{1} \cup  {P}_{2,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{n}^{1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right. \) , \( \left. {\left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{n}^{2}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required H-path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 3).

由于 \( Q\left\lbrack  0\right\rbrack   - F \) 是哈密顿连通的，存在一条从 \( s \) 到 \( t \) 的 H-路径 \( {P}_{0} \) 在 \( Q\left\lbrack  0\right\rbrack   - F \) 中。根据命题1，存在 \( {P}_{0} \) 中的至少一条边 \( \left( {x,{x}^{ * }}\right) \) 使得 \( \left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。类似地，由于 \( Q\left\lbrack  1\right\rbrack   - F \) 是哈密顿连通的，存在一条从 \( {n}^{1}\left( x\right) \) 到 \( {n}^{1}\left( {x}^{ * }\right) \) 的 \( \mathrm{H} \) -路径 \( {P}_{1} \) 在 \( Q\left\lbrack  1\right\rbrack   - F \) 中。根据命题1，存在 \( {P}_{1} \) 中的至少一条边 \( \left( {y,{y}^{ * }}\right) \) 使得 \( \left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{2}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。通过这种方式不断迭代，我们可以得到一条从 \( {n}^{2}\left( y\right) \) 到 \( {n}^{2}\left( {y}^{ * }\right) \) 的 H-路径 \( {P}_{2,k - 1} \) 在 \( Q\left\lbrack  {2,k - 1}\right\rbrack   - F \) 中。因此，\( {P}_{0} \cup  {P}_{1} \cup  {P}_{2,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{n}^{1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right. \) ，\( \left. {\left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{n}^{2}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) 构成了在 \( {Q}_{n}^{k} - F \) 中从 \( s \) 到 \( t \) 所需的 H-路径（见图3）。

Case 1.2: \( {l}_{s} \neq  {l}_{t} \) .

情况1.2：\( {l}_{s} \neq  {l}_{t} \) 。

Since \( \;\left| {V\left( {Q\left\lbrack  0\right\rbrack  }\right) }\right|  - \left| {\{ s,t\} }\right|  - \left| {F\left\lbrack  {0,1}\right\rbrack  }\right|  \geq  {k}^{n - 1} - 2 - \) \( \frac{{k}^{n - 1} - 3}{2} > 1 \) with \( n \geq  3 \) and odd \( k \geq  3 \) ,there exists one node \( x \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) such that \( x \neq  s,{n}^{1}\left( x\right)  \neq  t \) ,and \( \left( {x,{n}^{1}\left( x\right) }\right)  \notin  {F}_{n - 1} \) . Since \( Q\left\lbrack  0\right\rbrack   - F \) is Hamiltonian-connected, there exists an \( \mathrm{H} \) -path \( {P}_{0} \) in \( Q\left\lbrack  0\right\rbrack   - F \) between \( s \) and \( x \) . If \( {l}_{t} = 1 \) ,since \( Q\left\lbrack  1\right\rbrack   - F \) is Hamiltonian-connected,there exists an \( \mathrm{H} \) -path \( {P}_{1} \) in \( Q\left\lbrack  1\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( t \) ; otherwise, if \( 2 \leq  {l}_{t} \leq  k - 1 \) ,proceeding iteratively in this manner can construct an \( \mathrm{H} \) -path \( {P}_{1,{l}_{t}} \) in \( Q\left\lbrack  {1,{l}_{t}}\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( t \) . If \( {l}_{t} = k - 1 \) ,then \( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup  \left\{  \left( {x,{n}^{1}\left( x\right) }\right) \right\} \) forms the required H-path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) ; otherwise,by Claim 1, there exists one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{1,{l}_{t}}\right)  \cap  E\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) such that \( \left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similar to Case 1.1, we can construct an H-path \( {P}_{{l}_{t} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{t} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{t} + 1}\left( y\right) \) and \( {n}^{{l}_{t} + 1}\left( {y}^{ * }\right) \) . Therefore, \( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup \) \( {P}_{{l}_{t} + 1,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \) \( \left\{  \left( {y,{y}^{ * }}\right) \right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 4).

由于 \( \;\left| {V\left( {Q\left\lbrack  0\right\rbrack  }\right) }\right|  - \left| {\{ s,t\} }\right|  - \left| {F\left\lbrack  {0,1}\right\rbrack  }\right|  \geq  {k}^{n - 1} - 2 - \) \( \frac{{k}^{n - 1} - 3}{2} > 1 \) 具有 \( n \geq  3 \) 并且是奇数 \( k \geq  3 \)，存在一个节点 \( x \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) 使得 \( x \neq  s,{n}^{1}\left( x\right)  \neq  t \)，并且 \( \left( {x,{n}^{1}\left( x\right) }\right)  \notin  {F}_{n - 1} \)。由于 \( Q\left\lbrack  0\right\rbrack   - F \) 是哈密顿连通的，存在一个 \( \mathrm{H} \) -路径 \( {P}_{0} \) 在 \( Q\left\lbrack  0\right\rbrack   - F \) 中介于 \( s \) 和 \( x \) 之间。如果 \( {l}_{t} = 1 \)，由于 \( Q\left\lbrack  1\right\rbrack   - F \) 是哈密顿连通的，存在一个 \( \mathrm{H} \) -路径 \( {P}_{1} \) 在 \( Q\left\lbrack  1\right\rbrack   - F \) 中介于 \( {n}^{1}\left( x\right) \) 和 \( t \) 之间；否则，如果 \( 2 \leq  {l}_{t} \leq  k - 1 \)，以这种方式迭代进行可以构造一个 \( \mathrm{H} \) -路径 \( {P}_{1,{l}_{t}} \) 在 \( Q\left\lbrack  {1,{l}_{t}}\right\rbrack   - F \) 中介于 \( {n}^{1}\left( x\right) \) 和 \( t \) 之间。如果 \( {l}_{t} = k - 1 \)，那么 \( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup  \left\{  \left( {x,{n}^{1}\left( x\right) }\right) \right\} \) 形成了在 \( {Q}_{n}^{k} - F \) 中介于 \( s \) 和 \( t \) 之间的所需的 H-路径；否则，根据引理1，存在一条边 \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{1,{l}_{t}}\right)  \cap  E\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) 使得 \( \left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \)。类似于情况1.1，我们可以构造一个 H-路径 \( {P}_{{l}_{t} + 1,k - 1} \) 在 \( Q\left\lbrack  {{l}_{t} + 1,k - 1}\right\rbrack   - F \) 中介于 \( {n}^{{l}_{t} + 1}\left( y\right) \) 和 \( {n}^{{l}_{t} + 1}\left( {y}^{ * }\right) \) 之间。因此，\( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup \) \( {P}_{{l}_{t} + 1,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \) \( \left\{  \left( {y,{y}^{ * }}\right) \right\} \) 形成了在 \( {Q}_{n}^{k} - F \) 中介于 \( s \) 和 \( t \) 之间的所需的 \( \mathrm{H} \) -路径（见图4）。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_5.jpg?x=193&y=183&w=580&h=348"/>

Fig. 4. The constructions in Case 1.2 of Theorem III.2.

图4。定理III.2中情况1.2的构造。

<!-- Media -->

Case 2: \( {l}_{s} \geq  1 \) .

情况2：\( {l}_{s} \geq  1 \)。

When \( {l}_{t} = k - 1 \) ,we can construct the required \( \mathrm{H} \) -path similar to Case 1. Then we discuss the case of \( 1 \leq  {l}_{s} \leq  {l}_{t} \leq  k - 2 \) .

当 \( {l}_{t} = k - 1 \) 时，我们可以类似于情况1构造所需的 \( \mathrm{H} \) -路径。然后我们讨论 \( 1 \leq  {l}_{s} \leq  {l}_{t} \leq  k - 2 \) 的情况。

Similar to Case 1,we can construct an H-path \( {P}_{{l}_{s},k - 1} \) in \( Q\left\lbrack  {{l}_{s},k - 1}\right\rbrack   - F \) between \( s \) and \( t \) . If the part of \( {P}_{{l}_{s},k - 1} \) in \( Q\left\lbrack  {l}_{s}\right\rbrack \) is constructed by the method similar to Case 1.1,by the proof of Case 1.1,let \( \left( {w,{w}^{ * }}\right) \) be the edge in \( Q\left\lbrack  {l}_{s}\right\rbrack \) satisfying \( \left\{  {\left( {w,{n}^{{l}_{s} + 1}\left( w\right) }\right) ,\left( {{w}^{ * },{n}^{{l}_{s} + 1}\left( {w}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) . By Claim 1,there exists one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) (or \( \;\left( {x,{x}^{ * }}\right)  \in  \left( {E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) }\right)  \cup  \left\{  \left( {w,{w}^{ * }}\right) \right\}  \; \) if \( \left( {w,{w}^{ * }}\right) \) exists) and \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similar to Case 1.1,we can construct an H-path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( x\right) \) and \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) . If \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) ,\; \) then \( \;{P}_{{l}_{s},k - 1} \cup  {P}_{0,{l}_{s} - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\}   - \left\{  \left( {x,{x}^{ * }}\right) \right\}  \; \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) .

与案例1类似，我们可以在 \( {P}_{{l}_{s},k - 1} \) 中构造一个 H-路径 \( Q\left\lbrack  {{l}_{s},k - 1}\right\rbrack   - F \) ，连接 \( s \) 和 \( t \) 。如果 \( {P}_{{l}_{s},k - 1} \) 在 \( Q\left\lbrack  {l}_{s}\right\rbrack \) 中的部分是通过类似于案例1.1的方法构造的，根据案例1.1的证明，设 \( \left( {w,{w}^{ * }}\right) \) 是满足 \( \left\{  {\left( {w,{n}^{{l}_{s} + 1}\left( w\right) }\right) ,\left( {{w}^{ * },{n}^{{l}_{s} + 1}\left( {w}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) 的 \( Q\left\lbrack  {l}_{s}\right\rbrack \) 中的边。根据命题1，存在一个边 \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) （如果存在 \( \left( {w,{w}^{ * }}\right) \) ，则为 \( \;\left( {x,{x}^{ * }}\right)  \in  \left( {E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) }\right)  \cup  \left\{  \left( {w,{w}^{ * }}\right) \right\}  \; \)）和 \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。类似于案例1.1，我们可以在 \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) 中构造一个 H-路径 \( {P}_{0,{l}_{s} - 1} \) ，连接 \( {n}^{{l}_{s} - 1}\left( x\right) \) 和 \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) 。如果 \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) ,\; \) ，那么 \( \;{P}_{{l}_{s},k - 1} \cup  {P}_{0,{l}_{s} - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\}   - \left\{  \left( {x,{x}^{ * }}\right) \right\}  \; \) 形成所需的 \( \mathrm{H} \) -路径，连接 \( s \) 和 \( t \) 在 \( {Q}_{n}^{k} - F \) 中。

Otherwise,we have \( \left( {x,{x}^{ * }}\right)  = \left( {w,{w}^{ * }}\right) \) and thus \( \left\{  {\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) . In this situation,the \( \mathrm{H} \) -path \( {P}_{{l}_{s},k - 1} \) must be constructed by the manner in Case 1.1. It implies that \( {l}_{s} = {l}_{t} \) and there exists an H-path \( {P}_{{l}_{s}} \) in \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) between \( s \) and \( t \) ,which passes through the edge \( \left( {x,{x}^{ * }}\right) \) . If \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  = \frac{{k}^{n - 1} - 3}{2} \) ,then

否则，我们有 \( \left( {x,{x}^{ * }}\right)  = \left( {w,{w}^{ * }}\right) \) ，因此 \( \left\{  {\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) 。在这种情况下，\( \mathrm{H} \) -路径 \( {P}_{{l}_{s},k - 1} \) 必须以案例1.1中的方式构造。这意味着 \( {l}_{s} = {l}_{t} \) ，并且存在一个 H-路径 \( {P}_{{l}_{s}} \) 在 \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) 中，连接 \( s \) 和 \( t \) ，该路径通过边 \( \left( {x,{x}^{ * }}\right) \) 。如果 \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  = \frac{{k}^{n - 1} - 3}{2} \) ，那么

\[\left| {F\left\lbrack  {{l}_{s} - 1,{l}_{s}}\right\rbrack  }\right|  \leq  {k}^{n - 1} - 2 - \left( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right| \right. \]

\[ + \left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right| )\]

\[ \leq  {k}^{n - 1} - 2 - 2 \times  \frac{{k}^{n - 1} - 3}{2} \leq  1.\]

Since \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s} - 1,{l}_{s}}\right\rbrack  }\right|  > 2 \) with \( n \geq  3 \) and odd \( k \geq \) 3,there exists one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq \) \( \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Otherwise, if \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \leq  \frac{{k}^{n - 1} - 5}{2} \) ,since \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \geq  2 \) , then there exists at least one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Thus,there exists at least one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) or \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) .

由于 \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s} - 1,{l}_{s}}\right\rbrack  }\right|  > 2 \) 有 \( n \geq  3 \) 和奇数 \( k \geq \) 3，存在一个边 \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) 使得 \( \left( {y,{y}^{ * }}\right)  \neq \) \( \left( {x,{x}^{ * }}\right) \) 和 \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。否则，如果 \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \leq  \frac{{k}^{n - 1} - 5}{2} \) ，由于 \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \geq  2 \) ，那么存在至少一个边 \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) 使得 \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) 和 \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。因此，存在至少一个边 \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) 使得 \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) 和 \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 或者 \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。

Note that the four edges \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right) \) , \( \left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . If \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {y}^{ * }\right. \) , \( \left. {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ,by Case 1.1,we can construct an H-path \( {P}_{{l}_{s} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} + 1}\left( y\right) \) and \( {n}^{{l}_{s} + 1}\left( {y}^{ * }\right) \) . Similar to Case 1.1,we can construct an H-path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( x\right) \) and \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) . Then \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) ,\left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) . If \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ,by Case 1.1,we can construct an \( \mathrm{H} \) -path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( y\right) \) and \( {n}^{{l}_{s} - 1}\left( {y}^{ * }\right) \) . Similar to Case 1.1,we construct an H-path \( {P}_{{l}_{s} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} + 1}\left( x\right) \) and \( {n}^{{l}_{s} + 1}\left( {x}^{ * }\right) \) . Then \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) ,\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 5).

注意四个边缘 \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right) \) ,\( \left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) 。如果 \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {y}^{ * }\right. \) ,\( \left. {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ，根据情况1.1，我们可以在 \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) 中构造一个从 \( {n}^{{l}_{s} + 1}\left( y\right) \) 到 \( {n}^{{l}_{s} + 1}\left( {y}^{ * }\right) \) 的 H-路径 \( {P}_{{l}_{s} + 1,k - 1} \) 。类似于情况1.1，我们可以在 \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) 中构造一个从 \( {n}^{{l}_{s} - 1}\left( x\right) \) 到 \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) 的 H-路径 \( {P}_{0,{l}_{s} - 1} \) 。那么 \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) ,\left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) 形成了所需的 \( \mathrm{H} \) -路径，从 \( s \) 到 \( t \) 在 \( {Q}_{n}^{k} - F \) 中。如果 \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ，根据情况1.1，我们可以在 \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) 中构造一个从 \( {n}^{{l}_{s} - 1}\left( y\right) \) 到 \( {n}^{{l}_{s} - 1}\left( {y}^{ * }\right) \) 的 \( \mathrm{H} \) -路径 \( {P}_{0,{l}_{s} - 1} \) 。类似于情况1.1，我们构造一个从 \( {n}^{{l}_{s} + 1}\left( x\right) \) 到 \( {n}^{{l}_{s} + 1}\left( {x}^{ * }\right) \) 的 H-路径 \( {P}_{{l}_{s} + 1,k - 1} \) 在 \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) 中。那么 \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) ,\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) 形成了所需的 \( \mathrm{H} \) -路径，从 \( s \) 到 \( t \) 在 \( {Q}_{n}^{k} - F \) 中（见图5）。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_5.jpg?x=959&y=182&w=589&h=354"/>

Fig. 5. The constructions in Case 2 of Theorem III.2.

图5。定理III.2中情况2的构造。

<!-- Media -->

## IV. FAULT-TOLERANT HAMILTONIAN PATH EMBEDDING ALGORITHM OF \( k \) -ARY \( n \) -CUBES

## IV. \( k \) -元 \( n \) -立方体的容错哈密顿路径嵌入算法

In this section,we present the fault-tolerant \( \mathrm{H} \) -path embedding algorithm for \( {Q}_{n}^{k} \) under the PEF model.

在本节中，我们提出了在PEF模型下的 \( \mathrm{H} \) -路径的容错嵌入算法，适用于 \( {Q}_{n}^{k} \)。

First,we design Algorithm 1 costing \( O\left( {k}^{2}\right) \) time to construct the \( \mathrm{H} \) -path in \( {Q}_{2}^{k} - F \) according to the theoretical proof in [42], where Procedure HP-rtFree is utilized to find the H-path in a fault-free \( {rt}\left( {p,q}\right) \) . In Algorithm 1,we let \( s = {u}_{a,b} \) and \( t = {u}_{c,d} \) be two arbitrary distinct nodes in \( {rt}\left( {p,q}\right) \) . In addition,without loss of generality,suppose that \( a \leq  c \) . Given an edge fault set \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) with \( \left| F\right|  \leq  1 \) ,by the symmetry of \( {Q}_{2}^{k} \) ,suppose that \( \left( {{u}_{0,0},{u}_{0,1}}\right) \) is the faulty edge if it exists.

首先，我们设计算法1，该算法花费 \( O\left( {k}^{2}\right) \) 时间来根据 [42] 中的理论证明在 \( {Q}_{2}^{k} - F \) 中构建 \( \mathrm{H} \) -路径，其中使用了过程 HP-rtFree 来在无故障的 \( {rt}\left( {p,q}\right) \) 中查找 H-路径。在算法1中，我们让 \( s = {u}_{a,b} \) 和 \( t = {u}_{c,d} \) 成为 \( {rt}\left( {p,q}\right) \) 中的两个任意不同节点。此外，不失一般性，假设 \( a \leq  c \) 。给定一个边缘故障集 \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) ，其中包含 \( \left| F\right|  \leq  1 \) ，由于 \( {Q}_{2}^{k} \) 的对称性，假设 \( \left( {{u}_{0,0},{u}_{0,1}}\right) \) 是存在故障的边缘。

Based on Algorithm 1, we design the Algorithm HP-PEF under the PEF model. Note that the Algorithm HP-PEF is essentially based on the theoretical basis in Section III, where Procedures HP-Round and HP-Direct correspond to the constructive approaches of Case 1.1 and Case 1.2 in Theorem III.2, respectively. In addition, the fault tolerance of Algorithm HP-PEF has been determined by Theorem III.2 (i.e., the three conditions shown in Theorem III.2).

基于 Algorithm 1，我们设计了在 PEF 模型下的 Algorithm HP-PEF。注意 Algorithm HP-PEF 实质上基于第 III 节中的理论依据，其中过程 HP-Round 和 HP-Direct 分别对应定理 III.2 中的案例 1.1 和案例 1.2 的构造方法。此外，Algorithm HP-PEF 的容错性已经由定理 III.2 确定（即定理 III.2 中显示的三个条件）。

Theorem IV.1. For \( n \geq  2 \) and odd \( k \geq  3 \) ,the algorithm HP-PEF can embed an H-path between arbitrary two nodes \( s \) and \( t \) into \( {Q}_{n}^{k} - F \) ,where \( F \) is a PEF set satisfying (1) \( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \) ; (2) \( {e}_{i} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ; (3) \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) .

定理 IV.1。对于 \( n \geq  2 \) 和奇数 \( k \geq  3 \) ，算法 HP-PEF 可以在任意两个节点 \( s \) 和 \( t \) 之间嵌入一个 H-路径到 \( {Q}_{n}^{k} - F \) 中，其中 \( F \) 是满足以下条件的 PEF 集合：（1）\( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \)；（2）对于每个 \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ，\( {e}_{i} \leq  {k}^{i} - 2 \)；（3）\( {e}_{0} = 0 \) 和 \( {e}_{1} \leq  1 \)。

Proof. We prove this theorem by induction on \( n \) . When \( n = 2 \) , HP-PEF calls Algorithm 1 to embed the required H-path into \( {Q}_{2}^{k} - F \) ,where \( F \) is a PEF set satisfying \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) . Assume this theorem holds for all \( {Q}_{m}^{k} \) ’s with \( m < n \) . Then we need to prove that this theorem holds for \( {Q}_{n}^{k} \) .

证明。我们通过对 \( n \) 进行归纳来证明这个定理。当 \( n = 2 \) 时，HP-PEF 调用算法 1 将所需的 H-路径嵌入到 \( {Q}_{2}^{k} - F \) 中，其中 \( F \) 是满足 \( {e}_{0} = 0 \) 和 \( {e}_{1} \leq  1 \) 的 PEF 集合。假设这个定理对于所有 \( {Q}_{m}^{k} \) 的 \( m < n \) 都成立。那么我们需要证明这个定理对于 \( {Q}_{n}^{k} \) 也成立。

<!-- Media -->

Algorithm 1: Embed an H-path \( P \) into \( {Q}_{2}^{k} - F \) .

算法 1：将 H-路径 \( P \) 嵌入到 \( {Q}_{2}^{k} - F \) 中。

---

Input: A \( k \) -ary 2-cube \( {Q}_{2}^{k} \) with odd \( k \geq  3 \) ,an edge fault

		set \( F \) with \( \left| F\right|  \leq  1 \) ,two distinct nodes \( s = {u}_{a,b} \)

		and \( t = {u}_{c,d} \) .

Output: An H-path \( P \) in \( {Q}_{2}^{k} - F \) between \( s \) and \( t \) .

\( P \leftarrow  \varnothing \) ;

if \( k = 3 \) then Implement an exhaustive search;

else

	if there exists no faulty edge then

		Call HP-rtFree \( \left( {{rt}\left( {0,k - 1}\right) ,s,t}\right) \) ;

	else

		if \( \{ a,c\}  = \{ 0,1\} \) then

				Construct two disjoint paths avoiding the

				edge \( \left( {{u}_{0,0},{u}_{0,1}}\right) ,{P}^{\prime } \) from \( s \) to \( {s}^{\prime } \) and \( {P}^{\prime \prime } \)

				from \( t \) to \( {t}^{\prime } \) in \( {rt}\left( {0,1}\right) \) such that \( V\left( {P}^{\prime }\right)  \cup \)

				\( V\left( {P}^{\prime \prime }\right)  = V\left( {{rt}\left( {0,1}\right) }\right) \) ;

				Let \( {s}^{\prime \prime } \) (resp. \( {t}^{\prime \prime } \) ) be the neighbor of \( {s}^{\prime } \) (resp.

				\( \left. {t}^{\prime }\right) \) in \( {Q}_{2}^{k} - {rt}\left( {0,1}\right) \) ;

				Call HP-rtFree \( \left( {{rt}\left( {2,k - 1}\right) ,{s}^{\prime \prime },{t}^{\prime \prime }}\right) \) ;

				\( P \leftarrow  {P}^{\prime } \cup  P \cup  {P}^{\prime \prime } \cup  \left\{  {\left( {{s}^{\prime },{s}^{\prime \prime }}\right) ,\left( {{t}^{\prime \prime },{t}^{\prime }}\right) }\right\}  ; \)

		else if \( \{ a,c\}  \subseteq  \{ 2,3,\ldots ,k - 1\} \) then

				\( C \leftarrow  {C}_{0}^{ + }\left( {{u}_{0,1},{u}_{1,1}}\right)  \cup  \left\{  \left( {{u}_{1,1},{u}_{0,1}}\right) \right\} \)

				Call HP-rtFree \( \left( {{rt}\left( {2,k - 1}\right) ,s,t}\right) \) ;

				Merge \( C \) into \( P \) through an edge of \( P \) on

				row 2,which isn’t \( \left( {{u}_{2,0},{u}_{2,1}}\right) \) ;

		else

				Let \( {P}^{\prime } \) be the \( \mathrm{H} \) -path avoiding the edge

				\( \left( {{u}_{0,0},{u}_{0,1}}\right) \) in \( {rt}\left( {0,1}\right) \) which begins at \( s \) and

				ends at \( {s}^{\prime } \) that isn’t adjacent to \( t \) ;

				Let \( {t}^{\prime } \) be the neighbor of \( {s}^{\prime } \) in \( {Q}_{2}^{k} - {rt}\left( {0,1}\right) \) ;

				Call HP-rtFree \( \left( {{rt}\left( {2,k - 1}\right) ,{t}^{\prime },t}\right) \) ;

				\( P \leftarrow  {P}^{\prime } \cup  P \cup  \left\{  \left( {{s}^{\prime },{t}^{\prime }}\right) \right\} \)

return \( P \) ;

---

<!-- Media -->

In Lines 4-5,HP-PEF divides \( {Q}_{n}^{k} \) into \( k \) disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) along the \( {i}^{\prime } \) -dimension such that \( \left| {F}_{{i}^{\prime }}\right|  = {e}_{n - 1} \leq  {k}^{n - 1} - 2 \) . Given an integer \( l \in  {\mathbb{Z}}_{k} \) ,let \( \left\{  {{e}_{n - 2}^{\prime },{e}_{n - 3}^{\prime },\ldots ,{e}_{0}^{\prime }}\right\}   = \left\{  {\left| {{F}_{j} \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) }\right|  \mid  j \in  {\mathbb{Z}}_{n} - \left\{  {i}^{\prime }\right\}  }\right\} \) such that \( {e}_{n - 2}^{\prime } \geq  {e}_{n - 3}^{\prime } \geq  \cdots  \geq  {e}_{0}^{\prime } \) . Then similar to the proof of Theorem III.2,for each \( l \in  {\mathbb{Z}}_{k},F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) \) is a PEF set satisfying (1) \( \left| {F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) }\right|  \leq  \frac{{k}^{n - 1} - {k}^{2}}{k - 1} - 2\left( {n - 1}\right)  + 5 \) ; (2) \( {e}_{i}^{\prime } \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n - 1} - {\mathbb{Z}}_{2};\left( 3\right) {e}_{0}^{\prime } = 0 \) and \( {e}_{1}^{\prime } \leq  1 \) . By the induction hypothesis, HP-PEF can embed an H-path between arbitrary two nodes into \( Q\left\lbrack  l\right\rbrack   - F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) \) . Next,in Line 6,let \( {l}^{\prime } \) denote the position of the subgraph which connects to the largest number of \( {i}^{\prime } \) -dimensional faulty edges. The subsequent constructive process prohibits the use of the edges between \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) and \( Q\left\lbrack  {{l}^{\prime } + 1}\right\rbrack \) (see the cross sign in Fig. 6). In Lines 7-10,if the node \( s \) (resp. \( t) \) is closer to \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) along the clockwise (see the green lines in Fig. 6) than \( t \) (resp. \( s \) ),HP-PEF takes \( s \) (resp. \( t \) ) as the source node \( a \) . Correspondingly,the node \( t \) (resp. \( s \) ) is deemed as the destination node \( b \) .

在第4-5行中，HP-PEF将 \( {Q}_{n}^{k} \) 分解为 \( k \) 沿着 \( {i}^{\prime } \) 维的 \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) 不相交子图，使得 \( \left| {F}_{{i}^{\prime }}\right|  = {e}_{n - 1} \leq  {k}^{n - 1} - 2 \) 。给定一个整数 \( l \in  {\mathbb{Z}}_{k} \) ，设 \( \left\{  {{e}_{n - 2}^{\prime },{e}_{n - 3}^{\prime },\ldots ,{e}_{0}^{\prime }}\right\}   = \left\{  {\left| {{F}_{j} \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) }\right|  \mid  j \in  {\mathbb{Z}}_{n} - \left\{  {i}^{\prime }\right\}  }\right\} \) 使得 \( {e}_{n - 2}^{\prime } \geq  {e}_{n - 3}^{\prime } \geq  \cdots  \geq  {e}_{0}^{\prime } \) 。然后类似于定理III.2的证明，对于每个 \( l \in  {\mathbb{Z}}_{k},F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) \) 是一个满足（1）\( \left| {F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) }\right|  \leq  \frac{{k}^{n - 1} - {k}^{2}}{k - 1} - 2\left( {n - 1}\right)  + 5 \)；（2）对于每个 \( i \in  {\mathbb{Z}}_{n - 1} - {\mathbb{Z}}_{2};\left( 3\right) {e}_{0}^{\prime } = 0 \) 和 \( {e}_{1}^{\prime } \leq  1 \) 的PEF集合。根据归纳假设，HP-PEF可以在 \( Q\left\lbrack  l\right\rbrack   - F \cap  E\left( {Q\left\lbrack  l\right\rbrack  }\right) \) 之间嵌入任意两个节点之间的H路径。接下来，在第6行中，设 \( {l}^{\prime } \) 表示连接到最大数量的 \( {i}^{\prime } \) 维故障边的子图的位置（见图6中的交叉符号）。在第7-10行中，如果节点 \( s \)（分别地 \( t) \) 沿着顺时针方向（见图6中的绿色线条）比 \( t \)（分别地 \( s \) ）更接近 \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) ，HP-PEF将 \( s \)（分别地 \( t \) ）作为源节点 \( a \) 。相应地，节点 \( t \)（分别地 \( s \) ）被视为目标节点 \( b \) 。

In Lines 11-20, HP-PEF embeds the required H-path into \( {Q}_{n}^{k} - F \) by discussing three situations similar to Theorem III.2. In the constructive process,we let \( l \) denote the destination subgraph,and \( d \) denote the constructive direction. More specifically, \( d = 1 \) (resp. \( d =  - 1 \) ) means the constructive process will proceed along the clockwise (resp. counterclockwise). Theorem III. 2 provides a detailed proof related to the correctness of Lines 11-20 when \( {i}^{\prime } = n - 1,{l}^{\prime } = k - 1,a = s \) ,and \( b = t \) . Though \( {l}^{\prime },a \) ,and \( b \) may change in each iteration,their relative positions remain unchanged, and the construction process is similar in different cases. Thus, the theorem holds.

在第11-20行中，HP-PEF通过讨论类似于定理III.2的三个情况，将所需的H路径嵌入到 \( {Q}_{n}^{k} - F \) 中。在构造过程中，我们让 \( l \) 表示目标子图，\( d \) 表示构造方向。更具体地说，\( d = 1 \)（分别\( d =  - 1 \)）意味着构造过程将沿顺时针（分别逆时针）方向进行。定理III.2提供了与 \( {i}^{\prime } = n - 1,{l}^{\prime } = k - 1,a = s \) 和 \( b = t \) 时第11-20行正确性相关的详细证明。尽管 \( {l}^{\prime },a \) 和 \( b \) 在每次迭代中可能发生变化，但它们的相对位置保持不变，且在不同情况下构造过程相似。因此，该定理成立。

<!-- Media -->

Procedure: \( \mathrm{{HP}} - \mathrm{{rtFree}}\left( {{rt}\left( {p,q}\right) ,s,t}\right) \) .

过程：\( \mathrm{{HP}} - \mathrm{{rtFree}}\left( {{rt}\left( {p,q}\right) ,s,t}\right) \)。

begin

开始

---

if \( q - p = 1 \) then

	if \( a = c \) and \( d - b\left( {\;\operatorname{mod}\;k}\right) \) is odd then

		\( P \leftarrow  {N}^{ + }\left( {s,{u}_{a,d - 1}}\right)  \cup  {C}_{b - 1}^{ + }\left( {{u}_{\bar{a},d},t}\right)  \cup \)

			\( \left\{  \left( {{u}_{a,d - 1},{u}_{\bar{a},d - 1},{u}_{\bar{a},d}}\right) \right\} \) ;

	else if \( a = c \) and \( d - b\left( {\;\operatorname{mod}\;k}\right) \) is even then

		\( P \leftarrow  {C}_{d - 1}^{ + }\left( {s,{u}_{\bar{a},b}}\right)  \cup  {N}^{ - }\left( {{u}_{\bar{a},b - 1},{u}_{\bar{a},d}}\right)  \cup \)

			\( \left\{  {\left( {{u}_{\bar{a},b},{u}_{\bar{a},b - 1}}\right) ,\left( {{u}_{\bar{a},d},t}\right) }\right\} \) ;

	else if \( a \neq  c \) and \( d - b\left( {\;\operatorname{mod}\;k}\right) \) is odd then

		\( P \leftarrow  {N}^{ - }\left( {s,{u}_{a,d}}\right)  \cup  {C}_{b + 1}^{ - }\left( {{u}_{a,d},t}\right) ; \)

	else if \( a \neq  c \) and \( d - b\left( {\;\operatorname{mod}\;k}\right) \) is even then

		\( P \leftarrow  {C}_{d + 1}^{ - }\left( {s,{u}_{\bar{a},b}}\right) \)

		if \( b \neq  d \) then \( P \leftarrow  P \cup  {N}^{ + }\left( {{u}_{\bar{a},b + 1},{u}_{\bar{a},d - 1}}\right) \)

			\( \cup  \left\{  {\left( {{u}_{\bar{a},b},{u}_{\bar{a},b + 1}}\right) ,\left( {{u}_{\bar{a},d - 1},{u}_{a,d - 1},{u}_{a,d},t}\right) }\right\}  ; \)

else

	if \( a = p \) and \( c = q \) then

		Call HP-rtFree \( \left( {{rt}\left( {p,q - 1}\right) ,s,{u}_{c - 1,d - 1}}\right) \) ;

		\( P \leftarrow  P \cup  \left\{  \left( {{u}_{c - 1,d - 1},{u}_{c,d - 1},{u}_{c,d - 2},\ldots ,}\right. \right. \)

			\( \left. \left. {{u}_{c,d + 1},t}\right) \right\} \) ;

	else

		Without loss of generality, suppose that

			\( s,t \in  V\left( {{rt}\left( {p,q - 1}\right) }\right) \) ;

		Call HP-rtFree \( \left( {{rt}\left( {p,q - 1}\right) ,s,t}\right) \) ;

		Choose an edge \( \left( {{u}_{q - 1,h},{u}_{q - 1,h + 1}}\right) \) of \( P \) on

			row \( q - 1 \) ;

		\( P \leftarrow  P \cup  \left\{  {({u}_{q - 1,h},{u}_{q,h},{u}_{q,h - 1},\ldots ,{u}_{q,h + 2},}\right. \)

			\( \left. \left. {{u}_{q,h + 1},{u}_{q - 1,h + 1}}\right) \right\}   - \left\{  \left( {{u}_{q - 1,h},{u}_{q - 1,h + 1}}\right) \right\}  ; \)

---

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_6.jpg?x=1024&y=1219&w=479&h=544"/>

Fig. 6. The relative positions of \( {l}^{\prime } \) and nodes \( a \) and \( b \) .

图6。\( {l}^{\prime } \)与节点 \( a \) 和 \( b \) 的相对位置。

Algorithm 2: Embed an H-path \( P \) into \( {Q}_{n}^{k} - F \) where \( F \) is a PEF set (HP-PEF).

算法2：将H路径 \( P \) 嵌入到 \( {Q}_{n}^{k} - F \) 中，其中 \( F \) 是PEF集合（HP-PEF）。

---

Input: A \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) with \( n \geq  2 \) and odd \( k \geq  3 \) ,

			two distinct nodes \( s \) and \( t \) ,a PEF set \( F \) satisfying

			(1) \( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) ; (2) \( {e}_{i} \leq  {k}^{i} - 2 \) for each

			\( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ; (3) \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) .

Output: An H-path \( P \) in \( {Q}_{n}^{k} - F \) between \( s \) and \( t \) .

\( P \leftarrow  \varnothing \) ;

if \( n = 2 \) then \( P \leftarrow  \operatorname{Algorithm}1\left( {{Q}_{2}^{k},F,s,t}\right) \) ;

else

		\( {i}^{\prime } \leftarrow  \arg \mathop{\max }\limits_{{i \in  {\mathbb{Z}}_{n}}}\left\{  \left| {F}_{i}\right| \right\} \)

		Divide \( {Q}_{n}^{k} \) into \( k \) disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ,

		\( \ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) along the \( {i}^{\prime } \) -dimension;

		\( {l}^{\prime } \leftarrow  \arg \mathop{\max }\limits_{{l \in  {\mathbb{Z}}_{k}}}\left\{  \left| {{F}_{{i}^{\prime }}\left\lbrack  {l,\left( {l + 1}\right) {\;\operatorname{mod}\;k}}\right\rbrack  }\right| \right\}  ; \)

		// Take \( a \) (resp. \( b \) ) as the source node (resp.

		destination node).

		if \( {l}_{s} - {l}^{\prime }\left( {\;\operatorname{mod}\;k}\right)  \leq  {l}_{t} - {l}^{\prime }\left( {\;\operatorname{mod}\;k}\right) \) then

				\( a \leftarrow  s;b \leftarrow  t; \)

		else \( a \leftarrow  t;b \leftarrow  s \) ;

		if \( {l}_{a} = {l}^{\prime } + 1\left( {\;\operatorname{mod}\;k}\right) \) then // Situation 1

				if \( {l}_{a} = {l}_{b} \) then Call HP-Round \( \left( {{l}^{\prime },1,a,b}\right) \) ;

				else Call HP-Direct \( \left( {{l}^{\prime },1,a,b}\right) \) ;

		else if \( {l}_{a} = {l}^{\prime } \) then // Situation 2

			Call HP-Direct \( \left( {{l}^{\prime } + 1, - 1,a,b}\right) \) ;

		else // Situation 3

				Call HP-Direct \( \left( {{l}^{\prime },1,a,b}\right) \) ;

				Select an edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( P\right)  \cap  E\left( {Q\left\lbrack  {l}_{a}\right\rbrack  }\right) \) such

				that \( \left( {x,{n}^{{l}_{a} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{a} - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{{i}^{\prime }} \) ;

				Call HP-Round \( \left( {{l}^{\prime } + 1, - 1,{n}^{{l}_{a} - 1}\left( x\right) ,{n}^{{l}_{a} - 1}\left( {x}^{ * }\right) }\right) \) ;

				\( P \leftarrow  P \cup  \left\{  {\left( {x,{n}^{{l}_{a} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{a} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\}   - \)

				\( \left\{  \left( {x,{x}^{ * }}\right) \right\} \) ;

return \( P \) ;

---

Procedure: HP-Round(l,d,s,t).

过程：HP-Round(l,d,s,t)。

---

begin

		\( u \leftarrow  s;v \leftarrow  t \) ;

		// Take \( l \) as the destination subgraph,and \( d \) as the

		constructive direction.

		if \( l \neq  {l}_{s} \) then

			for 1 to \( d \times  \left( {l - {l}_{s}}\right) {\;\operatorname{mod}\;k} \) do

					\( {P}^{\prime } \leftarrow \) Algorithm \( 2\left( {Q\left\lbrack  {l}_{u}\right\rbrack  ,F \cap  E\left( {Q\left\lbrack  {l}_{u}\right\rbrack  }\right) ,u,v}\right) \) ;

					Select an edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}^{\prime }\right) \) such that

					\( \left( {x,{n}^{{l}_{u} + d}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{u} + d}\left( {x}^{ * }\right) }\right)  \notin  {F}_{{i}^{\prime }}; \)

					\( P \leftarrow  P \cup  {P}^{\prime } \cup  \left\{  {\left( {x,{n}^{{l}_{u} + d}\left( x\right) }\right) ,\left( {{n}^{{l}_{u} + d}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\} \)

					\( - \{ \left( {x,{x}^{ * }}\right) \} \) ;

					\( u \leftarrow  {n}^{{l}_{u} + d}\left( x\right) ;v \leftarrow  {n}^{{l}_{u} + d}\left( {x}^{ * }\right) ; \)

		\( {P}^{\prime } \leftarrow \) Algorithm \( 2\left( {Q\left\lbrack  {l}_{u}\right\rbrack  ,F \cap  E\left( {Q\left\lbrack  {l}_{u}\right\rbrack  }\right) ,u,v}\right) \) ;

		\( P \leftarrow  P \cup  {P}^{\prime }; \)

---

<!-- Media -->

The time complexity of HP-PEF is showed as follows.

HP-PEF的时间复杂度如下所示。

Theorem IV.2. The time complexity of HP-PEF is \( O\left( N\right) \) , where \( N = {k}^{n} \) .

定理IV.2。HP-PEF的时间复杂度是 \( O\left( N\right) \) ，其中 \( N = {k}^{n} \) 。

Proof. When \( n = 2 \) ,HP-PEF calls Algorithm 1,which costs \( O\left( {k}^{2}\right) \) time. Next,we discuss the case of \( n \geq  3 \) in the following.

证明。当 \( n = 2 \) 时，HP-PEF调用算法1，其花费 \( O\left( {k}^{2}\right) \) 时间。接下来，我们讨论 \( n \geq  3 \) 的情况。

Since HP-PEF calls HP-Round and HP-Direct frequently, we first analyse the time complexity of HP-Round and HP-Direct.

由于HP-PEF频繁调用HP-Round和HP-Direct，我们首先分析HP-Round和HP-Direct的时间复杂度。

<!-- Media -->

Procedure: \( \operatorname{HP-Direct}\left( {l,d,s,t}\right) \) .

过程：\( \operatorname{HP-Direct}\left( {l,d,s,t}\right) \)。

---

begin

		\( u \leftarrow  s \) ;

		// Take \( l \) as the destination subgraph,and \( d \) as the

		constructive direction.

	if \( {l}_{t} \neq  {l}_{s} \) then

			for 1 to \( d \times  \left( {{l}_{t} - {l}_{s}}\right) {\;\operatorname{mod}\;k} \) do

				Select a node \( x \in  V\left( {Q\left\lbrack  {l}_{u}\right\rbrack  }\right) \) with \( x \neq  u \) ,

					\( {n}^{{l}_{u} + d}\left( x\right)  \neq  t \) ,and \( \left( {x,{n}^{{l}_{u} + d}\left( x\right) }\right)  \notin  {F}_{{i}^{\prime }} \) ;

				\( {P}^{\prime } \leftarrow \) Algorithm \( 2\left( {Q\left\lbrack  {l}_{u}\right\rbrack  ,F \cap  E\left( {Q\left\lbrack  {l}_{u}\right\rbrack  }\right) ,u,x}\right) \) ;

				\( P \leftarrow  P \cup  {P}^{\prime } \cup  \left\{  \left( {x,{n}^{{l}_{u} + d}\left( x\right) }\right) \right\}  ; \)

				\( u \leftarrow  {n}^{{l}_{u} + d}\left( x\right) ; \)

		\( {P}^{\prime } \leftarrow \) Algorithm \( 2\left( {Q\left\lbrack  {l}_{u}\right\rbrack  ,F \cap  E\left( {Q\left\lbrack  {l}_{u}\right\rbrack  }\right) ,u,t}\right) \) ;

		\( P \leftarrow  P \cup  {P}^{\prime }; \)

		if \( {l}_{t} \neq  l \) then

			Select an edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( P\right)  \cap  E\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) such

			that \( \left( {x,{n}^{{l}_{t} + d}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{t} + d}\left( {x}^{ * }\right) }\right)  \notin  {F}_{{i}^{\prime }} \) ;

			Call HP-Round \( \left( {l,d,{n}^{{l}_{t} + d}\left( x\right) ,{n}^{{l}_{t} + d}\left( {x}^{ * }\right) }\right) \) ;

			\( P \leftarrow  P \cup  \left\{  {\left( {x,{n}^{{l}_{t} + d}\left( x\right) }\right) ,\left( {{n}^{{l}_{t} + d}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\}   - \)

			\( \left\{  \left( {x,{x}^{ * }}\right) \right\} \) ;

---

<!-- Media -->

The main time costs of HP-Round and HP-Direct are on the for-loops. The for-loops execute at most \( k \) iterations. Each iteration calls HP-PEF and selects a required edge or node. Calling HP-PEF to construct the H-path in every subgraph of \( {Q}_{n}^{k} \) requires \( {k}^{n - 3}O\left( {k}^{2}\right)  = O\left( {k}^{n - 1}\right) \) . The H-path obtained contains \( \frac{{k}^{n - 1} - 1}{2} \) mutually disjoint edges,and every subgraph of \( {Q}_{n}^{k} \) contains \( {k}^{n - 1} \) nodes. Thus,selecting a required edge or node costs \( O\left( {k}^{n - 1}\right) \) time. Therefore, the time complexity of both HP-Round and HP-Direct is \( k\left( {O\left( {k}^{n - 1}\right)  + O\left( {k}^{n - 1}\right) }\right)  = O\left( {k}^{n}\right) \) .

HP-Round 和 HP-Direct 的主要时间成本在于 for 循环。这些 for 循环最多执行 \( k \) 次迭代。每次迭代调用 HP-PEF 并选择所需的边或节点。在 \( {Q}_{n}^{k} \) 的每个子图中调用 HP-PEF 构建路径 H 需要 \( {k}^{n - 3}O\left( {k}^{2}\right)  = O\left( {k}^{n - 1}\right) \) 时间。得到的 H 路径包含 \( \frac{{k}^{n - 1} - 1}{2} \) 个互不相交的边，且 \( {Q}_{n}^{k} \) 的每个子图包含 \( {k}^{n - 1} \) 个节点。因此，选择所需的边或节点需要 \( O\left( {k}^{n - 1}\right) \) 时间。因此，HP-Round 和 HP-Direct 的时间复杂度是 \( k\left( {O\left( {k}^{n - 1}\right)  + O\left( {k}^{n - 1}\right) }\right)  = O\left( {k}^{n}\right) \)。

Line 4 costs \( O\left( n\right) \) time. Line 5 classifies all \( {k}^{n} \) nodes of \( {Q}_{n}^{k} \) ,costing \( O\left( {k}^{n}\right) \) time. Line 6 costs \( O\left( k\right) \) time. Lines 7-10 cost \( O\left( 1\right) \) time. Lines 11-13 cost \( O\left( {k}^{n}\right) \) time. Lines 14-15 cost \( O\left( {k}^{n}\right) \) time. Line 17 costs \( O\left( {k}^{n}\right) \) time. Line 18 costs \( O\left( {k}^{n - 1}\right) \) time since there exist \( \frac{{k}^{n - 1} - 1}{2} \) mutually disjoint edges of \( P \) in each subgraph. Line 19 costs \( O\left( {k}^{n}\right) \) time. Line 20 costs \( O\left( 1\right) \) time.

第 4 行需要 \( O\left( n\right) \) 时间。第 5 行对 \( {Q}_{n}^{k} \) 的所有 \( {k}^{n} \) 节点进行分类，需要 \( O\left( {k}^{n}\right) \) 时间。第 6 行需要 \( O\left( k\right) \) 时间。第 7-10 行需要 \( O\left( 1\right) \) 时间。第 11-13 行需要 \( O\left( {k}^{n}\right) \) 时间。第 14-15 行需要 \( O\left( {k}^{n}\right) \) 时间。第 17 行需要 \( O\left( {k}^{n}\right) \) 时间。第 18 行需要 \( O\left( {k}^{n - 1}\right) \) 时间，因为在每个子图中存在 \( \frac{{k}^{n - 1} - 1}{2} \) 个互不相交的 \( P \) 边。第 19 行需要 \( O\left( {k}^{n}\right) \) 时间。第 20 行需要 \( O\left( 1\right) \) 时间。

Therefore,HP-PEF needs \( O\left( N\right) \) time.

因此，HP-PEF 需要 \( O\left( N\right) \) 时间。

In the following, we give a detailed example to explain how Algorithm HP-PEF embeds the H-path into \( {Q}_{n}^{k} \) under the PEF model. In the example,we set \( n = k = 3,s = {001} \) , and \( t = {210} \) . Moreover,let \( {F}_{2} = \{ \left( {{100},{200}}\right) \} ,\;{F}_{1} = \) \( \{ \left( {{001},{011}}\right) ,\left( {{010},{020}}\right) ,\left( {{210},{220}}\right) ,\left( {{200},{220}}\right) ,\left( {{201},{221}}\right) , \) \( \left( {{102},{122}}\right) ,\left( {{002},{022}}\right) \} ,\;{F}_{0} = \varnothing \) . Obviously,we have \( {e}_{2} = \left| {F}_{1}\right|  = 7,\;{e}_{1} = \left| {F}_{2}\right|  = 1 \) ,and \( {e}_{0} = \left| {F}_{0}\right|  = 0 \) . Since \( \left| {F}_{1}\right|  > \left| {F}_{0}\right| ,\left| {F}_{2}\right| \) ,the algorithm HP-PEF divides \( {Q}_{3}^{3} \) into 3 disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ,and \( Q\left\lbrack  2\right\rbrack \) along the 1-dimension. Since \( \left| {{F}_{1}\left\lbrack  {2,0}\right\rbrack  }\right|  > \left| {{F}_{1}\left\lbrack  {0,1}\right\rbrack  }\right| ,\left| {{F}_{1}\left\lbrack  {1,2}\right\rbrack  }\right| \) ,we have \( {l}^{\prime } = 2 \) ,which implies that the subsequent constructive process doesn't involve the edges between \( Q\left\lbrack  2\right\rbrack \) and \( Q\left\lbrack  0\right\rbrack \) . Next,the algorithm HP-PEF will take the node \( s \) as the source node \( a \) since \( s \) is closer to \( Q\left\lbrack  2\right\rbrack \) along the clockwise than \( t \) . Correspondingly,the node \( t \) is deemed as the destination node \( b \) . Since \( {l}_{a} = 0 = {l}^{\prime } + 1\left( {\;\operatorname{mod}\;3}\right) \) and \( {l}_{a} = 0 \neq  {l}_{b} = 1 \) ,the algorithm HP-PEF calls the procedure HP-Direct to construct the H-path along the clockwise (i.e., \( d = 1 \) ) by the approaches of Case 1.2 in Theorem III.2,where the H-path in \( Q\left\lbrack  l\right\rbrack \) (isomorphic to \( {Q}_{2}^{3} \) ) with \( l \in  {\mathbb{Z}}_{3} \) is constructed by Algorithm 1. Fig. 7 shows the H-path constructed by Algorithm HP-PEF in \( {Q}_{3}^{3} \) ,where the dashed lines represent the faulty edges.

在以下内容中，我们给出了一个详细的示例，以解释算法 HP-PEF 如何在 PEF 模型下将 H 路径嵌入 \( {Q}_{n}^{k} \) 中。在示例中，我们设定 \( n = k = 3,s = {001} \) ，并且 \( t = {210} \) 。此外，令 \( {F}_{2} = \{ \left( {{100},{200}}\right) \} ,\;{F}_{1} = \) \( \{ \left( {{001},{011}}\right) ,\left( {{010},{020}}\right) ,\left( {{210},{220}}\right) ,\left( {{200},{220}}\right) ,\left( {{201},{221}}\right) , \) \( \left( {{102},{122}}\right) ,\left( {{002},{022}}\right) \} ,\;{F}_{0} = \varnothing \) 。显然，我们有 \( {e}_{2} = \left| {F}_{1}\right|  = 7,\;{e}_{1} = \left| {F}_{2}\right|  = 1 \) ，并且 \( {e}_{0} = \left| {F}_{0}\right|  = 0 \) 。由于 \( \left| {F}_{1}\right|  > \left| {F}_{0}\right| ,\left| {F}_{2}\right| \) ，算法 HP-PEF 将 \( {Q}_{3}^{3} \) 划分为 3 个不相交的子图 \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ，并沿 1 维方向 \( Q\left\lbrack  2\right\rbrack \) 和 \( \left| {{F}_{1}\left\lbrack  {2,0}\right\rbrack  }\right|  > \left| {{F}_{1}\left\lbrack  {0,1}\right\rbrack  }\right| ,\left| {{F}_{1}\left\lbrack  {1,2}\right\rbrack  }\right| \) 。由于 \( {l}^{\prime } = 2 \) ，我们得到 \( Q\left\lbrack  0\right\rbrack \) ，这意味着后续构造过程不涉及 \( Q\left\lbrack  2\right\rbrack \) 和 \( s \) 之间的边。接下来，算法 HP-PEF 将选择节点 \( a \) 作为源节点 \( t \) ，因为 \( a \) 沿顺时针方向比 \( b \) 更接近 \( Q\left\lbrack  2\right\rbrack \) 。相应地，节点 \( b \) 被视为目标节点 \( {l}_{a} = 0 = {l}^{\prime } + 1\left( {\;\operatorname{mod}\;3}\right) \) 。由于 \( {l}_{a} = 0 \neq  {l}_{b} = 1 \) 和 \( d = 1 \) ，算法 HP-PEF 调用过程 HP-Direct 沿顺时针方向（即 \( Q\left\lbrack  l\right\rbrack \) ）构造 H 路径，方法遵循定理 III.2 中的案例 1.2。其中，\( {Q}_{2}^{3} \) 中的 H 路径（同构于 \( l \in  {\mathbb{Z}}_{3} \) ）通过算法 1 构建。图 7 展示了算法 HP-PEF 在 \( {Q}_{3}^{3} \) 中构造的 H 路径，其中虚线表示故障边。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_8.jpg?x=214&y=186&w=553&h=695"/>

Fig. 7. The H-path constructed by Algorithm HP-PEF in \( {Q}_{3}^{3} \) ,where the solid lines represent the fault-free edges and the dashed lines represent the faulty edges.

图 7。算法 HP-PEF 在 \( {Q}_{3}^{3} \) 中构造的 H 路径，其中实线表示无故障边，虚线表示故障边。

<!-- Media -->

## V. Performance Analysis

## V. 性能分析

In this section, we implement the algorithm HP-PEF by using Python programs and evaluate the performance of HP-PEF. We carry out the simulation by using a \( {2.80}\mathrm{{GHz}} \) Intel \( {}^{\circledR }{\mathrm{{Core}}}^{\mathrm{{TM}}} \) i9-10900 CPU and 127 GB RAM under the Linux operating system.

在本节中，我们通过使用Python程序实现了算法HP-PEF，并评估了HP-PEF的性能。我们在Linux操作系统下，使用\( {2.80}\mathrm{{GHz}} \)英特尔\( {}^{\circledR }{\mathrm{{Core}}}^{\mathrm{{TM}}} \) i9-10900 CPU和127 GB RAM进行了模拟。

## A. Average Path Length

## A. 平均路径长度

Along the H-path constructed by HP-PEF, we aim to compute the path length which is the number of edges between the source and destination nodes. We are interested in computing the path length since it's beneficial to evaluate the performance of HP-PEF for future applications to NoCs. Moreover, based on the following concerns,we choose the adjacent node pair in \( {Q}_{n}^{k} \) as the source and the destination to compute the path length:

沿着HP-PEF构建的H路径，我们旨在计算路径长度，即源节点和目标节点之间的边数。我们感兴趣于计算路径长度，因为这有助于评估HP-PEF在未来应用于NoCs的性能。此外，基于以下考虑，我们选择\( {Q}_{n}^{k} \)中的相邻节点对作为源节点和目标节点来计算路径长度：

1) When there exists no faulty edge in networks, the adjacent node pair possesses the shortest path length 1.

1) 当网络中不存在故障边时，相邻节点对具有最短路径长度1。

2) The algorithm HP-PEF can address large-scale edge faults. When the edge fault occurs, the path length of adjacent node pairs is naturally more sensitive to edge fault than other kinds of node pairs.

2) 算法HP-PEF能够处理大规模的边故障。当边故障发生时，相邻节点对的路径长度自然比其他类型的节点对对边故障更敏感。

3) Our method focuses on the distribution pattern of edge faults in each dimension. The path length of adjacent node pairs may be followed by the distribution pattern of edge faults. It's interesting to explore the relationship between the path length of adjacent node pairs and the number of edge faults in each dimension.

3) 我们的方法关注每个维度中边故障的分布模式。相邻节点对的路径长度可能会遵循边故障的分布模式。探索相邻节点对的路径长度与每个维度中边故障数量之间的关系是很有趣的。

<!-- Media -->

TABLE I

EXPERIMENTAL SETTINGS

实验设置

<table><tr><td/><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>Adjacent node pairs</td><td>\( {27} \times  3 \)</td><td>\( {125} \times  3 \)</td><td>\( {343} \times  3 \)</td><td>\( {729} \times  3 \)</td><td>\( {81} \times  4 \)</td><td>\( {243} \times \)</td><td>\( {729} \times  6 \)</td></tr><tr><td>Faulty edges</td><td>8</td><td>24</td><td>48</td><td>80</td><td>33</td><td>112</td><td>353</td></tr><tr><td>H-paths</td><td>351</td><td>7750</td><td>58653</td><td>265356</td><td>3240</td><td>29403</td><td>265356</td></tr></table>

<table><tbody><tr><td></td><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>相邻节点对</td><td>\( {27} \times  3 \)</td><td>\( {125} \times  3 \)</td><td>\( {343} \times  3 \)</td><td>\( {729} \times  3 \)</td><td>\( {81} \times  4 \)</td><td>\( {243} \times \)</td><td>\( {729} \times  6 \)</td></tr><tr><td>故障边</td><td>8</td><td>24</td><td>48</td><td>80</td><td>33</td><td>112</td><td>353</td></tr><tr><td>H路径</td><td>351</td><td>7750</td><td>58653</td><td>265356</td><td>3240</td><td>29403</td><td>265356</td></tr></tbody></table>

<!-- Media -->

One edge corresponds to exactly one adjacent node pair. Thus, we define the dimension of an adjacent node pair as the dimension of the edge connecting it. By the definition of \( {Q}_{n}^{k} \) , there exist \( {k}^{n}i \) -dimensional edges,which implies that there exist \( {k}^{n}i \) -dimensional adjacent node pairs for each \( i \in  {\mathbb{Z}}_{n} \) . Then \( {Q}_{n}^{k} \) has \( n{k}^{n} \) adjacent node pairs in total. As for the faulty edges,we directly set \( \left| F\right|  = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) to prevent the occurrence of \( \left| {F}_{2}\right|  = \cdots  = \left| {F}_{n - 1}\right| \) in the experimental process. Moreover, to make comparisons more obvious, we assume \( \left| {F}_{n - 1}\right|  \geq  \left| {F}_{n - 2}\right|  \geq  \cdots  \geq  \left| {F}_{0}\right| \) . Then we set \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) with \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ,and \( \left| {F}_{0}\right|  = 0 \) . Given a node pair(s,t) and a PEF set, the algorithm HP-PEF can generate an H-path between \( s \) and \( t \) . There exist \( \left( \begin{matrix} {k}^{n} \\  2 \end{matrix}\right)  = \frac{{k}^{2n} - {k}^{n}}{2} \) different node pairs (s,t). Thus,given a PEF set,the algorithm HP-PEF can generate \( \frac{{k}^{2n} - {k}^{n}}{2} \) different H-paths. Our experimental settings are shown in Table I.

一个边对应于恰好一对相邻节点。因此，我们定义相邻节点对的维度为连接它的边的维度。根据 \( {Q}_{n}^{k} \) 的定义，存在 \( {k}^{n}i \) 维的边，这意味着对于每个 \( i \in  {\mathbb{Z}}_{n} \) 存在 \( {k}^{n}i \) 维的相邻节点对。那么 \( {Q}_{n}^{k} \) 总共有 \( n{k}^{n} \) 对相邻节点。至于故障边，我们直接设置 \( \left| F\right|  = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) 以防止实验过程中出现 \( \left| {F}_{2}\right|  = \cdots  = \left| {F}_{n - 1}\right| \) 。此外，为了使比较更加明显，我们假设 \( \left| {F}_{n - 1}\right|  \geq  \left| {F}_{n - 2}\right|  \geq  \cdots  \geq  \left| {F}_{0}\right| \) 。然后我们设置 \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) 为 \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ，并且 \( \left| {F}_{0}\right|  = 0 \) 。给定一个节点对 (s,t) 和一个PEF集合，算法HP-PEF可以在 \( s \) 和 \( t \) 之间生成一条H路径。存在 \( \left( \begin{matrix} {k}^{n} \\  2 \end{matrix}\right)  = \frac{{k}^{2n} - {k}^{n}}{2} \) 对不同的节点对 (s,t)。因此，给定一个PEF集合，算法HP-PEF可以生成 \( \frac{{k}^{2n} - {k}^{n}}{2} \) 条不同的H路径。我们的实验设置如表I所示。

In the simulation,we first randomly generate a PEF set \( F \) with \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) for \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ,and \( \left| {F}_{0}\right|  = 0 \) . Next, under the edge fault set \( F \) ,we utilize the algorithm HP-PEF to generate all possible \( \mathrm{H} \) -paths,that is, \( \frac{{k}^{2n} - {k}^{n}}{2} \) different \( \mathrm{H} \) -paths. And then, according to the dimension of adjacent node pairs, we compute the average path length (APL for short) of them in all \( \mathrm{H} \) -paths. More specifically,we denote the APL of each adjacent node pair in all \( \mathrm{H} \) -paths as \( p{l}_{i}^{j} \) ,where \( i \in  {\mathbb{Z}}_{n} \) is the dimension of the adjacent node pair and \( j \in  {\mathbb{Z}}_{{k}^{n}} \) is the unique identification of the adjacent node pair in the \( i \) -dimension. Then we compute the \( {\mathrm{{APL}}}_{i} \) for each dimension \( i \in  {\mathbb{Z}}_{n} \) as follows:

在模拟中，我们首先随机生成一个PEF集合 \( F \) ，包含 \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) 用于 \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ，以及 \( \left| {F}_{0}\right|  = 0 \) 。接着，在边缘故障集 \( F \) 下，我们使用算法HP-PEF生成所有可能的 \( \mathrm{H} \) -路径，即 \( \frac{{k}^{2n} - {k}^{n}}{2} \) 个不同的 \( \mathrm{H} \) -路径。然后，根据相邻节点对的维度，我们计算它们在所有 \( \mathrm{H} \) -路径中的平均路径长度（简称APL）。更具体地说，我们表示每个相邻节点对在所有 \( \mathrm{H} \) -路径中的APL为 \( p{l}_{i}^{j} \) ，其中 \( i \in  {\mathbb{Z}}_{n} \) 是相邻节点对的维度，\( j \in  {\mathbb{Z}}_{{k}^{n}} \) 是在 \( i \) 维度中相邻节点对的唯一标识。然后我们按以下方式计算每个维度 \( i \in  {\mathbb{Z}}_{n} \) 的 \( {\mathrm{{APL}}}_{i} \)：

\[{\mathrm{{APL}}}_{i} = \frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}p{l}_{i}^{j}}{{k}^{n}}.\]

Simultaneously, we compute the standard deviation (SD for short) to evaluate the quantity of difference in path length as follows:

同时，我们计算标准差（简称SD），以评估路径长度差异的数量，如下所示：

\[{\mathrm{{SD}}}_{i} = \sqrt{\frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}{\left( p{l}_{i}^{j} - {\mathrm{{APL}}}_{i}\right) }^{2}}{{k}^{n}}}.\]

Fig. 8 shows the simulation results about the \( {\mathrm{{APL}}}_{i} \) of \( {Q}_{n}^{k} \) with different \( k \) and \( n \) . We observe that \( {\mathrm{{APL}}}_{i} \) is positively related to \( \left| {F}_{i}\right| \) . For example,the values of \( \left| {F}_{0}\right| \) and \( \left| {F}_{1}\right| \) differ by only 1,and \( {\mathrm{{APL}}}_{1} \) is slightly higher than \( {\mathrm{{APL}}}_{0} \) . The value of \( \left| {F}_{i}\right| \) with \( i \geq  2 \) is a power function with base \( k \) ,and \( {\mathrm{{APL}}}_{i} \) is much higher than \( {\mathrm{{APL}}}_{0} \) and \( {\mathrm{{APL}}}_{1} \) . In \( {Q}_{n}^{3} \) ,the distance value between \( {\mathrm{{APL}}}_{i + 1} \) and \( {\mathrm{{APL}}}_{i} \) exhibits a sharper growth trend with larger \( i \) . Moreover,with increasing \( n \) and \( k \) ,we observe from the results that the growth rate of \( {\mathrm{{APL}}}_{i} \) is more sensitive to \( k \) than \( n \) . For example,in \( {Q}_{3}^{k} \) with increasing \( k \) from 3 to 9,the \( {\mathrm{{APL}}}_{2} \) grows rapidly from 11.1 to 154.3. However,in \( {Q}_{n}^{3} \) with increasing \( n \) from 3 to 6,the \( {\mathrm{{APL}}}_{2} \) grows slowly from 11.1 to 17.7. Although \( k \) increment by 2 and \( n \) increment by 1 on the \( x \) -axis,the growth rate of \( {\mathrm{{APL}}}_{i} \) in \( {Q}_{n}^{3} \) can still indicate the modest influence of \( n \) on \( {\mathrm{{APL}}}_{i} \) . This phenomenon may be caused by the fact that the value of \( \left| {F}_{i}\right| \) is only influenced by \( k \) ,independent of \( n \) . Fig. 9 shows the \( {\mathrm{{SD}}}_{i} \) of \( {Q}_{n}^{k} \) with different \( k \) and \( n \) . The growth trends for \( {\mathrm{{SD}}}_{i} \) and \( {\mathrm{{APL}}}_{i} \) are roughly the same. However,the parameter \( n \) can significantly affect the growth rate of \( {\mathrm{{SD}}}_{i} \) ,which is obviously different from that of \( {\mathrm{{APL}}}_{i} \) . It implies that the larger \( n \) and \( k \) , the more dispersed the values of path length.

图8显示了关于 \( {\mathrm{{APL}}}_{i} \) 的模拟结果，涉及不同 \( k \) 和 \( n \) 的 \( {Q}_{n}^{k} \) 。我们观察到 \( {\mathrm{{APL}}}_{i} \) 与 \( \left| {F}_{i}\right| \) 正相关。例如，\( \left| {F}_{0}\right| \) 和 \( \left| {F}_{1}\right| \) 的值仅相差1，而 \( {\mathrm{{APL}}}_{1} \) 略高于 \( {\mathrm{{APL}}}_{0} \) 。带有 \( i \geq  2 \) 的 \( \left| {F}_{i}\right| \) 的值是一个以 \( k \) 为底的幂函数，且 \( {\mathrm{{APL}}}_{i} \) 远高于 \( {\mathrm{{APL}}}_{0} \) 和 \( {\mathrm{{APL}}}_{1} \) 。在 \( {Q}_{n}^{3} \) 中，\( {\mathrm{{APL}}}_{i + 1} \) 与 \( {\mathrm{{APL}}}_{i} \) 之间的距离值在较大的 \( i \) 下呈现出更尖锐的增长趋势。此外，随着 \( n \) 和 \( k \) 的增加，我们从结果中观察到 \( {\mathrm{{APL}}}_{i} \) 的增长率对 \( k \) 比对 \( n \) 更敏感。例如，在 \( {Q}_{3}^{k} \) 中，随着 \( k \) 从3增加到9，\( {\mathrm{{APL}}}_{2} \) 从11.1迅速增长到154.3。然而，在 \( {Q}_{n}^{3} \) 中，随着 \( n \) 从3增加到6，\( {\mathrm{{APL}}}_{2} \) 仅从11.1缓慢增长到17.7。尽管在 \( x \) -轴上 \( k \) 增加2而 \( n \) 增加1，但在 \( {Q}_{n}^{3} \) 中 \( {\mathrm{{APL}}}_{i} \) 的增长率仍然可以表明 \( n \) 对 \( {\mathrm{{APL}}}_{i} \) 的适度影响。这种现象可能是由于 \( \left| {F}_{i}\right| \) 的值只受 \( k \) 影响，独立于 \( n \) 。图9显示了不同 \( k \) 和 \( n \) 的 \( {Q}_{n}^{k} \) 的 \( {\mathrm{{SD}}}_{i} \) 。\( {\mathrm{{SD}}}_{i} \) 和 \( {\mathrm{{APL}}}_{i} \) 的增长趋势大致相同。然而，参数 \( n \) 可以显著影响 \( {\mathrm{{SD}}}_{i} \) 的增长率，这与 \( {\mathrm{{APL}}}_{i} \) 的增长率明显不同。这意味着 \( n \) 和 \( k \) 越大，路径长度的值越分散。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=162&y=174&w=1404&h=490"/>

Fig. 8. The \( {\mathrm{{APL}}}_{i} \) of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

图 8. 相邻节点对在 \( {\mathrm{{APL}}}_{i} \) 中的 \( {Q}_{n}^{k} \) 随不同的 \( n \) 和 \( k \) 的变化。

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=162&y=774&w=1413&h=488"/>

Fig. 9. The \( {\mathrm{{SD}}}_{i} \) of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

图 9. 相邻节点对在 \( {\mathrm{{SD}}}_{i} \) 中的 \( {Q}_{n}^{k} \) 随不同的 \( n \) 和 \( k \) 的变化。

<!-- Media -->

Next,we compute the APL of all adjacent node pairs in \( {Q}_{n}^{k} \) as follows:

接下来，我们计算了 \( {Q}_{n}^{k} \) 中所有相邻节点对的 APL，如下所示：

\[\mathrm{{APL}} = \frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{n}}}\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}p{l}_{i}^{j}}{n{k}^{n}}.\]

Correspondingly, the SD of all adjacent node pairs is also computed:

对应地，所有相邻节点对的 SD 也被计算出来：

\[\mathrm{{SD}} = \sqrt{\frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{n}}}\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}{\left( p{l}_{i}^{j} - \mathrm{{APL}}\right) }^{2}}{n{k}^{n}}}.\]

In Fig. 10,we show the APL and SD of \( {Q}_{n}^{k} \) . For \( {Q}_{3}^{3} \) ,the SD is slightly smaller than APL. However,with increasing \( k \) and \( n \) ,the SD grows faster than the APL. In addition, from the simulation, we are aware of the following attractive phenomena. (1) For \( {Q}_{3}^{k} \) , the ratio of APL to \( \left| F\right| \) is about 1.3. With \( k \) increases,the value of \( \left| F\right| / \) APL remains essentially the same. (2) For \( {Q}_{n}^{3} \) ,the value of \( \left| F\right| / \) APL varies approximately linearly with \( n \) . We list these results in Table II. As for what caused the positive correlation between \( n \) and \( \left| F\right| /\mathrm{{APL}} \) ,there is currently no definite attribution. We roughly think that since the algorithm HP-PEF constructs the required \( \mathrm{H} \) -path according to the distribution pattern of edge faults in each dimension, the H-path obtained is endowed with the nature of traversing all \( n \) dimensions,which results in an additional effect of \( n \) on the APL.

在图 10 中，我们展示了 \( {Q}_{n}^{k} \) 的 APL 和 SD。对于 \( {Q}_{3}^{3} \)，SD 略小于 APL。然而，随着 \( k \) 和 \( n \) 的增加，SD 的增长速度超过了 APL。此外，从模拟中，我们注意到了以下吸引人的现象。（1）对于 \( {Q}_{3}^{k} \)，APL 与 \( \left| F\right| \) 的比例约为 1.3。随着 \( k \) 的增加，\( \left| F\right| / \) APL 的值基本保持不变。（2）对于 \( {Q}_{n}^{3} \)，\( \left| F\right| / \) APL 的值随 \( n \) 线性变化。我们将这些结果列在表 II 中。至于是什么导致了 \( n \) 和 \( \left| F\right| /\mathrm{{APL}} \) 之间的正相关，目前还没有确切的归因。我们大致认为，由于算法 HP-PEF 根据每个维度边缘故障的分布模式构建所需的 \( \mathrm{H} \) -路径，得到的 H-路径具有穿越所有 \( n \) 维度的性质，这导致了 \( n \) 对 APL 的额外影响。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=898&y=1377&w=705&h=513"/>

Fig. 10. The APL and SD of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

图 10. \( {Q}_{n}^{k} \) 中相邻节点对的 APL 和 SD 随不同的 \( n \) 和 \( k \) 的变化。

TABLE II

THE \( \left| F\right| \) AND APL OF \( {Q}_{n}^{k} \) WITH DIFFERENT \( k \) AND \( n \)

THE \( \left| F\right| \) 和 \( {Q}_{n}^{k} \) 的 APL 与不同的 \( k \) 和 \( n \)

<table><tr><td colspan="9"/></tr><tr><td/><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>\( \left| F\right| \)</td><td>8</td><td>24</td><td>48</td><td>80</td><td>8</td><td>33</td><td>112</td><td>353</td></tr><tr><td>APL</td><td>6.3</td><td>17.9</td><td>36.9</td><td>61.8</td><td>6.3</td><td>14.8</td><td>35.3</td><td>88.1</td></tr><tr><td>\( \left| F\right| / \) APL</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>2.2</td><td>3.2</td><td>4.0</td></tr></table>

<table><tbody><tr><td colspan="9"></td></tr><tr><td></td><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>\( \left| F\right| \)</td><td>8</td><td>24</td><td>48</td><td>80</td><td>8</td><td>33</td><td>112</td><td>353</td></tr><tr><td>APL</td><td>6.3</td><td>17.9</td><td>36.9</td><td>61.8</td><td>6.3</td><td>14.8</td><td>35.3</td><td>88.1</td></tr><tr><td>\( \left| F\right| / \) APL</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>2.2</td><td>3.2</td><td>4.0</td></tr></tbody></table>

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_10.jpg?x=148&y=460&w=684&h=561"/>

Fig. 11. The comparisons among FP with different \( k \) .

图 11. 比较 不同 \( k \) 的 FP 之间的差异。

<!-- Media -->

## B. Fault Tolerance Analysis

## B. 容错性分析

1) Comparison Results: Improving the fault tolerance of \( {Q}_{n}^{k} \) when we embed an \( \mathrm{H} \) -path into the faulty \( {Q}_{n}^{k} \) is the original purpose of our work. In this subsection, we compare the fault tolerance of our method with the similar known result. First, let's review the known results. Yang et al. [32] proved that \( {Q}_{n}^{k} \) with odd \( k \geq  3 \) is(2n - 3)-edge fault-tolerant Hamiltonian-connected. We demonstrated that \( {Q}_{n}^{k} \) with odd \( k \geq  3 \) is \( \left( {\frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + }\right. \) 5)-edge fault-tolerant Hamiltonian-connected under the PEF model. For convenience,let \( \mathrm{{FT}} = {2n} - 3 \) and \( \mathrm{{FP}} = \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \) .

1) 对比结果：当我们将 \( \mathrm{H} \) -路径嵌入到故障 \( {Q}_{n}^{k} \) 中以提高其容错性是我们工作的初衷。在本节中，我们将我们的方法与已知的相似结果进行比较。首先，让我们回顾一下已知的结果。杨等人 [32] 证明了具有奇数 \( k \geq  3 \) 的 \( {Q}_{n}^{k} \) 是 (2n - 3)-边容错哈密顿连通的。我们证明了在 PEF 模型下，具有奇数 \( k \geq  3 \) 的 \( {Q}_{n}^{k} \) 是 \( \left( {\frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + }\right. \) 5)-边容错哈密顿连通的。为了方便起见，设 \( \mathrm{{FT}} = {2n} - 3 \) 和 \( \mathrm{{FP}} = \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \) 。

Obviously, the value of FP is closely related to the parameters \( k \) and \( n \) . We first investigate how odd \( k \) with different values affect FP in Fig. 11. Note that it takes \( n \) on its \( x \) -axis and takes the corresponding metrics with the exponential scale on its \( y \) -axis. It can be seen that the values of FP with different \( k \) are all the same when \( n = 2 \) . That is because the base case of the induction in our proof is handled by the result in [42], which is irrelevant to the parameter \( k \) . As \( n \) increases,the distance between any two FP with different \( k \) increases rapidly. It’s easy to see that FP is positively correlated with \( k \) . When \( n = 3 \) ,FP with \( k = 9 \) is ten times that with \( k = 3 \) . When \( n = 8 \) ,FP with \( k = 9 \) is more than \( {1648}\left( { \approx  \frac{5380819}{3265}}\right) \) times that with \( k = 3 \) .

显然，FP的值与参数 \( k \) 和 \( n \) 密切相关。我们首先研究不同值的 \( k \) 如何影响图11中的FP。注意，它在 \( x \) 轴上取 \( n \) ，并在 \( y \) 轴上以指数尺度取相应的度量。可以看出，当 \( n = 2 \) 时，不同 \( k \) 的FP值都是相同的。这是因为我们证明中归纳的基础情况是由 [42] 的结果处理的，这与参数 \( k \) 无关。随着 \( n \) 的增加，任何两个不同 \( k \) 的FP之间的距离迅速增加。很容易看出，FP与 \( k \) 正相关。当 \( n = 3 \) 时，具有 \( k = 9 \) 的FP是具有 \( k = 3 \) 的FP的十倍。当 \( n = 8 \) 时，具有 \( k = 9 \) 的FP是具有 \( k = 3 \) 的FP的超过 \( {1648}\left( { \approx  \frac{5380819}{3265}}\right) \) 倍。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_10.jpg?x=894&y=188&w=745&h=542"/>

Fig. 12. The comparisons between FT and FP.

图12。FT与FP的比较。

<!-- Media -->

Next, we compare the values of FP and FT. Since the value of FT is irrelevant to the parameter \( k \) ,we fix \( k = 3 \) in FP to ensure fairness when making the comparisons. Fig. 12 shows the trends of FT and FP with increasing \( n \) . Note that it takes even \( n \) on its \( x \) -axis and takes the corresponding metrics with the exponential scale on its \( y \) -axis. Since the base case of both our method and the method in [32] is handled by the result in [42], FT = FP when \( n = 2 \) . However,as \( n \) increases,FP and the ratio of FP to FT increase rapidly,while FT is linear on \( n \) . When \( n = {10},\mathrm{{FT}} = {17} \) and \( \mathrm{{FP}} = {29505} \) ,which implies \( \mathrm{{FP}}/\mathrm{{FT}} = \frac{29505}{17} \approx  {1735} \) . The great disparity between them comes from our complete consideration of the distribution pattern of edge faults in each dimension.

接下来，我们比较 FP 和 FT 的值。由于 FT 的值与参数 \( k \) 无关，我们在 FP 中固定 \( k = 3 \) 以确保在比较时的公平性。图 12 显示了随着 \( n \) 的增加，FT 和 FP 的趋势。注意，其在 \( x \) 轴上均匀取 \( n \) 的值，并在 \( y \) 轴上以指数尺度取相应的指标。由于我们的方法和文献 [32] 中的方法的基础情况都由文献 [42] 的结果处理，当 \( n = 2 \) 时，FT = FP。然而，随着 \( n \) 的增加，FP 以及 FP 与 FT 的比值迅速增加，而 FT 在 \( n \) 上是线性的。当 \( n = {10},\mathrm{{FT}} = {17} \) 和 \( \mathrm{{FP}} = {29505} \) 时，这意味着 \( \mathrm{{FP}}/\mathrm{{FT}} = \frac{29505}{17} \approx  {1735} \) 。它们之间的巨大差异来源于我们对每个维度边缘故障分布模式的完整考虑。

The fault tolerance of \( {Q}_{n}^{k} \) under the PEF model is positively correlated with \( k \) and \( n \) . This proposed model builds the positive relation between the fault tolerance with the parameter \( k \) ,which is its unique advantage. Even without considering the positive effect of \( k \) ,our method still outperforms the similar known result in terms of fault tolerance. Thus, our method can evaluate the fault tolerance of \( {Q}_{n}^{k} \) even when the edge faults are large-scale.

在 PEF 模型下，\( {Q}_{n}^{k} \) 的容错性与 \( k \) 和 \( n \) 正相关。该提议的模型建立了容错性与参数 \( k \) 之间的正相关关系，这是其独特的优势。即使不考虑 \( k \) 的积极影响，我们的方法在容错性方面仍然优于已知相似结果。因此，我们的方法即使是在边缘故障大规模的情况下，也能评估 \( {Q}_{n}^{k} \) 的容错性。

2) Average Success Rate: Next, we analyze the fault-tolerant capacity of the algorithm HP-PEF with increasing faulty edges. Given any PEF set \( F \) satisfying the three conditions in Theorem III.2, the required H-path can be constructed by HP-PEF with \( {100}\% \) probability,which has been proved in Theorem IV.1. In the following,provided that the PEF set \( F \) exceeds the above conditions, by implementing computer programs, we evaluate the average success rate (ASR for short) of HP-PEF, which is the ratio of the number of successfully constructed \( \mathrm{H} \) -paths over generated instances.

2) 平均成功率：接下来，我们分析算法 HP-PEF 在增加故障边时的容错能力。对于满足定理 III.2 中三个条件的任意 PEF 集合 \( F \)，所需的 H-路径可以通过 HP-PEF 以 \( {100}\% \) 的概率构建，这一点已在定理 IV.1 中得到证明。以下假设 PEF 集合 \( F \) 超出上述条件，通过实施计算机程序，我们评估 HP-PEF 的平均成功率（简称 ASR），即成功构建的 \( \mathrm{H} \) -路径数量与生成的实例数量的比率。

Before the simulation, we randomly generate one thousand PEF sets of \( {Q}_{n}^{k} \) ,denoted by \( F\left\lbrack  i\right\rbrack \) with \( i \in  {\mathbb{Z}}_{{10}^{3}} \) ,which satisfy (1) \( \left| {F\left\lbrack  i\right\rbrack  }\right|  = \mathrm{{FP}} \) ,(2) \( {e}_{j} = {k}^{j} - 2 \) for each \( j \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ,and (3) \( {e}_{0} = 0 \) and \( {e}_{1} = 1 \) . Next,in each simulation,we randomly add exactly one faulty edge into \( F\left\lbrack  i\right\rbrack \) such that \( {e}_{j} \) grows by 1 with \( j \in \) \( {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) . We don’t add faulty edges to increase \( {e}_{0} \) or \( {e}_{1} \) since it will seriously affect the effectiveness of Algorithm 1, which is designed based on the method of [42]. Then, for each faulty edge set \( F\left\lbrack  i\right\rbrack \) ,we determine whether the required \( \mathrm{H} \) -path between the node pair(s,t)is constructed successfully (i.e.,success \( \left( j\right)  = \) 1) or not (i.e.,success \( \left( j\right)  = 0 \) ),where \( j \in  {\mathbb{Z}}_{{k}^{2n} - {k}^{n}} \) is the unique identification of the node pair(s,t). Then we compute the \( {\mathrm{{ASR}}}_{i} \) of \( F\left\lbrack  i\right\rbrack \) as follows:

在模拟之前，我们随机生成一千个满足以下条件的 PEF 集合 \( {Q}_{n}^{k} \)，记为 \( F\left\lbrack  i\right\rbrack \) ，其中 \( i \in  {\mathbb{Z}}_{{10}^{3}} \)：（1）\( \left| {F\left\lbrack  i\right\rbrack  }\right|  = \mathrm{{FP}} \)，（2）对于每个 \( j \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ，\( {e}_{j} = {k}^{j} - 2 \)；（3）\( {e}_{0} = 0 \) 且 \( {e}_{1} = 1 \)。接下来，在每次模拟中，我们随机向 \( F\left\lbrack  i\right\rbrack \) 中添加一条故障边，使得 \( {e}_{j} \) 增加 1，且 \( j \in \) \( {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \)。我们不会添加故障边以增加 \( {e}_{0} \) 或 \( {e}_{1} \)，因为这会严重影响基于方法 [42] 设计的算法 1 的有效性。然后，对于每个故障边集合 \( F\left\lbrack  i\right\rbrack \)，我们确定在节点对 (s,t) 之间是否成功构建了所需的 \( \mathrm{H} \) -路径（即，成功 \( \left( j\right)  = \) 1）或未成功（即，成功 \( \left( j\right)  = 0 \)），其中 \( j \in  {\mathbb{Z}}_{{k}^{2n} - {k}^{n}} \) 是节点对 (s,t) 的唯一标识符。然后我们按如下方式计算 \( {\mathrm{{ASR}}}_{i} \) 的 \( F\left\lbrack  i\right\rbrack \)：

\[{\mathrm{{ASR}}}_{i} = \frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{2n} - {k}^{n}}}}\operatorname{success}\left( j\right) }{{k}^{2n} - {k}^{n}} \times  {100}\% .\]

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_11.jpg?x=201&y=183&w=1326&h=515"/>

Fig. 13. The \( {\mathrm{{ASR}}}_{\min } \) and ASR of \( {Q}_{3}^{3},{Q}_{4}^{3} \) ,and \( {Q}_{3}^{5} \) with half dots drawn.

图 13。 \( {\mathrm{{ASR}}}_{\min } \) 和 \( {Q}_{3}^{3},{Q}_{4}^{3} \) 的 ASR 以及带有半点的 \( {Q}_{3}^{5} \) 绘图。

<!-- Media -->

Based on \( {\mathrm{{ASR}}}_{i} \) ,we then compute the \( {\mathrm{{ASR}}}_{\min } \) and \( \mathrm{{ASR}} \) as follows:

基于 \( {\mathrm{{ASR}}}_{i} \)，然后我们计算如下 \( {\mathrm{{ASR}}}_{\min } \) 和 \( \mathrm{{ASR}} \)：

\[{\mathrm{{ASR}}}_{\min } = \min \left\{  {{\mathrm{{ASR}}}_{i} \mid  i \in  {\mathbb{Z}}_{{10}^{3}}}\right\}  .\]

\[\mathrm{{ASR}} = \frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{{10}^{3}}}}{\mathrm{{ASR}}}_{i}}{{10}^{3}}.\]

For \( \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {80} \) ,we show the variation trend of \( {\mathrm{{ASR}}}_{\min } \) and ASR of \( {Q}_{3}^{3},{Q}_{4}^{3} \) ,and \( {Q}_{3}^{5} \) in Fig. 13,respectively. Note that for \( {Q}_{3}^{3} \) (resp. \( {Q}_{4}^{3} \) and \( {Q}_{3}^{5} \) ),FP \( = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 = 8 \) (resp. 33 and 24). It can be observed that both \( {\mathrm{{ASR}}}_{\min } \) and ASR support dynamic degradation. For \( {Q}_{3}^{3},{\mathrm{{ASR}}}_{\min } \) (resp. ASR) maintains over 47.2% (resp. 90.2%) even when increasing faulty edges achieve 2FP (i.e., 16).

对于 \( \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {80} \)，我们在图 13 中分别展示了 \( {\mathrm{{ASR}}}_{\min } \) 和 ASR 的变化趋势以及 \( {Q}_{3}^{5} \)。注意，对于 \( {Q}_{3}^{3} \)（分别对于 \( {Q}_{4}^{3} \) 和 \( {Q}_{3}^{5} \)），FP \( = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 = 8 \)（分别对应 33 和 24）。可以观察到 \( {\mathrm{{ASR}}}_{\min } \) 和 ASR 都支持动态降级。对于 \( {Q}_{3}^{3},{\mathrm{{ASR}}}_{\min } \)（分别对于 ASR）即使故障边增加至 2FP（即 16）也保持了超过 47.2%（分别对于 90.2%）。

As \( n \) or \( k \) increases,the decline curve of ASR becomes moderate. The ASR of \( {Q}_{4}^{3} \) (resp. \( {Q}_{3}^{5} \) ) maintains \( {96.2}\% \) (resp. over 100%) when increasing faulty edges achieve 2FP (i.e., 66 and 48, respectively). As for \( {\mathrm{{ASR}}}_{\min } \) ,the curve is much steeper than that of ASR though it still decreases in general. We can observe that there exist several intervals in the curve where \( {\mathrm{{ASR}}}_{\min } \) remains unchanged. For instance,when \( {42} \leq  \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {46} \) ,the \( {\mathrm{{ASR}}}_{\text{min }} \) of \( {Q}_{4}^{3} \) is always equal to \( {92}\% \) . After the increasing faulty edges exceed 46,the \( {\mathrm{{ASR}}}_{\text{min }} \) of \( {Q}_{4}^{3} \) exhibits a cliff-like drop to \( {77}\% \) . The reason behind this phenomenon can be easily explained. In the for-loops of Procedures HP-Round and HP-Direct, some expected fault-free edges connecting two consecutive subgraphs are selected to construct the H-path, which determines whether algorithm HP-PEF can successfully be implemented. Thus, if the faulty edges we add into \( F\left\lbrack  i\right\rbrack \) aren’t concentrated between two consecutive subgraphs, then the success rate of HP-PEF will not be affected by the number of faulty edges. However, when the number of faulty edges exceeds a threshold value which makes that the procedures cannot successfully select the expected edges connecting two consecutive subgraphs,all the \( \mathrm{H} \) -paths that should pass through these two consecutive subgraphs cannot be successfully constructed.

随着 \( n \) 或 \( k \) 的增加，ASR 的下降曲线变得平缓。当增加的故障边达到 2FP（即 66 和 48 分别对应）时，\( {Q}_{4}^{3} \)（相应地 \( {Q}_{3}^{5} \)）的 ASR 保持 \( {96.2}\% \)（相应地超过 100%）。至于 \( {\mathrm{{ASR}}}_{\min } \)，其曲线比 ASR 的曲线要陡峭得多，尽管总体上仍在下降。我们可以观察到曲线中存在几个区间，\( {\mathrm{{ASR}}}_{\min } \) 在这些区间内保持不变。例如，当 \( {42} \leq  \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {46} \) 时，\( {Q}_{4}^{3} \) 的 \( {\mathrm{{ASR}}}_{\text{min }} \) 总是等于 \( {92}\% \)。在增加的故障边超过 46 之后，\( {Q}_{4}^{3} \) 的 \( {\mathrm{{ASR}}}_{\text{min }} \) 出现悬崖式下降至 \( {77}\% \)。这个现象背后的原因很容易解释。在 HP-Round 和 HP-Direct 过程的 for 循环中，选择一些预期的无故障边连接两个连续子图来构建 H 路径，这决定了算法 HP-PEF 是否能够成功实施。因此，如果我们添加到 \( F\left\lbrack  i\right\rbrack \) 中的故障边没有集中在两个连续子图之间，那么 HP-PEF 的成功率将不会受故障边数量的影响。然而，当故障边的数量超过一个阈值，使得过程无法成功选择连接两个连续子图的预期边时，所有应该通过这两个连续子图的 \( \mathrm{H} \) 路径都无法成功构建。

It can be observed that the larger the values of \( n \) and \( k \) ,the smaller the decline rate of both ASR and \( {\mathrm{{ASR}}}_{\text{min }} \) . Therefore,it can be predicted that the algorithm HP-PEF will exhibit a more excellent fault-tolerant capacity for \( {Q}_{n}^{k} \) systems with larger \( n \) and \( k \) .

可以观察到，\( n \) 和 \( k \) 的值越大，ASR 和 \( {\mathrm{{ASR}}}_{\text{min }} \) 的下降率越小。因此，可以预测算法 HP-PEF 对于具有较大 \( n \) 和 \( k \) 值的 \( {Q}_{n}^{k} \) 系统将展现出更优秀的容错能力。

## VI. Concluding Remarks

## VI. 结论性评述

We introduce a new edge-fault model named PEF model, for the purpose of embedding \( \mathrm{H} \) -paths into the \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) containing a large number of faulty edges. Using the PEF model, we design, as well as implement, an efficient fault-tolerant H-path embedding algorithm for \( {Q}_{n}^{k} \) . Due to the recursive structure of \( {Q}_{n}^{k} \) and its edge symmetry property,we believe similar algorithms can be designed for other recursive, edge symmetric networks, such as generalized hypercubes and star graphs. Another promising direction of future work is to investigate the embedding of the H-path in \( k \) -ary \( n \) -cubes under the PEF model, so that the H-path passes through some prescribed linear forests, exponentially improving the recent results in [34].

我们介绍了一种新的边缘故障模型，名为PEF模型，目的是将 \( \mathrm{H} \) -路径嵌入到包含大量故障边的 \( k \) -ary \( n \) -立方体 \( {Q}_{n}^{k} \) 中。使用PEF模型，我们设计并实现了针对 \( {Q}_{n}^{k} \) 的高效容错H-路径嵌入算法。由于 \( {Q}_{n}^{k} \) 的递归结构及其边缘对称性，我们相信可以为其他递归、边缘对称网络设计出类似算法，例如广义超立方体和星形图。未来工作的另一个有前景的方向是研究在PEF模型下，\( k \) -ary \( n \) -立方体中的H-路径嵌入，使得H-路径通过某些指定的线性森林，指数级地改进了文献[34]中的近期结果。

## REFERENCES

## 参考文献

[1] W. J. Dally and B. Towles, "Route packets, not wires: On-chip interconnection networks," in Proc. 38th Des. Automat. Conf., 2001, pp. 684-689.

[1] W. J. Dally和B. Towles, "Route packets, not wires: On-chip interconnection networks," in Proc. 38th Des. Automat. Conf., 2001, pp. 684-689.

[2] A. Jantsch and H. Tenhunen, Networks on Chip, Norwell, MA, USA: Kluwer, 2003.

[2] A. Jantsch和H. Tenhunen, Networks on Chip, Norwell, MA, USA: Kluwer, 2003.

[3] J. H. Lau, "Evolution, challenge, and outlook of TSV, 3D IC integration and 3D silicon integration," in Proc. IEEE Int. Symp. Adv. Packag. Mater., 2011, pp. 462-488.

[3] J. H. Lau, "Evolution, challenge, and outlook of TSV, 3D IC integration and 3D silicon integration," in Proc. IEEE Int. Symp. Adv. Packag. Mater., 2011, pp. 462-488.

[4] P. Bogdan, T. Dumitraş, and R. Marculescu, "Stochastic communication: A new paradigm for fault-tolerant networks-on-chip," VLSI Des., vol. 2007, p. 17, 2007.

[4] P. Bogdan, T. Dumitraş,和R. Marculescu, "Stochastic communication: A new paradigm for fault-tolerant networks-on-chip," VLSI Des., vol. 2007, p. 17, 2007.

[5] Y. Zhang et al., "A deterministic-path routing algorithm for tolerating many faults on very-large-scale network-on-chip," ACM Trans. Des. Automat. Electron. Syst., vol. 26, no. 1, pp. 1-26, Jan. 2021.

[5] 张宇等，"一种适用于容忍大规模网络芯片上多故障的确定性路径路由算法"，ACM Trans. Des. Automat. Electron. Syst.，卷26，期1，页码1-26，2021年1月。

[6] E. Taheri, M. Isakov, A. Patooghy, and M. A. Kinsy, "Addressing a new class of reliability threats in 3-D network-on-chips," IEEE Trans. Comput. Aided Des. Integr. Circuits Syst., vol. 39, no. 7, pp. 1358-1371, Jul. 2020.

[6] 塔赫里，伊萨科夫，帕图吉，金斯基，"应对三维网络芯片中的新型可靠性威胁"，IEEE Trans. Comput. Aided Des. Integr. Circuits Syst.，卷39，期7，页码1358-1371，2020年7月。

[7] M. Ebrahimi, M. Daneshtalab, and J. Plosila, "Fault-tolerant routing algorithm for 3D NoC using Hamiltonian path strategy," in Proc. Des. Autom. Test Eur. Conf. Exhib., 2013, pp. 1601-1604.

[7] 易卜拉希米，达内什塔拉布，普洛西拉，"基于哈密顿路径策略的三维网络芯片容错路由算法"，2013年设计自动化测试欧洲会议展览，页码1601-1604。

[8] C. Hu, C. M. Meyer, X. Jiang, and T. Watanabe, "A fault-tolerant Hamiltonian-based odd-even routing algorithm for network-on-chip," in Proc. IEEE 35th Int. Technol. Conf. Circuits/Syst. Circuits Commun., 2020, pp. 217-222.

[8] 胡，梅耶，姜，渡边，"一种基于哈密顿的奇偶网络芯片容错路由算法"，2020年IEEE第35届国际技术会议电路/系统电路通信，页码217-222。

[9] P. Bahrebar and D. Stroobandt, "Improving Hamiltonian-based routing methods for on-chip networks: A turn model approach," in Proc. Des. Autom. Test Eur. Conf. Exhib., 2014, pp. 1-4.

[9] 巴雷巴尔，斯特鲁班特，"改进基于哈密顿的芯片网络路由方法：转向模型方法"，2014年设计自动化测试欧洲会议展览，页码1-4。

[10] P. Bahrebar and D. Stroobandt, "The Hamiltonian-based odd-even turn model for maximally adaptive routing in 2D mesh networks-on-chip," Comput. Electr. Eng., vol. 45, pp. 386-401, Jul. 2015.

[10] P. Bahrebar 和 D. Stroobandt, "基于哈密顿算子的奇偶转向模型，用于2D网格网络上的最大适应性路由," Comput. Electr. Eng., 卷45，页386-401，Jul. 2015。

[11] E. O. Amnah and W. L. Zuo, "Hamiltonian paths for designing deadlock-free multicasting wormhole-routing algorithms in 3-D meshes," J. Appl. Sci., vol. 7, no. 22, pp. 3410-3419, 2007.

[11] E. O. Amnah 和 W. L. Zuo, "用于设计3-D网格中无死锁的多播虫孔路由算法的哈密顿路径," J. Appl. Sci., 卷7，号22，页3410-3419，2007。

[12] M. Ebrahimi, M. Daneshtalab, P. Liljeberg, and H. Tenhunen, "HAMUM - A novel routing protocol for unicast and multicast traffic in MPSoCs," in Proc. 18th Euromicro Int. Conf. Parallel Distrib. Netw. Based Process., 2010, pp. 525-532.

[12] M. Ebrahimi, M. Daneshtalab, P. Liljeberg, 和 H. Tenhunen, "HAMUM - 面向MPSoCs单播和多播通信的新型路由协议," in Proc. 第18届 Euromicro 国际并行分布式网络处理会议，2010，页525-532。

[13] X. Lin and L. M. Ni, "Multicast communication in multicomputer networks," IEEE Trans. Parallel Distrib. Syst., vol. 4, no. 10, pp. 1105-1117, Oct. 1993.

[13] X. Lin 和 L. M. Ni, "多计算机网络中的多播通信," IEEE Trans. Parallel Distrib. Syst., 卷4，号10，页1105-1117，Oct. 1993。

[14] W. J. Dally,"Performance analysis of \( k \) -ary \( n \) -cube interconnection networks," IEEE Trans. Comput., vol. 19, no. 6, pp. 775-785, Jun. 1990.

[14] W. J. Dally, "性能分析 \( k \) -元 \( n \) -立方体互联网络," IEEE Trans. Comput., 卷19，号6，页775-785，Jun. 1990。

[15] W. J. Dally and C. L. Seitz, "The tours routing chip," Distrib. Comput., vol. 1, no. 3, pp. 187-196, Dec. 1986.

[15] W. J. Dally 和 C. L. Seitz, "Tours路由芯片," Distrib. Comput., 卷1，号3，页187-196，Dec. 1986。

[16] D. H. Linder and J. C. Harden, "An adaptive and fault tolerant wormhole routing strategy for \( k \) -ary \( n \) -cubes," IEEE Trans. Comput.,vol. 40,no. 1, pp. 2-12, Jan. 1991.

[16] D. H. Linder和J. C. Harden, "一种自适应且容错的虫洞路由策略，适用于 \( k \) -元 \( n \) -立方体," IEEE Trans. Comput.，卷40，期1，页码2-12，1991年1月。

[17] W. Luo and D. Xiang, "An efficient adaptive deadlock-free routing algorithm for torus networks," IEEE Trans. Parallel Distrib. Syst., vol. 23, no. 5, pp. 800-808, May 2012.

[17] W. Luo和D. Xiang, "一种高效的自适应无死锁路由算法，适用于环面网络," IEEE Trans. Parallel Distrib. Syst.，卷23，期5，页码800-808，2012年5月。

[18] P. Ren, X. Ren, S. Sane, M. A. Kinsy, and N. Zheng, "A deadlock-free and connectivity-guaranteed methodology for achieving fault-tolerance in on-chip networks," IEEE Trans. Comput., vol. 65, no. 2, pp. 353-366, Feb. 2016.

[18] P. Ren, X. Ren, S. Sane, M. A. Kinsy和N. Zheng, "一种无死锁且保证连通性的方法，用于实现片上网络的容错性," IEEE Trans. Comput.，卷65，期2，页码353-366，2016年2月。

[19] H. Abu-Libdeh et al., "Symbiotic routing in future data centers," ACM SIGCOMM Comput. Commun. Rev., vol. 40, no. 4, pp. 51-62, Oct. 2010.

[19] H. Abu-Libdeh等人，"未来数据中心的共生路由," ACM SIGCOMM Comput. Commun. Rev.，卷40，期4，页码51-62，2010年10月。

[20] T. Wang, Z. Su, Y. Xia, B. Qin, and M. Hamdi, "NovaCube: A low latency torus-based network architecture for data centers," in Proc. IEEE Glob. Commun. Conf., 2014, pp. 2252-2257.

[20] T. Wang, Z. Su, Y. Xia, B. Qin和M. Hamdi, "NovaCube: 一种面向数据中心的低延迟环面网络架构," 在IEEE Glob. Commun. Conf.会议论文集，2014年，页码2252-2257。

[21] T. Wang, Z. Su, Y. Xia, and M. Hamdi, "CLOT: A cost-effective low-latency overlaid torus-based network architecture for data centers," in Proc. IEEE Conf. Commun., 2015, pp. 5479-5484.

[21] T. Wang, Z. Su, Y. Xia和M. Hamdi, "CLOT: 一种面向数据中心的低成本低延迟叠加环面网络架构," 在IEEE Conf. Commun.会议论文集，2015年，页码5479-5484。

[22] K. Chen et al., "WaveCube: A scalable, fault-tolerant, high-performance optical data center architecture," in Proc. IEEE Conf. Comput. Commun., 2015, pp. 1903-1911.

[22] K. Chen等人，“WaveCube：一种可扩展、容错、高性能的光学数据中心架构”，在IEEE计算机通信会议论文集，2015年，pp. 1903-1911。

[23] S. Borkar et al., "iWarp: An integrated solution to high-speed parallel computing," in Proc. Int. Conf. Supercomput., 1988, pp. 330-339.

[23] S. Borkar等人，“iWarp：一种高速并行计算的集成解决方案”，在国际超级计算会议论文集，1988年，pp. 330-339。

[24] M. D. Noakes, D. A. Wallach, and W. J. Dally, "The J-Machine multicomputer: An architectural evaluation," in Proc. 20th Ann. Int. Symp. Comput. Architecture, 1993, pp. 224-235.

[24] M. D. Noakes, D. A. Wallach和W. J. Dally，“J-Machine多计算机：架构评估”，在第20届年度国际计算机体系结构研讨会论文集，1993年，pp. 224-235。

[25] R. E. Kessler and J. L. Schwarzmeier, "Cray T3D: A new dimension for Cray research," in Proc. 38th IEEE Int. Comput. Conf., 1993, pp. 176-182.

[25] R. E. Kessler和J. L. Schwarzmeier，“Cray T3D：Cray研究的新维度”，在第38届IEEE国际计算机会议论文集，1993年，pp. 176-182。

[26] E. Anderson, J. Brooks, C. Grassl, and S. Scott, "Performance of the Cray T3E multiprocessor," in Proc. ACM/IEEE Conf. Supercomput., 1997, pp. 1-17.

[26] E. Anderson, J. Brooks, C. Grassl和S. Scott，“Cray T3E多处理器的性能”，在ACM/IEEE超级计算会议论文集，1997年，pp. 1-17。

[27] N. R. Adiga et al., "Blue Gene/L torus interconnection network," IBM J. Res. Dev., vol. 49, no. 2/3, pp. 265-276, Mar. 2005.

[27] N. R. Adiga等人，“Blue Gene/L环状互联网络”，IBM研究与发展杂志，卷49，第2/3期，pp. 265-276，2005年3月。

[28] Y. Lv, C.-K. Lin, J. Fan, and X. Jia, "Hamiltonian cycle and path embed-dings in 3-ary \( n \) -cubes based on \( {K}_{1,3} \) -structure faults," J. Parallel Distrib. Comput., vol. 120, pp. 148-158, 2018.

[28] 吕阳, 林志坚, 樊建, 贾晓, "基于 \( n \) 结构故障的 3-元 \( {K}_{1,3} \) -立方体的哈密顿回路和路径嵌入," 《并行分布式计算杂志》, 第120卷, 第148-158页, 2018年。

[29] S. Wang, S. Zhang, and Y. Yang, "Hamiltonian path embeddings in conditional faulty \( k \) -ary \( n \) -cubes," Inf. Sci.,vol. 268,pp. 463-488,Jun. 2014.

[29] 王森, 张三, 杨阳, "在条件故障的 \( k \) -元 \( n \) -立方体中的哈密顿路径嵌入," 《信息科学》, 第268卷, 第463-488页, 2014年6月。

[30] Y. Xiang and I. A. Stewart,"Bipancyclicity in \( k \) -ary \( n \) -cubes with faulty edges under a conditional fault assumption," IEEE Trans. Parallel Distrib. Syst., vol. 22, no. 9, pp. 1506-1513, Sep. 2011.

[30] 向阳, 斯图尔特·伊恩·A, "在条件故障假设下带有故障边的 \( k \) -元 \( n \) -立方体的双泛周期性," 《IEEE并行与分布式系统汇刊》, 第22卷, 第9期, 第1506-1513页, 2011年9月。

[31] S. Zhang and X. Zhang, "Fault-free Hamiltonian cycles passing through prescribed edges in \( k \) -ary \( n \) -cubes with faulty edges," IEEE Trans. Parallel Distrib. Syst., vol. 26, no. 2, pp. 434-443, Feb. 2015.

[31] 张四, 张晓, "在带有故障边的 \( k \) -元 \( n \) -立方体中通过指定边的无故障哈密顿回路," 《IEEE并行与分布式系统汇刊》, 第26卷, 第2期, 第434-443页, 2015年2月。

[32] M.-C. Yang, J. J. M. Tan, and L.-H. Hsu, "Hamiltonian circuit and linear array embeddings in faulty \( k \) -ary \( n \) -cubes," J. Parallel Distrib. Comput., vol. 67, pp. 362-368, Apr. 2007.

[32] 杨美慈, 谭家骏, 胡立宏, "在故障的 \( k \) -元 \( n \) -立方体中的哈密顿回路和线性阵列嵌入," 《并行分布式计算杂志》, 第67卷, 第362-368页, 2007年4月。

[33] I. A. Stewart and Y. Xiang,"Embedding long paths in \( k \) -ary \( n \) -cubes with faulty nodes and links," IEEE Trans. Parallel Distrib. Syst., vol. 19, no. 8, pp. 1071-1085, Aug. 2008.

[33] I. A. Stewart 和 Y. Xiang, "在具有故障节点和链路的 \( k \) 元 \( n \) 立方体中嵌入长路径," IEEE Trans. Parallel Distrib. Syst., 卷 19, 第 8 期, 页码 1071-1085, 2008 年 8 月。

[34] Y. Yang and L. Zhang,"Hamiltonian paths of \( k \) -ary \( n \) -cubes avoiding faulty links and passing through prescribed linear forests," IEEE Trans. Parallel Distrib. Syst., vol. 33, no. 7, pp. 1752-1760, Jul. 2022.

[34] Y. Yang 和 L. Zhang, "避免故障链路并穿过指定线性森林的 \( k \) 元 \( n \) 立方体的哈密顿路径," IEEE Trans. Parallel Distrib. Syst., 卷 33, 第 7 期, 页码 1752-1760, 2022 年 7 月。

[35] J. Yuan et al.,"The \( g \) -good-neighbor conditional diagnosability of \( k \) -ary \( n \) -cubes under the PMC model and MM model," IEEE Trans. Parallel Distrib. Syst., vol. 26, no. 4, pp. 1165-1177, Apr. 2015.

[35] J. Yuan 等, "在 PMC 模型和 MM 模型下 \( g \) 元 \( k \) 立方体的 \( n \) 良好邻居条件可诊断性," IEEE Trans. Parallel Distrib. Syst., 卷 26, 第 4 期, 页码 1165-1177, 2015 年 4 月。

[36] L. Xu, L. Lin, S. Zhou, and S.-Y. Hsieh, "The extra connectivity, extra conditional diagnosability,and \( t/m \) -diagnosability of arrangement graphs," IEEE Trans. Rel., vol. 65, no. 3, pp. 1248-1262, Sep. 2016.

[36] L. Xu, L. Lin, S. Zhou 和 S.-Y. Hsieh, "排列图的额外连通性、额外条件可诊断性和 \( t/m \) 可诊断性," IEEE Trans. Rel., 卷 65, 第 3 期, 页码 1248-1262, 2016 年 9 月。

[37] R. Salamat, M. Khayambashi, M. Ebrahimi, and N. Bagherzadeh, "A resilient routing algorithm with formal reliability analysis for partially connected 3D-NoCs," IEEE Trans. Comput., vol. 65, no. 11, pp. 3265- 3279, Nov. 2016.

[37] R. Salamat, M. Khayambashi, M. Ebrahimi 和 N. Bagherzadeh, "具有形式可靠性分析的面向部分连接三维网络芯片的弹性路由算法," IEEE Trans. Comput., 卷 65, 第 11 期, 页码 3265-3279, 2016 年 11 月。

[38] A. Charif, A. Coelho, M. Ebrahimi, N. Bagherzadeh, and N.-E. Zergainoh, "First-Last: A cost-effective adaptive routing solution for TSV-based three-dimensional networks-on-chip," IEEE Trans. Comput., vol. 67, no. 10, pp. 1430-1444, Oct. 2018.

[38] A. Charif, A. Coelho, M. Ebrahimi, N. Bagherzadeh, 和 N.-E. Zergainoh, "First-Last: 一种基于TSV的三维网络芯片的成本效益型自适应路由解决方案," IEEE Trans. Comput., 卷67，第10期，页码1430-1444，2018年10月。

[39] H. Zhang, R.-X. Hao, X.-W. Qin, C.-K. Lin, and S.-Y. Hsieh, "The high faulty tolerant capability of the alternating group graphs," IEEE Trans. Parallel Distrib. Syst., vol. 34, no. 1, pp. 225-233, Jan. 2023.

[39] H. Zhang, R.-X. Hao, X.-W. Qin, C.-K. Lin, 和 S.-Y. Hsieh, "交错群图的故障容错能力," IEEE Trans. Parallel Distrib. Syst., 卷34，第1期，页码225-233，2023年1月。

[40] L.-H. Hsu and C.-K. Lin, Graph Theory and Interconnection Networks. Boca Raton, FL, USA: CRC Press, 2008.

[40] L.-H. Hsu 和 C.-K. Lin, 图论与互连网络. 美国佛罗里达州博卡拉顿: CRC出版社，2008年。

[41] Y. A. Ashir and I. A. Stewart,"On embedding cycles in \( k \) -ary \( n \) -cubes," Parallel Process. Lett., vol. 7, no. 1, pp. 49-55, 1997.

[41] Y. A. Ashir 和 I. A. Stewart, "关于在 \( k \) 元 \( n \) 立方体中嵌入环的研究," Parallel Process. Lett., 卷7，第1期，页码49-55，1997年。

[42] H.-C. Kim and J.-H. Park, "Fault Hamiltonicity of two-dimensional torus networks," in Proc. 5th Jpn. Korea Joint Workshop Algorithms Comput., 2000, pp. 110-117.

[42] H.-C. Kim 和 J.-H. Park, "二维环面网络的故障哈密顿性," in Proc. 第5届日韩联合算法计算研讨会，2000年，页码110-117。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_12.jpg?x=889&y=1244&w=232&h=289"/>

<!-- Media -->

Hongbin Zhuang received the BEng degree from Huaqiao University, Xiamen, China, in 2019. He is currently working toward the doctoral degree with the College of Computer and Data Science, Fuzhou University, China. His research interests include design and analysis of algorithms, fault diagnosis, and fault-tolerant computing.

郑红彬于2019年在中国厦门华侨大学获得工学学士学位。他目前在中国福州大学计算机与数据科学学院攻读博士学位。他的研究兴趣包括算法设计与分析、故障诊断和容错计算。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_12.jpg?x=892&y=1670&w=222&h=275"/>

<!-- Media -->

Xiao-Yan Li received the PhD degree in computer science from Soochow University, Suzhou, China, in 2019. She was a visiting scholar in the Department of Computer Science, the City University of Hong Kong, Hong Kong, from June 2018-June 2019. She is currently an associate professor with the College of Computer and Data Science, Fuzhou University, China. She has published more than 30 papers in research-related journals and conferences, such as IEEE Transactions on Parallel and Distributed Systems, IEEE/ACM Transactions on Networking, IEEE Transactions on Computers, Journal of Parallel and Distributed Computing, and Association for the Advancement of Artificial Intelligence. She has served some conferences as the Session Chair and Program Committee Member, including IEEE BIBM 2020, IEEE TrustCom 2020 Workshop, WWW 2021, AAAI 2022 Workshop. Her research interests include graph theory, data center networks, parallel and distributed systems, design and analysis of algorithms, and fault diagnosis.

李晓燕于2019年在中国苏州的苏州大学获得计算机科学博士学位。她曾于2018年6月至2019年6月担任香港城市大学计算机科学系的访问学者。她目前是中国福州大学计算机与数据科学学院的副教授。她已在相关研究期刊和会议上发表了30余篇论文，例如《IEEE并行与分布式系统汇刊》、《IEEE/ACM网络汇刊》、《IEEE计算机汇刊》、《并行与分布式计算杂志》以及《人工智能促进协会》。她曾担任一些会议的会议主席和程序委员会成员，包括2020年IEEE BIBM、2020年IEEE TrustCom研讨会、2021年WWW会议以及2022年AAAI研讨会。她的研究兴趣包括图论、数据中心网络、并行与分布式系统、算法设计与分析以及故障诊断。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_13.jpg?x=105&y=190&w=226&h=279"/>

<!-- Media -->

Jou-Ming Chang received the BS degree in applied mathematics from Chinese Culture University, Taipei, Taiwan, in 1987, the MS degree in information management from National Chiao Tung University, Hsinchu, Taiwan, in 1992, and the PhD degree in computer science and information engineering from National Central University, Zhongli, Taiwan, in 2001. He served as the Dean of the College of Management, National Taipei University of Business (NTUB), Taipei, in 2014. He is currently a Distinguished Professor in the Institute of Information and Decision Sciences, NTUB. He has published more than 150 research papers in refereed journals and conferences, including IEEE Transactions on Parallel and Distributed Systems, IEEE/ACM Transactions on Networking, IEEE Transactions on Computers. His major research interests include algorithm analysis and design, graph theory, and parallel and distributed computing.

张俊铭于1987年在中国台湾台北的中国文化大学获得应用数学学士学位，1992年在中国台湾新竹的国立交通大学获得信息管理硕士学位，2001年在中国台湾中坜的国立中央大学获得计算机科学与信息工程博士学位。他在2014年担任台北商业大学（NTUB）管理学院的院长。他目前是NTUB信息与决策科学研究所的杰出教授。他已在同行评审的期刊和会议上发表了150余篇研究论文，包括《IEEE并行与分布式系统汇刊》、《IEEE/ACM网络汇刊》、《IEEE计算机汇刊》。他的主要研究兴趣包括算法分析与设计、图论以及并行与分布式计算。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_13.jpg?x=878&y=191&w=224&h=277"/>

<!-- Media -->

Dajin Wang received the BEng degree in computer engineering from Shanghai University of Science and Technology, Shanghai, China, in 1982, and the PhD degree in computer science from the Stevens Institute of Technology, Hoboken, USA, in 1990. He is currently a professor in the Department of Computer Science with Montclair State University. He received several university level awards for his scholarly accomplishments. He has held visiting positions in other universities, and has consulted in industry. He served as an associate editor of IEEE Transactions on Parallel and Distributed Systems from 2010 to 2014. He has published over one hundred papers in these areas. Many of his works appeared in premier journals including IEEE Transactions on Computers, IEEE Transactions on Parallel and Distributed Systems, IEEE Transactions on Systems, Man and Cybernetics, IEEE Transactions on Relibility, Journal of Parallel and Distributed Computing, and Parallel Computing. He has served on the program committees of influential conferences. His main research interests include interconnection networks, fault tolerant computing, algorithmic robotics, parallel processing, and wireless ad hoc and sensor networks.

王大金获得了上海科技大学计算机工程专业的工学学士学位（1982年，中国上海），以及史蒂文斯理工学院（美国霍博肯）计算机科学专业的博士学位（1990年）。他目前是蒙特克莱尔州立大学计算机科学系的教授。由于他在学术上的成就，获得了几所大学的奖项。他在其他大学担任过访问学者，并在工业界提供咨询。他从2010年至2014年担任IEEE Transactions on Parallel and Distributed Systems的副编辑。他在这些领域发表了超过一百篇论文。他的许多作品发表在顶级期刊上，包括IEEE Transactions on Computers、IEEE Transactions on Parallel and Distributed Systems、IEEE Transactions on Systems, Man and Cybernetics、IEEE Transactions on Relibility、《Journal of Parallel and Distributed Computing》以及《Parallel Computing》。他曾担任有影响力会议的程序委员会成员。他的主要研究领域包括互联网络、容错计算、算法机器人学、并行处理以及无线自组织和传感器网络。