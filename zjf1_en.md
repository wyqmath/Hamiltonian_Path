# An Efficient Algorithm for Hamiltonian Path Embedding of \( k \) -Ary \( n \) -Cubes Under the Partitioned Edge Fault Model

Hongbin Zhuang ©, Xiao-Yan Li D, Jou-Ming Chang (C), and Dajin Wang (C)

Abstract-The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is one of the most important interconnection networks for building network-on-chips, data center networks, and parallel computing systems owing to its desirable properties. Since edge faults grow rapidly and the path structure plays a vital role in large-scale networks for parallel computing, fault-tolerant path embedding and its related problems have attracted extensive attention in the literature. However, the existing path embedding approaches usually only focus on the theoretical proofs and produce an \( n \) -related linear fault tolerance since they are based on the traditional fault model, which allows all faults to be adjacent to the same node. In this paper, we design an efficient fault-tolerant Hamiltonian path embedding algorithm for enhancing the fault-tolerant capacity of \( k \) -ary \( n \) -cubes. To facilitate the algorithm, we first introduce a new conditional fault model, named Partitioned Edge Fault model (PEF model). Based on this model,for the \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) with \( n \geq  2 \) and odd \( k \geq  3 \) ,we explore the existence of a Hamiltonian path in \( {Q}_{n}^{k} \) with large-scale edge faults. Then we give an \( O\left( N\right) \) algorithm,named HP-PEF, to embed the Hamiltonian path into \( {Q}_{n}^{k} \) under the PEF model, where \( N \) is the number of nodes in \( {Q}_{n}^{k} \) . The performance analysis of HP-PEF shows the average path length of adjacent node pairs in the Hamiltonian path constructed by HP-PEF. We also make comparisons to show that our result of edge fault tolerance has exponentially improved other known results. We further experimentally show that HP-PEF can support the dynamic degradation of average success rate of constructing Hamiltonian paths when increasing faulty edges exceed the fault tolerance.

Index Terms- \( k \) -ary \( n \) -cubes,algorithm,fault-tolerant embedding, Hamiltonian path, interconnection networks.

## NOMENCLATURE

NoC Network-on-Chip

TSV Through silicon via

IC Integrated circuit

VLSI Very large scale integration

TRC Tours routing chip

DCN Data center network

H-path Hamiltonian path

PEF Partitioned edge fault

\( {Q}_{n}^{k}\;k \) -Ary \( n \) -cube

APL Average path length

SD Standard deviation

FT Fault tolerance of \( {Q}_{n}^{k} \) when embedding the H-path under the traditional model

FP Fault tolerance of \( {Q}_{n}^{k} \) when embedding the H-path under the PEF model

ASR Average success rate

## I. INTRODUCTION

NETWORK-ON-CHIPS (NoCs) have emerged as a promis- ing fabric for supercomputers due to their reusability and scalability [1], [2]. This fabric allows a chip to include a large number of computing nodes and effectively turn it into a tiny supercomputer. Therefore, it alleviates the bottlenecks faced by the further development of supercomputers. With the rapidly increasing demand for computing capacity, the number of on-chip cores increases quickly, which results in a high average internode distance in two-dimensional NoCs (2D NoCs). Consequently, 2D NoCs exhibit high communication delay and power consumption as the scale of networks increases [3]. Hence, 3D \( \mathrm{{NoCs}} \) have been designed to solve the scalability problem of 2D NoCs. In 3D NoCs, the so-called through silicon via (TSV) links are used to connect various planes or layers. Though the fabrication cost of TSV links is quite high, 3D NoCs can reduce the probability of long-distance communication while still maintaining high integration density.

However, since the additional expense is required for incorporating more processing nodes, NoCs demand a robust fault tolerance. For example, 3D integrated circuit (IC) fabrication technology improves the power density of modern chips, which results in a thermal-intensive environment for 3D NoCs. High core temperatures reduce chip lifetime and mean time to failure, as well as resulting in low reliability and high cooling costs. The faults in NoCs are mainly divided into two categories, namely, transient faults and permanent faults. Generally speaking, permanent fault deserves more attention [4] since it seriously affects the transmission of more packets. With the rapid increase in the number of processing nodes, NoCs may encounter many permanent faults problems [5], and more reliability threats accompanied by permanent faults will also appear [6]. Therefore, we mainly discuss the permanent faults in this paper.

---

<!-- Footnote -->

Manuscript received 23 July 2022; revised 4 February 2023; accepted 1 April 2023. Date of publication 5 April 2023; date of current version 8 May 2023. This work was supported by the National Natural Science Foundation of China under Grant 62002062 (X.-Y. Li), in part by the Ministry of Science and Technology of Taiwan under Grant MOST-111-2221-E-141-006 (J.-M. Chang), and in part by the Natural Science Foundation of Fujian Province under Grant 2022J05029 (X.-Y. Li). Recommended for acceptance by D. Yang. (Corresponding author: Xiao-Yan Li.)

Hongbin Zhuang and Xiao-Yan Li are with the College of Computer and Data Science, Fuzhou University, Fuzhou 350108, China (e-mail: hbzhuang476@gmail.com; xyli@fzu.edu.cn).

Jou-Ming Chang is with the Institute of Information and Decision Sciences, National Taipei University of Business, Taipei 10051, Taiwan (e-mail: spade@ntub.edu.tw).

Dajin Wang is with the Department of Computer Science, Montclair State University, Montclair, NJ 07043 USA (e-mail: wangd@montclair.edu).

Digital Object Identifier 10.1109/TPDS.2023.3264698

<!-- Footnote -->

---

A well-designed fault-tolerant routing algorithm can address the fault tolerance challenges of NoCs by bypassing the faults when delivering packets. Routing algorithms should be not only fault-tolerant but also deadlock-free. The Hamiltonian path (H-path for short) strategy is a powerful tool for deadlock avoidance. Since the H-path traverses every node in the network exactly once and contains no cycle structure, the deadlock can be easily prevented by transmitting the packets along the H-path [7], [8], [9], [10], [11], [12], [13]. In recent years, this excellent strategy is utilized for designing fault-tolerant routing algorithms. For instance, the HamFA algorithm [7] is one of the most famous fault-tolerant routing algorithms using the H-path strategy. It constructs two directed subnetworks through the \( \mathrm{H} \) -path strategy and limits packets to be routed in a single subnetwork so that the deadlock can be avoided. Simultaneously, HamFA can tolerate almost all one-faulty links. The FHOE algorithm [8] is also a fault-tolerant routing algorithm based on the H-path strategy for 2D NoCs. It fully combines the advantages of traditional odd-even turn model and HamFA strategy, and consequently can provide higher adaptivity and more choices of minimal paths compared to HamFA. Considering the importance of fault tolerance and extensive applications of the H-path in NoCs, it's natural to investigate the existence of the H-path in NoCs (i.e., the problem of embedding the H-path into NoCs), especially when faults occur (i.e., the fault-tolerant problem of embedding the H-path into NoCs). However, it is well-known that the problem of embedding an \( \mathrm{H} \) -path into a network is NP-complete, even when no fault exists.

NoCs usually take interconnection networks as their underlying topology, which inherently affects the performance of NoCs. The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is one of the most important interconnection networks for building NoCs owing to its desirable properties, such as regularity, recursive structure, node symmetry, edge symmetry, low-latency, and ease of implementation [14]. The two associated parameters \( k \) and \( n \) in \( {Q}_{n}^{k} \) provide it the ability to satisfy structural needs in a variety of circumstances. A commercial VLSI chip named the Tours Routing Chip (TRC) was designed early to perform wormhole routing in an arbitrary \( k \) -ary \( n \) -cube [15]. Furthermore,many fault-tolerant deadlock-free routing algorithms have been developed in \( {Q}_{n}^{k} \) -based NoCs [16], [17],[18]. The desirable properties of \( {Q}_{n}^{k} \) have even attracted a lot of research actually to build data center networks (DCNs), such as CamCube [19], NovaCube [20], CLOT [21], and Wave-Cube [22]. Though the scale of DCNs is much larger than that of NoCs, \( k \) -ary \( n \) -cubes can easily cope with it. It’s worth pointing out that stronger fault tolerance is necessary for the DCN since it possesses a lot of servers. Moreover, a lot of well-known parallel computing systems like iWarp [23], J-machine [24], Cray T3D [25], Cray T3E [26], and IBM Blue Gene/L [27] all have adopted \( k \) -ary \( n \) -cubes as their underlying topologies. These \( {Q}_{n}^{k} \) - based architectures usually have high bisection width, high path diversity, high scalability, and affordable implementation cost.

In order to apply the attractive H-path structure in \( {Q}_{n}^{k} \) with as many faults as possible, the fault-tolerant problem of embedding the H-path into \( {Q}_{n}^{k} \) has been extensively investigated in [28], [29],[30],[31],[32],[33],[34]. A network \( G \) is Hamiltonian-connected if an H-path exists between any two nodes in \( G \) . Also, \( G \) is \( f \) -edge fault-tolerant Hamiltonian-connected provided it is Hamiltonian-connected after removing arbitrary \( f \) edges in \( G \) . Yang et al. [32] proved that for any odd integer \( k \geq  3,{Q}_{n}^{k} \) is(2n - 3)-edge fault-tolerant Hamiltonian-connected. Stewart and Xiang [33] proved that for any even integer \( k \geq  4 \) ,there is an \( \mathrm{H} \) -path between any two nodes in different partite sets in \( {Q}_{n}^{k} \) with at most \( {2n} - 2 \) faulty edges. Yang and Zhang [34] recently showed that for every odd integer \( k,{Q}_{n}^{k} \) admits an \( \mathrm{H} \) -path between any two nodes that avoids a set \( F \) of faulty edges and passes through a set \( L \) of prescribed linear forests when \( \left| {E\left( L\right) }\right|  + \left| F\right|  \leq  {2n} - 3 \) . All the above results are obtained under the traditional fault model, which doesn't exert any restriction on the distribution of faulty edges. However, Yuan et al. [35] and \( \mathrm{{Xu}} \) et al. [36] respectively pointed out that this model has many flaws in the realistic situation since it ignores the fact that it's almost impossible for all faulty nodes (resp. faulty edges) to be adjacent to the same node simultaneously (unless that the node fails). In other words, the fault tolerance assessment approaches under the traditional fault model seriously underestimate the fault tolerance potential of \( {Q}_{n}^{k} \) .

The conditional fault model was proposed for tolerating more faulty edges by restricting each node to be adjacent to at least two fault-free edges. Under this model, Wang et al. [29] proved that for any even integer \( k \geq  4 \) ,there is an \( \mathrm{H} \) -path between any two nodes in different partite sets in \( {Q}_{n}^{k} \) with at most \( {4n} - 5 \) conditional faulty edges. Though the fault tolerance they obtained is about twice that under the traditional fault model, it remains linearly correlated with \( n \) . In addition,all the literature mentioned above only provides theoretical proofs about the existence of the \( \mathrm{H} \) -path in \( {Q}_{n}^{k} \) ,while executable fault-tolerant \( \mathrm{H} \) -path embedding algorithms and their performance analysis are missing. Thus, this may hinder the practical application of the \( \mathrm{H} \) -path on \( {Q}_{n}^{k} \) .

In this paper, we pay more attention to the distribution pattern of faulty edges in each dimension of \( {Q}_{n}^{k} \) . This consideration is based on the fact that various dimensions of \( {Q}_{n}^{k} \) usually possess different faulty features in practical fields. For example, to minimize the fabrication cost of TSV links, \( {Q}_{n}^{k} \) -based 3D \( \mathrm{{NoCs}} \) are often designed with only partial connection in the vertical dimension (i.e., partial TSVs) [37], [38]. Particularly, the TSV density of 3D NoCs was suggested for only 12.5% in [38]. It implies that only 12.5% of vertical links are available, and 87.5% of vertical links can be deemed faulty. In other words, many missed links exist in one dimension inherently when \( {Q}_{n}^{k} \) is utilized for building 3D NoCs. In this case, we can deem the vertically partially connected NoC topology as a \( {Q}_{n}^{k} \) with many faulty links concentrated at the same dimension.

Based on the above concerns, for a class of networks that exhibit different faulty features in each dimension, we introduce another fault model, named the partitioned edge fault model (PEF model for short), to help such networks achieve a better fault-tolerant capacity. In essence, this model imposes different restrictions on faulty edges in each dimension according to flawed features. In fact, these restrictions are similar to the concept recently proposed by Zhang et al. [39], which pointed out that restricting the number of faulty edges in each dimension is quite important for reflecting the actual fault-tolerant capacity of a network and can be utilized to improve network edge connectivity. Thus, we utilize the PEF model to explore the fault tolerance potential of \( {Q}_{n}^{k} \) when embedding an H-path into \( {Q}_{n}^{k} \) with \( n \geq  2 \) and odd \( k \geq  3 \) and evaluate the performance of our approach. Our contributions are presented as follows:

1) We propose a new fault model, the PEF model, to improve the fault tolerance of \( {Q}_{n}^{k} \) when we embed an H-path into the faulty \( {Q}_{n}^{k} \) .

2) Under the PEF model, we provide a theoretical analysis for proving the existence of the \( \mathrm{H} \) -path in \( {Q}_{n}^{k} - F \) ,where \( F \) is a PEF set (defined in Section III) such that \( \left| F\right|  \leq \) \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \)

3) Based on the obtained theoretical results, we design an \( O\left( N\right) \) algorithm,named \( {HP} - {PEF} \) ,for embedding the \( \mathrm{H} \) - path into \( {Q}_{n}^{k} \) under the PEF model,where \( N \) is the number of nodes in \( {Q}_{n}^{k} \) . To our knowledge,this is the first time that an algorithm is not only proposed and proved correct, but also actually implemented, for H-path embedding into an edge-faulty \( {Q}_{n}^{k} \) .

The implementation of the algorithm afforded us the ability to observe some features of the generated \( \mathrm{H} \) -paths. For example,if an edge connecting nodes \( u \) and \( v \) became faulty,then the path length of \( u \) and \( v \) in the generated \( \mathrm{H} \) -path can be an indicator of how important the missed edge is. By experimenting with the algorithm, we gather the data of average path lengths for all edges in the generated \( \mathrm{H} \) -path.

Our algorithm is shown to outperform all known similar works in terms of tolerated faulty-edges. In particular, compared to [32],the improvement is from linear(2n - 3) to exponential \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) . We also show that HP-PEF can support the dynamic degradation of average success rate of constructing required \( \mathrm{H} \) -paths even when increasing faulty edges exceed \( \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) .

Organization: The rest of this paper is organized as follows. In Section II, we provide the preliminaries used throughout this paper. In Section III, we first present the definition of the PEF model, and then give the theoretical proof related to the existence of the H-path in \( {Q}_{n}^{k} \) under the PEF model. In Section IV,we design the fault-tolerant H-path embedding algorithm HP-PEF based on the theoretical basis in Section III and offer a detailed example for illuminating the execution process of HP-PEF. In Section V, we evaluate the performance of our method by implementing computer programs. Section VI concludes this paper.

## II. Preliminaries

## A. Terminologies and Notations

For terminologies and notations not defined in this subsection, please refer to the reference [40]. An interconnection network can be modeled as a graph \( G = \left( {V\left( G\right) ,E\left( G\right) }\right) \) ,where \( V\left( G\right) \) represents its node set and \( E\left( G\right) \) represents its edge set. The notations \( \left| {V\left( G\right) }\right| \) and \( \left| {E\left( G\right) }\right| \) denote the size of \( V\left( G\right) \) and \( E\left( G\right) \) , respectively. Given a graph \( S \) ,if \( V\left( S\right)  \subseteq  V\left( G\right) \) and \( E\left( S\right)  \subseteq \) \( E\left( G\right) \) ,then \( S \) is a subgraph of \( G \) . Given a node set \( M \subseteq  V\left( G\right) \) , the subgraph of \( G \) induced by \( M \) ,denoted by \( G\left\lbrack  M\right\rbrack \) ,is a graph with the node set \( M \) and edge set \( \{ \left( {u,v}\right)  \in  E\left( G\right)  \mid  u,v \in  M\} \) . Let \( F \) be a faulty edge set of \( G \) ,and \( G - F \) be the graph with the node set \( V\left( G\right) \) and the edge set \( E\left( G\right)  - F \) . Given a positive integer \( n \) ,we denote the set \( \{ 1,2,\ldots ,n\} \) as \( \left\lbrack  n\right\rbrack \) . Moreover,let \( {\mathbb{Z}}_{n} = \left\lbrack  {n - 1}\right\rbrack   \cup  \{ 0\} \) when \( n \geq  2 \) and \( {\mathbb{Z}}_{1} = \{ 0\} \) . A graph \( P = \) \( \left( {{v}_{0},{v}_{1},\ldots ,{v}_{p}}\right) \) is called a path if \( p + 1 \) nodes \( {v}_{0},{v}_{1},\ldots ,{v}_{p} \) are distinct and \( \left( {{v}_{i},{v}_{i + 1}}\right) \) is an edge of \( P \) with \( i \in  {\mathbb{Z}}_{p} \) . The length of \( P \) is the number of the edges in \( P \) . If \( V\left( P\right)  = V\left( G\right) \) ,then \( P \) is a Hamiltonian path (H-path for short) of \( G \) .

## B. \( k \) -Ary \( n \) -Cube \( {Q}_{n}^{k} \)

Definition II.1. (See [33]). The \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) is a graph with the node set \( V\left( {Q}_{n}^{k}\right)  = \{ 0,1,\ldots ,k - 1{\} }^{n} \) such that two nodes \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{0} \) and \( v = {v}_{n - 1}{v}_{n - 2}\cdots {v}_{0} \) are adjacent in \( {Q}_{n}^{k} \) if and only if there is an integer \( i \in  {\mathbb{Z}}_{n} \) satisfying \( {u}_{i} = {v}_{i} \pm  1\left( {\;\operatorname{mod}\;k}\right) \) and \( {u}_{j} = {v}_{j} \) for every \( j \in  {\mathbb{Z}}_{n} - \{ i\} \) . In this case,such an edge(u,v)is called an \( i \) -dimensional edge for \( i \in  {\mathbb{Z}}_{n} \) ,and the set of all \( i \) -dimensional edges of \( {Q}_{n}^{k} \) is denoted by \( {E}_{i}\left( {Q}_{n}^{k}\right) \) ,or \( {E}_{i} \) for short.

Hereafter,for brevity,we will omit "(mod \( k \) )" in a situation similar to the above definition. By Definition II.1, \( {Q}_{n}^{k} \) is \( {2n} \) -regular and contains \( {k}^{n} \) nodes. In addition, \( {Q}_{n}^{k} \) is edge symmetric [41]. The \( {Q}_{n}^{k} \) can be partitioned into \( k \) disjoint subgraphs \( {Q}_{n}^{k}\left\lbrack  0\right\rbrack  ,{Q}_{n}^{k}\left\lbrack  1\right\rbrack  ,\ldots ,{Q}_{n}^{k}\left\lbrack  {k - 1}\right\rbrack \) (abbreviated as \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack  ) \) along the \( i \) -dimension for \( i \in \) \( {\mathbb{Z}}_{n} \) . All these \( k \) subgraphs are isomorphic to \( {Q}_{n - 1}^{k} \) . Given a faulty edge set \( F \) ,let \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack   = \{ \left( {u,v}\right)  \mid  u \in  V\left( {Q\left\lbrack  l\right\rbrack  }\right) ,v \in \) \( \left. {V\left( {Q\left\lbrack  {l + 1}\right\rbrack  }\right) \text{and}\left( {u,v}\right)  \in  F \cap  {E}_{i}}\right\} \) . If there is no ambiguity, we abbreviate \( {F}_{i}\left\lbrack  {l,l + 1}\right\rbrack \) to \( F\left\lbrack  {l,l + 1}\right\rbrack \) . Each node of \( Q\left\lbrack  l\right\rbrack \) has the form \( u = {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}l{u}_{i - 1}\cdots {u}_{0} \) . The node \( v = \) \( {u}_{n - 1}{u}_{n - 2}\cdots {u}_{i + 1}{l}^{\prime }{u}_{i - 1}\cdots {u}_{0} \) is the neighbor of \( u \) in \( Q\left\lbrack  {l}^{\prime }\right\rbrack \) if \( {l}^{\prime } = l \pm  1 \) ,which is denoted by \( {n}^{{l}^{\prime }}\left( u\right) \) . To distinguish the positions of the subgraphs where different nodes are located, let \( {l}_{u} = {u}_{i} = l \) . That is, \( {l}_{v} = {l}_{u} \pm  1 \) . Although the values of \( {l}_{u} \) and \( {u}_{i} \) are equal,the notation \( {l}_{u} \) mainly focuses on the position \( l \) rather than the dimension \( i \) of node \( u \) . Let \( {Q}_{n}^{k}\left\lbrack  {\ell ,h}\right\rbrack \) (abbreviated as \( Q\left\lbrack  {\ell ,h}\right\rbrack \) ) be the subgraph induced by node set \( \{ u \mid  u \in \) \( V\left( {Q\left\lbrack  j\right\rbrack  }\right) \) with \( j = \ell ,\ell  + 1,\ldots ,h - 1,h\} \) . Moreover,we have \( Q\left\lbrack  {\ell ,h}\right\rbrack   = Q\left\lbrack  \ell \right\rbrack \) when \( \ell  = h \) ,and taken modulo \( k \) ,we have \( Q\left\lbrack  {\ell ,h}\right\rbrack   = {Q}_{n}^{k} \) when \( h = \ell  - 1 \) . A path in \( Q\left\lbrack  {\ell ,h}\right\rbrack \) is denoted by \( {P}_{\ell ,h} \) with \( V\left( {P}_{\ell ,h}\right)  \subseteq  V\left( {Q\left\lbrack  {\ell ,h}\right\rbrack  }\right) \) . For convenience,we abbreviate \( {P}_{\ell ,h} \) to \( {P}_{\ell } \) when \( \ell  = h \) ,and to \( P \) when \( h = \ell  - 1 \) by taking modulo \( k \) .

Fig. 1 shows \( {Q}_{1}^{3},{Q}_{2}^{3} \) ,and \( {Q}_{3}^{3} \) . We color each edge according to its dimension. In Fig. 1(c),we partition \( {Q}_{3}^{3} \) into 3 disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ,and \( Q\left\lbrack  2\right\rbrack \) along the 2-dimension. The subgraph \( Q\left\lbrack  {0,1}\right\rbrack \) is induced by node set \( \{ u \mid  u \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right)  \cup \) \( V\left( {Q\left\lbrack  1\right\rbrack  }\right) \} \) . The node \( u = {110} \in  V\left( {Q\left\lbrack  1\right\rbrack  }\right) \) is adjacent to two nodes \( v = {010} \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) and \( w = {210} \in  V\left( {Q\left\lbrack  2\right\rbrack  }\right) \) . In addition, \( v = {n}^{0}\left( u\right) \) and \( w = {n}^{2}\left( u\right) \) . It’s easy to see that \( {l}_{v} = \) \( {l}_{u} - 1 = 0 \) and \( {l}_{w} = {l}_{u} + 1 = 2 \) .

In particular,the \( {Q}_{2}^{k} \) can be deemed a \( k \times  k \) grid with wraparound edges,where a node \( {u}_{i,j} = {ij} \) is indexed by its row \( i \) and column \( j \) . Let \( p,q \in  {\mathbb{Z}}_{k} \) be two row indices with \( p \neq  q \) . If \( p < q \) ,we define the row torus \( {rt}\left( {p,q}\right) \) to be the subgraph of \( {Q}_{2}^{k} \) induced by the nodes on rows \( p,p + 1,\ldots ,q \) ,and particularly, all column edges between nodes on row \( p \) and row \( q \) are removed when \( p = 0 \) and \( q = k - 1 \) . Otherwise,if \( p > q \) ,we define the row torus \( {rt}\left( {p,q}\right) \) to be the subgraph of \( {Q}_{2}^{k} \) induced by the nodes on rows \( p,p + 1,\ldots ,k - 1,0,\ldots ,q \) ,and particularly,all column edges between nodes on row \( p \) and row \( q \) are removed when \( p = q + 1 \) . Fig. 2 depicts the \( {rt}\left( {0,4}\right) \) ,which is obtained by removing the column edges between nodes on row 0 and row 4 from \( {Q}_{2}^{5} \) . Throughout,we assume that the addition of row or column indices is modulo \( k \) .

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_3.jpg?x=123&y=184&w=1463&h=571"/>

Fig. 1. The structures of (a) \( {Q}_{1}^{3} \) ; (b) \( {Q}_{2}^{3} \) ; (c) \( {Q}_{3}^{3} \) .

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_3.jpg?x=162&y=885&w=635&h=409"/>

Fig. 2. The structure of \( {rt}\left( {0,4}\right) \) .

<!-- Media -->

For the row torus \( {rt}\left( {p,q}\right) \) with \( q - p = 1 \) ,we define the four types of paths in \( {rt}\left( {p,q}\right) \) as follows. The notations of these paths are derived from the shape of their pictorial representations, where \( \bar{i} = q + p - i \) .

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

## III. THEORETICAL BASIS FOR EMBEDDING THE HAMILTONIAN PATH INTO \( k \) -ARY \( n \) -CUBES

In this section, we establish the theoretical basis for embedding the H-path into \( k \) -ary \( n \) -cubes under the PEF model. That is,we will prove that an \( \mathrm{H} \) -path can be found in a \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) in the presence of a partitioned edge fault set,described below.

Let \( F \) be a faulty edge set in \( {Q}_{n}^{k} \) ,and let \( {F}_{i} = F \cap  {E}_{i} \) with \( i \in  {\mathbb{Z}}_{n} \) . We set \( \left\{  {{e}_{0},{e}_{1},\ldots ,{e}_{n - 1}}\right\}   = \left\{  {\left| {F}_{i}\right|  \mid  i \in  {\mathbb{Z}}_{n}}\right\} \) such that \( {e}_{n - 1} \geq  {e}_{n - 2} \geq  \cdots  \geq  {e}_{0} \) . The faulty edge set \( F \) is a partitioned edge fault set (PEF set for short) if and only if \( {e}_{i} \leq  f\left( i\right)  < \) \( \frac{\left| E\left( {Q}_{n}^{k}\right) \right| }{n} = {k}^{n} \) for each \( i \in  {\mathbb{Z}}_{n} \) ,where \( f\left( i\right) \) is a function of \( i \) or a fixed value.

For a PEF set \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) ,since \( {Q}_{n}^{k} \) is recursively constructed with edge symmetry, we can utilize the inductive method to analyze the Hamiltonian property of \( {Q}_{n}^{k} \) by partitioning it into \( k \) subgraphs along a dimension we expected. Consequently,provided that \( {Q}_{n}^{k} - F \) is Hamiltonian-connected, the exact values of \( f\left( i\right) \) can be derived by analyzing the number of faulty edges that can be tolerated when the H-path passes through two consecutive subgraphs.

The following result is provided for dealing with the base case of our forthcoming inductive proof.

Lemma III.1. (See [42]) For odd \( k \geq  3 \) ,let \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) with \( \left| F\right|  \leq  1 \) . Then \( {Q}_{2}^{k} - F \) is Hamiltonian-connected.

Theorem III.2. For \( n \geq  2 \) and odd \( k \geq  3 \) ,let \( F \subseteq  E\left( {Q}_{n}^{k}\right) \) be a PEF set satisfying the following conditions:

1) \( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) ;

2) \( {e}_{i} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ;

3) \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) .

Then, \( {Q}_{n}^{k} - F \) is Hamiltonian-connected.

Proof. The proof is by induction on \( n \) . When \( n = 2 \) ,by Lemma III.1,the theorem holds obviously. For \( n \geq  3 \) ,assume this theorem holds for all \( {Q}_{m}^{k} \) ’s with \( m < n \) . Therefore,what we need to prove is that this theorem holds for \( {Q}_{n}^{k} \) .

Since \( {Q}_{n}^{k} \) is edge symmetric,without loss of generality, let \( \left| {F}_{n - 1}\right|  = \max \left\{  {\left| {F}_{n - 1}\right| ,\left| {F}_{n - 2}\right| ,\ldots ,\left| {F}_{0}\right| }\right\} \) . That is, \( \left| {F}_{n - 1}\right|  = \) \( {e}_{n - 1} \leq  {k}^{n - 1} - 2 \) . Along the(n - 1)-dimension,we divide \( {Q}_{n}^{k} \) into \( k \) disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack  ,\ldots ,Q\left\lbrack  {k - 1}\right\rbrack \) ,all of which are isomorphic to \( {Q}_{n - 1}^{k} \) . Let \( s \) and \( t \) be arbitrary two vertices of \( {Q}_{n}^{k} \) with \( s \in  V\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) and \( t \in  V\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) . By the arbitrariness of \( s \) and \( t \) ,suppose that \( {l}_{s} \leq  {l}_{t} \) .

Let \( {C}_{h}^{l} = E\left( {Q\left\lbrack  l\right\rbrack  }\right)  \cap  {F}_{h} \) with \( l \in  {\mathbb{Z}}_{k} \) and \( h \in  {\mathbb{Z}}_{n - 1} \) . Moreover,for each \( l \in  {\mathbb{Z}}_{k} \) ,let \( \left\{  {{e}_{n - 2}^{l},{e}_{n - 3}^{l},\ldots ,{e}_{0}^{l}}\right\}   = \) \( \left\{  {\left| {C}_{n - 2}^{l}\right| ,\left| {C}_{n - 3}^{l}\right| ,\ldots ,\left| {C}_{0}^{l}\right| }\right\} \) such that \( {e}_{n - 2}^{l} \geq  {e}_{n - 3}^{l} \geq  \cdots  \geq  {e}_{0}^{l} \) . According to the recursive nature and conditions 2) and 3), when \( n = 3 \) ,we have \( \left| F\right|  - \left| {F}_{n - 1}\right|  = \mathop{\sum }\limits_{{i = 0}}^{1}\left| {F}_{i}\right|  \leq  1,{e}_{0}^{l} = 0 \) , and \( {e}_{1}^{l} \leq  1 \) . When \( n \geq  4 \) ,we have

\[\left| F\right|  - \left| {F}_{n - 1}\right|  = \mathop{\sum }\limits_{{i = 0}}^{{n - 2}}\left| {F}_{i}\right|  = \mathop{\sum }\limits_{{i = 2}}^{{n - 2}}{e}_{i} + \mathop{\sum }\limits_{{i = 0}}^{1}{e}_{i}\]

\[ \leq  \mathop{\sum }\limits_{{i = 2}}^{{n - 2}}\left( {{k}^{i} - 2}\right)  + 1\]

\[ = \frac{{k}^{n - 1} - {k}^{2}}{k - 1} - 2\left( {n - 1}\right)  + 5.\]

In addition, \( {e}_{0}^{l} = 0,{e}_{1}^{l} \leq  1 \) ,and \( {e}_{i}^{l} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n - 1} - {\mathbb{Z}}_{2} \) . Therefore,every \( Q\left\lbrack  l\right\rbrack   - F \) with \( l \in  {\mathbb{Z}}_{k} \) is Hamiltonian-connected. That is,when \( {l}_{s} = {l}_{t} \) ,there exists an \( \mathrm{H} \) -path in \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) between \( s \) and \( t \) .

Without loss of generality,suppose that \( \left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right|  = \) \( \max \{ \left| {F\left\lbrack  {0,1}\right\rbrack  }\right| ,\ldots ,\left| {F\left\lbrack  {k - 2,k - 1}\right\rbrack  }\right| ,\left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right| \} .\; \) Since \( \mathop{\sum }\limits_{{l = 0}}^{{k - 1}}\left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  = \left| {F}_{n - 1}\right|  \leq  {k}^{n - 1} - 2 \) and \( k \geq  3 \) is odd, \( \left| {F\left\lbrack  {l,l + 1}\right\rbrack  }\right|  \leq  \left\lfloor  \frac{{k}^{n - 1} - 2}{2}\right\rfloor   = \frac{{k}^{n - 1} - 3}{2} \) for all \( l \in  {\mathbb{Z}}_{k - 1} \) .

Claim 1: Suppose that there exists an \( \mathrm{H} \) -path \( {P}_{q} \) in \( Q\left\lbrack  q\right\rbrack   - F \) between any two distinct nodes in \( Q\left\lbrack  q\right\rbrack \) . When \( 0 \leq  q \leq  k - \)

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_4.jpg?x=980&y=182&w=578&h=347"/>

Fig. 3. The constructions in Case 1.1 of Theorem III.2.

<!-- Media -->

2,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . And when \( 1 \leq  q \leq  k - \) 1,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) .

The length of \( {P}_{q} \) is \( {k}^{n - 1} - 1 \) . Then there exist \( \frac{{k}^{n - 1} - 1}{2} \) mutually disjoint edges on \( {P}_{q} \) . When \( 0 \leq  q \leq  k - 2 \) ,since \( \frac{{k}^{n - 1} - 1}{2} - \) \( \left| {F\left\lbrack  {q,q + 1}\right\rbrack  }\right|  \geq  1 \) ,there exists at least one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Analogously, when \( 1 \leq  q \leq  k - 1 \) ,we can also find at least one edge \( \left( {x,{x}^{ * }}\right)  \in \) \( E\left( {P}_{q}\right) \) such that \( \left( {x,{n}^{q - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{q - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Then the claim holds.

Next, we discuss the following cases separately.

Case 1: \( {l}_{s} = 0 \) .

Case 1.1: \( {l}_{s} = {l}_{t} \) .

Since \( Q\left\lbrack  0\right\rbrack   - F \) is Hamiltonian-connected,there exists an H-path \( {P}_{0} \) in \( Q\left\lbrack  0\right\rbrack   - F \) between \( s \) and \( t \) . By Claim 1,there exists at least one edge \( \left( {x,{x}^{ * }}\right) \) of \( {P}_{0} \) such that \( \left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similarly,since \( Q\left\lbrack  1\right\rbrack   - F \) is Hamiltonian-connected,there exists an \( \mathrm{H} \) -path \( {P}_{1} \) in \( Q\left\lbrack  1\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( {n}^{1}\left( {x}^{ * }\right) \) . By Claim 1,there exists at least one edge \( \left( {y,{y}^{ * }}\right) \) of \( {P}_{1} \) such that \( \left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{2}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . By constantly iterating in this way, we can obtain an H-path \( {P}_{2,k - 1} \) in \( Q\left\lbrack  {2,k - 1}\right\rbrack   - F \) between \( {n}^{2}\left( y\right) \) and \( {n}^{2}\left( {y}^{ * }\right) \) . Thus, \( {P}_{0} \cup  {P}_{1} \cup  {P}_{2,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {{n}^{1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right. \) , \( \left. {\left( {y,{n}^{2}\left( y\right) }\right) ,\left( {{n}^{2}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required H-path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 3).

Case 1.2: \( {l}_{s} \neq  {l}_{t} \) .

Since \( \;\left| {V\left( {Q\left\lbrack  0\right\rbrack  }\right) }\right|  - \left| {\{ s,t\} }\right|  - \left| {F\left\lbrack  {0,1}\right\rbrack  }\right|  \geq  {k}^{n - 1} - 2 - \) \( \frac{{k}^{n - 1} - 3}{2} > 1 \) with \( n \geq  3 \) and odd \( k \geq  3 \) ,there exists one node \( x \in  V\left( {Q\left\lbrack  0\right\rbrack  }\right) \) such that \( x \neq  s,{n}^{1}\left( x\right)  \neq  t \) ,and \( \left( {x,{n}^{1}\left( x\right) }\right)  \notin  {F}_{n - 1} \) . Since \( Q\left\lbrack  0\right\rbrack   - F \) is Hamiltonian-connected, there exists an \( \mathrm{H} \) -path \( {P}_{0} \) in \( Q\left\lbrack  0\right\rbrack   - F \) between \( s \) and \( x \) . If \( {l}_{t} = 1 \) ,since \( Q\left\lbrack  1\right\rbrack   - F \) is Hamiltonian-connected,there exists an \( \mathrm{H} \) -path \( {P}_{1} \) in \( Q\left\lbrack  1\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( t \) ; otherwise, if \( 2 \leq  {l}_{t} \leq  k - 1 \) ,proceeding iteratively in this manner can construct an \( \mathrm{H} \) -path \( {P}_{1,{l}_{t}} \) in \( Q\left\lbrack  {1,{l}_{t}}\right\rbrack   - F \) between \( {n}^{1}\left( x\right) \) and \( t \) . If \( {l}_{t} = k - 1 \) ,then \( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup  \left\{  \left( {x,{n}^{1}\left( x\right) }\right) \right\} \) forms the required H-path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) ; otherwise,by Claim 1, there exists one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{1,{l}_{t}}\right)  \cap  E\left( {Q\left\lbrack  {l}_{t}\right\rbrack  }\right) \) such that \( \left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similar to Case 1.1, we can construct an H-path \( {P}_{{l}_{t} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{t} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{t} + 1}\left( y\right) \) and \( {n}^{{l}_{t} + 1}\left( {y}^{ * }\right) \) . Therefore, \( {P}_{0} \cup  {P}_{1,{l}_{t}} \cup \) \( {P}_{{l}_{t} + 1,k - 1} \cup  \left\{  {\left( {x,{n}^{1}\left( x\right) }\right) ,\left( {y,{n}^{{l}_{t} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{t} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\}   - \) \( \left\{  \left( {y,{y}^{ * }}\right) \right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 4).

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_5.jpg?x=193&y=183&w=580&h=348"/>

Fig. 4. The constructions in Case 1.2 of Theorem III.2.

<!-- Media -->

Case 2: \( {l}_{s} \geq  1 \) .

When \( {l}_{t} = k - 1 \) ,we can construct the required \( \mathrm{H} \) -path similar to Case 1. Then we discuss the case of \( 1 \leq  {l}_{s} \leq  {l}_{t} \leq  k - 2 \) .

Similar to Case 1,we can construct an H-path \( {P}_{{l}_{s},k - 1} \) in \( Q\left\lbrack  {{l}_{s},k - 1}\right\rbrack   - F \) between \( s \) and \( t \) . If the part of \( {P}_{{l}_{s},k - 1} \) in \( Q\left\lbrack  {l}_{s}\right\rbrack \) is constructed by the method similar to Case 1.1,by the proof of Case 1.1,let \( \left( {w,{w}^{ * }}\right) \) be the edge in \( Q\left\lbrack  {l}_{s}\right\rbrack \) satisfying \( \left\{  {\left( {w,{n}^{{l}_{s} + 1}\left( w\right) }\right) ,\left( {{w}^{ * },{n}^{{l}_{s} + 1}\left( {w}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) . By Claim 1,there exists one edge \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) \) (or \( \;\left( {x,{x}^{ * }}\right)  \in  \left( {E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) }\right)  \cup  \left\{  \left( {w,{w}^{ * }}\right) \right\}  \; \) if \( \left( {w,{w}^{ * }}\right) \) exists) and \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Similar to Case 1.1,we can construct an H-path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( x\right) \) and \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) . If \( \left( {x,{x}^{ * }}\right)  \in  E\left( {P}_{{l}_{s},k - 1}\right)  \cap  E\left( {Q\left\lbrack  {l}_{s}\right\rbrack  }\right) ,\; \) then \( \;{P}_{{l}_{s},k - 1} \cup  {P}_{0,{l}_{s} - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\}   - \left\{  \left( {x,{x}^{ * }}\right) \right\}  \; \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) .

Otherwise,we have \( \left( {x,{x}^{ * }}\right)  = \left( {w,{w}^{ * }}\right) \) and thus \( \left\{  {\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right) }\right\}   \subseteq  E\left( {P}_{{l}_{s},k - 1}\right) \) . In this situation,the \( \mathrm{H} \) -path \( {P}_{{l}_{s},k - 1} \) must be constructed by the manner in Case 1.1. It implies that \( {l}_{s} = {l}_{t} \) and there exists an H-path \( {P}_{{l}_{s}} \) in \( Q\left\lbrack  {l}_{s}\right\rbrack   - F \) between \( s \) and \( t \) ,which passes through the edge \( \left( {x,{x}^{ * }}\right) \) . If \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  = \frac{{k}^{n - 1} - 3}{2} \) ,then

\[\left| {F\left\lbrack  {{l}_{s} - 1,{l}_{s}}\right\rbrack  }\right|  \leq  {k}^{n - 1} - 2 - \left( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right| \right. \]

\[ + \left| {F\left\lbrack  {k - 1,0}\right\rbrack  }\right| )\]

\[ \leq  {k}^{n - 1} - 2 - 2 \times  \frac{{k}^{n - 1} - 3}{2} \leq  1.\]

Since \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s} - 1,{l}_{s}}\right\rbrack  }\right|  > 2 \) with \( n \geq  3 \) and odd \( k \geq \) 3,there exists one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq \) \( \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Otherwise, if \( \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \leq  \frac{{k}^{n - 1} - 5}{2} \) ,since \( \frac{{k}^{n - 1} - 1}{2} - \left| {F\left\lbrack  {{l}_{s},{l}_{s} + 1}\right\rbrack  }\right|  \geq  2 \) , then there exists at least one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . Thus,there exists at least one edge \( \left( {y,{y}^{ * }}\right)  \in  E\left( {P}_{{l}_{s}}\right) \) such that \( \left( {y,{y}^{ * }}\right)  \neq  \left( {x,{x}^{ * }}\right) \) and \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) or \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) .

Note that the four edges \( \left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) }\right) \) , \( \left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{x}^{ * },{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) . If \( \left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {y}^{ * }\right. \) , \( \left. {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ,by Case 1.1,we can construct an H-path \( {P}_{{l}_{s} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} + 1}\left( y\right) \) and \( {n}^{{l}_{s} + 1}\left( {y}^{ * }\right) \) . Similar to Case 1.1,we can construct an H-path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( x\right) \) and \( {n}^{{l}_{s} - 1}\left( {x}^{ * }\right) \) . Then \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {x,{n}^{{l}_{s} - 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) ,\left( {y,{n}^{{l}_{s} + 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) . If \( \left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{y}^{ * },{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) }\right)  \notin  {F}_{n - 1} \) ,by Case 1.1,we can construct an \( \mathrm{H} \) -path \( {P}_{0,{l}_{s} - 1} \) in \( Q\left\lbrack  {0,{l}_{s} - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} - 1}\left( y\right) \) and \( {n}^{{l}_{s} - 1}\left( {y}^{ * }\right) \) . Similar to Case 1.1,we construct an H-path \( {P}_{{l}_{s} + 1,k - 1} \) in \( Q\left\lbrack  {{l}_{s} + 1,k - 1}\right\rbrack   - F \) between \( {n}^{{l}_{s} + 1}\left( x\right) \) and \( {n}^{{l}_{s} + 1}\left( {x}^{ * }\right) \) . Then \( {P}_{{l}_{s}} \cup  {P}_{0,{l}_{s} - 1} \cup  {P}_{{l}_{s} + 1,k - 1} \cup \) \( \left\{  {\left( {y,{n}^{{l}_{s} - 1}\left( y\right) }\right) ,\left( {{n}^{{l}_{s} - 1}\left( {y}^{ * }\right) ,{y}^{ * }}\right) ,\left( {x,{n}^{{l}_{s} + 1}\left( x\right) }\right) ,\left( {{n}^{{l}_{s} + 1}\left( {x}^{ * }\right) ,{x}^{ * }}\right) }\right\} \) \( - \left\{  {\left( {x,{x}^{ * }}\right) ,\left( {y,{y}^{ * }}\right) }\right\} \) forms the required \( \mathrm{H} \) -path between \( s \) and \( t \) in \( {Q}_{n}^{k} - F \) (see Fig. 5).

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_5.jpg?x=959&y=182&w=589&h=354"/>

Fig. 5. The constructions in Case 2 of Theorem III.2.

<!-- Media -->

## IV. FAULT-TOLERANT HAMILTONIAN PATH EMBEDDING ALGORITHM OF \( k \) -ARY \( n \) -CUBES

In this section,we present the fault-tolerant \( \mathrm{H} \) -path embedding algorithm for \( {Q}_{n}^{k} \) under the PEF model.

First,we design Algorithm 1 costing \( O\left( {k}^{2}\right) \) time to construct the \( \mathrm{H} \) -path in \( {Q}_{2}^{k} - F \) according to the theoretical proof in [42], where Procedure HP-rtFree is utilized to find the H-path in a fault-free \( {rt}\left( {p,q}\right) \) . In Algorithm 1,we let \( s = {u}_{a,b} \) and \( t = {u}_{c,d} \) be two arbitrary distinct nodes in \( {rt}\left( {p,q}\right) \) . In addition,without loss of generality,suppose that \( a \leq  c \) . Given an edge fault set \( F \subseteq  E\left( {Q}_{2}^{k}\right) \) with \( \left| F\right|  \leq  1 \) ,by the symmetry of \( {Q}_{2}^{k} \) ,suppose that \( \left( {{u}_{0,0},{u}_{0,1}}\right) \) is the faulty edge if it exists.

Based on Algorithm 1, we design the Algorithm HP-PEF under the PEF model. Note that the Algorithm HP-PEF is essentially based on the theoretical basis in Section III, where Procedures HP-Round and HP-Direct correspond to the constructive approaches of Case 1.1 and Case 1.2 in Theorem III.2, respectively. In addition, the fault tolerance of Algorithm HP-PEF has been determined by Theorem III.2 (i.e., the three conditions shown in Theorem III.2).

Theorem IV.1. For \( n \geq  2 \) and odd \( k \geq  3 \) ,the algorithm HP-PEF can embed an H-path between arbitrary two nodes \( s \) and \( t \) into \( {Q}_{n}^{k} - F \) ,where \( F \) is a PEF set satisfying (1) \( \left| F\right|  \leq  \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \) ; (2) \( {e}_{i} \leq  {k}^{i} - 2 \) for each \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ; (3) \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) .

Proof. We prove this theorem by induction on \( n \) . When \( n = 2 \) , HP-PEF calls Algorithm 1 to embed the required H-path into \( {Q}_{2}^{k} - F \) ,where \( F \) is a PEF set satisfying \( {e}_{0} = 0 \) and \( {e}_{1} \leq  1 \) . Assume this theorem holds for all \( {Q}_{m}^{k} \) ’s with \( m < n \) . Then we need to prove that this theorem holds for \( {Q}_{n}^{k} \) .

<!-- Media -->

Algorithm 1: Embed an H-path \( P \) into \( {Q}_{2}^{k} - F \) .

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

In Lines 11-20, HP-PEF embeds the required H-path into \( {Q}_{n}^{k} - F \) by discussing three situations similar to Theorem III.2. In the constructive process,we let \( l \) denote the destination subgraph,and \( d \) denote the constructive direction. More specifically, \( d = 1 \) (resp. \( d =  - 1 \) ) means the constructive process will proceed along the clockwise (resp. counterclockwise). Theorem III. 2 provides a detailed proof related to the correctness of Lines 11-20 when \( {i}^{\prime } = n - 1,{l}^{\prime } = k - 1,a = s \) ,and \( b = t \) . Though \( {l}^{\prime },a \) ,and \( b \) may change in each iteration,their relative positions remain unchanged, and the construction process is similar in different cases. Thus, the theorem holds.

<!-- Media -->

Procedure: \( \mathrm{{HP}} - \mathrm{{rtFree}}\left( {{rt}\left( {p,q}\right) ,s,t}\right) \) .

begin

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

Algorithm 2: Embed an H-path \( P \) into \( {Q}_{n}^{k} - F \) where \( F \) is a PEF set (HP-PEF).

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

Theorem IV.2. The time complexity of HP-PEF is \( O\left( N\right) \) , where \( N = {k}^{n} \) .

Proof. When \( n = 2 \) ,HP-PEF calls Algorithm 1,which costs \( O\left( {k}^{2}\right) \) time. Next,we discuss the case of \( n \geq  3 \) in the following.

Since HP-PEF calls HP-Round and HP-Direct frequently, we first analyse the time complexity of HP-Round and HP-Direct.

<!-- Media -->

Procedure: \( \operatorname{HP-Direct}\left( {l,d,s,t}\right) \) .

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

Line 4 costs \( O\left( n\right) \) time. Line 5 classifies all \( {k}^{n} \) nodes of \( {Q}_{n}^{k} \) ,costing \( O\left( {k}^{n}\right) \) time. Line 6 costs \( O\left( k\right) \) time. Lines 7-10 cost \( O\left( 1\right) \) time. Lines 11-13 cost \( O\left( {k}^{n}\right) \) time. Lines 14-15 cost \( O\left( {k}^{n}\right) \) time. Line 17 costs \( O\left( {k}^{n}\right) \) time. Line 18 costs \( O\left( {k}^{n - 1}\right) \) time since there exist \( \frac{{k}^{n - 1} - 1}{2} \) mutually disjoint edges of \( P \) in each subgraph. Line 19 costs \( O\left( {k}^{n}\right) \) time. Line 20 costs \( O\left( 1\right) \) time.

Therefore,HP-PEF needs \( O\left( N\right) \) time.

In the following, we give a detailed example to explain how Algorithm HP-PEF embeds the H-path into \( {Q}_{n}^{k} \) under the PEF model. In the example,we set \( n = k = 3,s = {001} \) , and \( t = {210} \) . Moreover,let \( {F}_{2} = \{ \left( {{100},{200}}\right) \} ,\;{F}_{1} = \) \( \{ \left( {{001},{011}}\right) ,\left( {{010},{020}}\right) ,\left( {{210},{220}}\right) ,\left( {{200},{220}}\right) ,\left( {{201},{221}}\right) , \) \( \left( {{102},{122}}\right) ,\left( {{002},{022}}\right) \} ,\;{F}_{0} = \varnothing \) . Obviously,we have \( {e}_{2} = \left| {F}_{1}\right|  = 7,\;{e}_{1} = \left| {F}_{2}\right|  = 1 \) ,and \( {e}_{0} = \left| {F}_{0}\right|  = 0 \) . Since \( \left| {F}_{1}\right|  > \left| {F}_{0}\right| ,\left| {F}_{2}\right| \) ,the algorithm HP-PEF divides \( {Q}_{3}^{3} \) into 3 disjoint subgraphs \( Q\left\lbrack  0\right\rbrack  ,Q\left\lbrack  1\right\rbrack \) ,and \( Q\left\lbrack  2\right\rbrack \) along the 1-dimension. Since \( \left| {{F}_{1}\left\lbrack  {2,0}\right\rbrack  }\right|  > \left| {{F}_{1}\left\lbrack  {0,1}\right\rbrack  }\right| ,\left| {{F}_{1}\left\lbrack  {1,2}\right\rbrack  }\right| \) ,we have \( {l}^{\prime } = 2 \) ,which implies that the subsequent constructive process doesn't involve the edges between \( Q\left\lbrack  2\right\rbrack \) and \( Q\left\lbrack  0\right\rbrack \) . Next,the algorithm HP-PEF will take the node \( s \) as the source node \( a \) since \( s \) is closer to \( Q\left\lbrack  2\right\rbrack \) along the clockwise than \( t \) . Correspondingly,the node \( t \) is deemed as the destination node \( b \) . Since \( {l}_{a} = 0 = {l}^{\prime } + 1\left( {\;\operatorname{mod}\;3}\right) \) and \( {l}_{a} = 0 \neq  {l}_{b} = 1 \) ,the algorithm HP-PEF calls the procedure HP-Direct to construct the H-path along the clockwise (i.e., \( d = 1 \) ) by the approaches of Case 1.2 in Theorem III.2,where the H-path in \( Q\left\lbrack  l\right\rbrack \) (isomorphic to \( {Q}_{2}^{3} \) ) with \( l \in  {\mathbb{Z}}_{3} \) is constructed by Algorithm 1. Fig. 7 shows the H-path constructed by Algorithm HP-PEF in \( {Q}_{3}^{3} \) ,where the dashed lines represent the faulty edges.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_8.jpg?x=214&y=186&w=553&h=695"/>

Fig. 7. The H-path constructed by Algorithm HP-PEF in \( {Q}_{3}^{3} \) ,where the solid lines represent the fault-free edges and the dashed lines represent the faulty edges.

<!-- Media -->

## V. Performance Analysis

In this section, we implement the algorithm HP-PEF by using Python programs and evaluate the performance of HP-PEF. We carry out the simulation by using a \( {2.80}\mathrm{{GHz}} \) Intel \( {}^{\circledR }{\mathrm{{Core}}}^{\mathrm{{TM}}} \) i9-10900 CPU and 127 GB RAM under the Linux operating system.

## A. Average Path Length

Along the H-path constructed by HP-PEF, we aim to compute the path length which is the number of edges between the source and destination nodes. We are interested in computing the path length since it's beneficial to evaluate the performance of HP-PEF for future applications to NoCs. Moreover, based on the following concerns,we choose the adjacent node pair in \( {Q}_{n}^{k} \) as the source and the destination to compute the path length:

1) When there exists no faulty edge in networks, the adjacent node pair possesses the shortest path length 1.

2) The algorithm HP-PEF can address large-scale edge faults. When the edge fault occurs, the path length of adjacent node pairs is naturally more sensitive to edge fault than other kinds of node pairs.

3) Our method focuses on the distribution pattern of edge faults in each dimension. The path length of adjacent node pairs may be followed by the distribution pattern of edge faults. It's interesting to explore the relationship between the path length of adjacent node pairs and the number of edge faults in each dimension.

<!-- Media -->

TABLE I

EXPERIMENTAL SETTINGS

<table><tr><td/><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>Adjacent node pairs</td><td>\( {27} \times  3 \)</td><td>\( {125} \times  3 \)</td><td>\( {343} \times  3 \)</td><td>\( {729} \times  3 \)</td><td>\( {81} \times  4 \)</td><td>\( {243} \times \)</td><td>\( {729} \times  6 \)</td></tr><tr><td>Faulty edges</td><td>8</td><td>24</td><td>48</td><td>80</td><td>33</td><td>112</td><td>353</td></tr><tr><td>H-paths</td><td>351</td><td>7750</td><td>58653</td><td>265356</td><td>3240</td><td>29403</td><td>265356</td></tr></table>

<!-- Media -->

One edge corresponds to exactly one adjacent node pair. Thus, we define the dimension of an adjacent node pair as the dimension of the edge connecting it. By the definition of \( {Q}_{n}^{k} \) , there exist \( {k}^{n}i \) -dimensional edges,which implies that there exist \( {k}^{n}i \) -dimensional adjacent node pairs for each \( i \in  {\mathbb{Z}}_{n} \) . Then \( {Q}_{n}^{k} \) has \( n{k}^{n} \) adjacent node pairs in total. As for the faulty edges,we directly set \( \left| F\right|  = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 \) to prevent the occurrence of \( \left| {F}_{2}\right|  = \cdots  = \left| {F}_{n - 1}\right| \) in the experimental process. Moreover, to make comparisons more obvious, we assume \( \left| {F}_{n - 1}\right|  \geq  \left| {F}_{n - 2}\right|  \geq  \cdots  \geq  \left| {F}_{0}\right| \) . Then we set \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) with \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ,and \( \left| {F}_{0}\right|  = 0 \) . Given a node pair(s,t) and a PEF set, the algorithm HP-PEF can generate an H-path between \( s \) and \( t \) . There exist \( \left( \begin{matrix} {k}^{n} \\  2 \end{matrix}\right)  = \frac{{k}^{2n} - {k}^{n}}{2} \) different node pairs (s,t). Thus,given a PEF set,the algorithm HP-PEF can generate \( \frac{{k}^{2n} - {k}^{n}}{2} \) different H-paths. Our experimental settings are shown in Table I.

In the simulation,we first randomly generate a PEF set \( F \) with \( \left| {F}_{i}\right|  = {k}^{i} - 2 \) for \( i \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2},\left| {F}_{1}\right|  = 1 \) ,and \( \left| {F}_{0}\right|  = 0 \) . Next, under the edge fault set \( F \) ,we utilize the algorithm HP-PEF to generate all possible \( \mathrm{H} \) -paths,that is, \( \frac{{k}^{2n} - {k}^{n}}{2} \) different \( \mathrm{H} \) -paths. And then, according to the dimension of adjacent node pairs, we compute the average path length (APL for short) of them in all \( \mathrm{H} \) -paths. More specifically,we denote the APL of each adjacent node pair in all \( \mathrm{H} \) -paths as \( p{l}_{i}^{j} \) ,where \( i \in  {\mathbb{Z}}_{n} \) is the dimension of the adjacent node pair and \( j \in  {\mathbb{Z}}_{{k}^{n}} \) is the unique identification of the adjacent node pair in the \( i \) -dimension. Then we compute the \( {\mathrm{{APL}}}_{i} \) for each dimension \( i \in  {\mathbb{Z}}_{n} \) as follows:

\[{\mathrm{{APL}}}_{i} = \frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}p{l}_{i}^{j}}{{k}^{n}}.\]

Simultaneously, we compute the standard deviation (SD for short) to evaluate the quantity of difference in path length as follows:

\[{\mathrm{{SD}}}_{i} = \sqrt{\frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}{\left( p{l}_{i}^{j} - {\mathrm{{APL}}}_{i}\right) }^{2}}{{k}^{n}}}.\]

Fig. 8 shows the simulation results about the \( {\mathrm{{APL}}}_{i} \) of \( {Q}_{n}^{k} \) with different \( k \) and \( n \) . We observe that \( {\mathrm{{APL}}}_{i} \) is positively related to \( \left| {F}_{i}\right| \) . For example,the values of \( \left| {F}_{0}\right| \) and \( \left| {F}_{1}\right| \) differ by only 1,and \( {\mathrm{{APL}}}_{1} \) is slightly higher than \( {\mathrm{{APL}}}_{0} \) . The value of \( \left| {F}_{i}\right| \) with \( i \geq  2 \) is a power function with base \( k \) ,and \( {\mathrm{{APL}}}_{i} \) is much higher than \( {\mathrm{{APL}}}_{0} \) and \( {\mathrm{{APL}}}_{1} \) . In \( {Q}_{n}^{3} \) ,the distance value between \( {\mathrm{{APL}}}_{i + 1} \) and \( {\mathrm{{APL}}}_{i} \) exhibits a sharper growth trend with larger \( i \) . Moreover,with increasing \( n \) and \( k \) ,we observe from the results that the growth rate of \( {\mathrm{{APL}}}_{i} \) is more sensitive to \( k \) than \( n \) . For example,in \( {Q}_{3}^{k} \) with increasing \( k \) from 3 to 9,the \( {\mathrm{{APL}}}_{2} \) grows rapidly from 11.1 to 154.3. However,in \( {Q}_{n}^{3} \) with increasing \( n \) from 3 to 6,the \( {\mathrm{{APL}}}_{2} \) grows slowly from 11.1 to 17.7. Although \( k \) increment by 2 and \( n \) increment by 1 on the \( x \) -axis,the growth rate of \( {\mathrm{{APL}}}_{i} \) in \( {Q}_{n}^{3} \) can still indicate the modest influence of \( n \) on \( {\mathrm{{APL}}}_{i} \) . This phenomenon may be caused by the fact that the value of \( \left| {F}_{i}\right| \) is only influenced by \( k \) ,independent of \( n \) . Fig. 9 shows the \( {\mathrm{{SD}}}_{i} \) of \( {Q}_{n}^{k} \) with different \( k \) and \( n \) . The growth trends for \( {\mathrm{{SD}}}_{i} \) and \( {\mathrm{{APL}}}_{i} \) are roughly the same. However,the parameter \( n \) can significantly affect the growth rate of \( {\mathrm{{SD}}}_{i} \) ,which is obviously different from that of \( {\mathrm{{APL}}}_{i} \) . It implies that the larger \( n \) and \( k \) , the more dispersed the values of path length.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=162&y=174&w=1404&h=490"/>

Fig. 8. The \( {\mathrm{{APL}}}_{i} \) of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=162&y=774&w=1413&h=488"/>

Fig. 9. The \( {\mathrm{{SD}}}_{i} \) of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

<!-- Media -->

Next,we compute the APL of all adjacent node pairs in \( {Q}_{n}^{k} \) as follows:

\[\mathrm{{APL}} = \frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{n}}}\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}p{l}_{i}^{j}}{n{k}^{n}}.\]

Correspondingly, the SD of all adjacent node pairs is also computed:

\[\mathrm{{SD}} = \sqrt{\frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{n}}}\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{n}}}}{\left( p{l}_{i}^{j} - \mathrm{{APL}}\right) }^{2}}{n{k}^{n}}}.\]

In Fig. 10,we show the APL and SD of \( {Q}_{n}^{k} \) . For \( {Q}_{3}^{3} \) ,the SD is slightly smaller than APL. However,with increasing \( k \) and \( n \) ,the SD grows faster than the APL. In addition, from the simulation, we are aware of the following attractive phenomena. (1) For \( {Q}_{3}^{k} \) , the ratio of APL to \( \left| F\right| \) is about 1.3. With \( k \) increases,the value of \( \left| F\right| / \) APL remains essentially the same. (2) For \( {Q}_{n}^{3} \) ,the value of \( \left| F\right| / \) APL varies approximately linearly with \( n \) . We list these results in Table II. As for what caused the positive correlation between \( n \) and \( \left| F\right| /\mathrm{{APL}} \) ,there is currently no definite attribution. We roughly think that since the algorithm HP-PEF constructs the required \( \mathrm{H} \) -path according to the distribution pattern of edge faults in each dimension, the H-path obtained is endowed with the nature of traversing all \( n \) dimensions,which results in an additional effect of \( n \) on the APL.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_9.jpg?x=898&y=1377&w=705&h=513"/>

Fig. 10. The APL and SD of adjacent node pairs in \( {Q}_{n}^{k} \) with different \( n \) and \( k \) .

TABLE II

THE \( \left| F\right| \) AND APL OF \( {Q}_{n}^{k} \) WITH DIFFERENT \( k \) AND \( n \)

<table><tr><td colspan="9"/></tr><tr><td/><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{3}^{5} \)</td><td>\( {Q}_{3}^{7} \)</td><td>\( {Q}_{3}^{9} \)</td><td>\( {Q}_{3}^{3} \)</td><td>\( {Q}_{4}^{3} \)</td><td>\( {Q}_{5}^{3} \)</td><td>\( {Q}_{6}^{3} \)</td></tr><tr><td>\( \left| F\right| \)</td><td>8</td><td>24</td><td>48</td><td>80</td><td>8</td><td>33</td><td>112</td><td>353</td></tr><tr><td>APL</td><td>6.3</td><td>17.9</td><td>36.9</td><td>61.8</td><td>6.3</td><td>14.8</td><td>35.3</td><td>88.1</td></tr><tr><td>\( \left| F\right| / \) APL</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>2.2</td><td>3.2</td><td>4.0</td></tr></table>

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_10.jpg?x=148&y=460&w=684&h=561"/>

Fig. 11. The comparisons among FP with different \( k \) .

<!-- Media -->

## B. Fault Tolerance Analysis

1) Comparison Results: Improving the fault tolerance of \( {Q}_{n}^{k} \) when we embed an \( \mathrm{H} \) -path into the faulty \( {Q}_{n}^{k} \) is the original purpose of our work. In this subsection, we compare the fault tolerance of our method with the similar known result. First, let's review the known results. Yang et al. [32] proved that \( {Q}_{n}^{k} \) with odd \( k \geq  3 \) is(2n - 3)-edge fault-tolerant Hamiltonian-connected. We demonstrated that \( {Q}_{n}^{k} \) with odd \( k \geq  3 \) is \( \left( {\frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + }\right. \) 5)-edge fault-tolerant Hamiltonian-connected under the PEF model. For convenience,let \( \mathrm{{FT}} = {2n} - 3 \) and \( \mathrm{{FP}} = \frac{{k}^{n} - {k}^{2}}{k - 1} - \) \( {2n} + 5 \) .

Obviously, the value of FP is closely related to the parameters \( k \) and \( n \) . We first investigate how odd \( k \) with different values affect FP in Fig. 11. Note that it takes \( n \) on its \( x \) -axis and takes the corresponding metrics with the exponential scale on its \( y \) -axis. It can be seen that the values of FP with different \( k \) are all the same when \( n = 2 \) . That is because the base case of the induction in our proof is handled by the result in [42], which is irrelevant to the parameter \( k \) . As \( n \) increases,the distance between any two FP with different \( k \) increases rapidly. It’s easy to see that FP is positively correlated with \( k \) . When \( n = 3 \) ,FP with \( k = 9 \) is ten times that with \( k = 3 \) . When \( n = 8 \) ,FP with \( k = 9 \) is more than \( {1648}\left( { \approx  \frac{5380819}{3265}}\right) \) times that with \( k = 3 \) .

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_10.jpg?x=894&y=188&w=745&h=542"/>

Fig. 12. The comparisons between FT and FP.

<!-- Media -->

Next, we compare the values of FP and FT. Since the value of FT is irrelevant to the parameter \( k \) ,we fix \( k = 3 \) in FP to ensure fairness when making the comparisons. Fig. 12 shows the trends of FT and FP with increasing \( n \) . Note that it takes even \( n \) on its \( x \) -axis and takes the corresponding metrics with the exponential scale on its \( y \) -axis. Since the base case of both our method and the method in [32] is handled by the result in [42], FT = FP when \( n = 2 \) . However,as \( n \) increases,FP and the ratio of FP to FT increase rapidly,while FT is linear on \( n \) . When \( n = {10},\mathrm{{FT}} = {17} \) and \( \mathrm{{FP}} = {29505} \) ,which implies \( \mathrm{{FP}}/\mathrm{{FT}} = \frac{29505}{17} \approx  {1735} \) . The great disparity between them comes from our complete consideration of the distribution pattern of edge faults in each dimension.

The fault tolerance of \( {Q}_{n}^{k} \) under the PEF model is positively correlated with \( k \) and \( n \) . This proposed model builds the positive relation between the fault tolerance with the parameter \( k \) ,which is its unique advantage. Even without considering the positive effect of \( k \) ,our method still outperforms the similar known result in terms of fault tolerance. Thus, our method can evaluate the fault tolerance of \( {Q}_{n}^{k} \) even when the edge faults are large-scale.

2) Average Success Rate: Next, we analyze the fault-tolerant capacity of the algorithm HP-PEF with increasing faulty edges. Given any PEF set \( F \) satisfying the three conditions in Theorem III.2, the required H-path can be constructed by HP-PEF with \( {100}\% \) probability,which has been proved in Theorem IV.1. In the following,provided that the PEF set \( F \) exceeds the above conditions, by implementing computer programs, we evaluate the average success rate (ASR for short) of HP-PEF, which is the ratio of the number of successfully constructed \( \mathrm{H} \) -paths over generated instances.

Before the simulation, we randomly generate one thousand PEF sets of \( {Q}_{n}^{k} \) ,denoted by \( F\left\lbrack  i\right\rbrack \) with \( i \in  {\mathbb{Z}}_{{10}^{3}} \) ,which satisfy (1) \( \left| {F\left\lbrack  i\right\rbrack  }\right|  = \mathrm{{FP}} \) ,(2) \( {e}_{j} = {k}^{j} - 2 \) for each \( j \in  {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) ,and (3) \( {e}_{0} = 0 \) and \( {e}_{1} = 1 \) . Next,in each simulation,we randomly add exactly one faulty edge into \( F\left\lbrack  i\right\rbrack \) such that \( {e}_{j} \) grows by 1 with \( j \in \) \( {\mathbb{Z}}_{n} - {\mathbb{Z}}_{2} \) . We don’t add faulty edges to increase \( {e}_{0} \) or \( {e}_{1} \) since it will seriously affect the effectiveness of Algorithm 1, which is designed based on the method of [42]. Then, for each faulty edge set \( F\left\lbrack  i\right\rbrack \) ,we determine whether the required \( \mathrm{H} \) -path between the node pair(s,t)is constructed successfully (i.e.,success \( \left( j\right)  = \) 1) or not (i.e.,success \( \left( j\right)  = 0 \) ),where \( j \in  {\mathbb{Z}}_{{k}^{2n} - {k}^{n}} \) is the unique identification of the node pair(s,t). Then we compute the \( {\mathrm{{ASR}}}_{i} \) of \( F\left\lbrack  i\right\rbrack \) as follows:

\[{\mathrm{{ASR}}}_{i} = \frac{\mathop{\sum }\limits_{{j \in  {\mathbb{Z}}_{{k}^{2n} - {k}^{n}}}}\operatorname{success}\left( j\right) }{{k}^{2n} - {k}^{n}} \times  {100}\% .\]

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_11.jpg?x=201&y=183&w=1326&h=515"/>

Fig. 13. The \( {\mathrm{{ASR}}}_{\min } \) and ASR of \( {Q}_{3}^{3},{Q}_{4}^{3} \) ,and \( {Q}_{3}^{5} \) with half dots drawn.

<!-- Media -->

Based on \( {\mathrm{{ASR}}}_{i} \) ,we then compute the \( {\mathrm{{ASR}}}_{\min } \) and \( \mathrm{{ASR}} \) as follows:

\[{\mathrm{{ASR}}}_{\min } = \min \left\{  {{\mathrm{{ASR}}}_{i} \mid  i \in  {\mathbb{Z}}_{{10}^{3}}}\right\}  .\]

\[\mathrm{{ASR}} = \frac{\mathop{\sum }\limits_{{i \in  {\mathbb{Z}}_{{10}^{3}}}}{\mathrm{{ASR}}}_{i}}{{10}^{3}}.\]

For \( \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {80} \) ,we show the variation trend of \( {\mathrm{{ASR}}}_{\min } \) and ASR of \( {Q}_{3}^{3},{Q}_{4}^{3} \) ,and \( {Q}_{3}^{5} \) in Fig. 13,respectively. Note that for \( {Q}_{3}^{3} \) (resp. \( {Q}_{4}^{3} \) and \( {Q}_{3}^{5} \) ),FP \( = \frac{{k}^{n} - {k}^{2}}{k - 1} - {2n} + 5 = 8 \) (resp. 33 and 24). It can be observed that both \( {\mathrm{{ASR}}}_{\min } \) and ASR support dynamic degradation. For \( {Q}_{3}^{3},{\mathrm{{ASR}}}_{\min } \) (resp. ASR) maintains over 47.2% (resp. 90.2%) even when increasing faulty edges achieve 2FP (i.e., 16).

As \( n \) or \( k \) increases,the decline curve of ASR becomes moderate. The ASR of \( {Q}_{4}^{3} \) (resp. \( {Q}_{3}^{5} \) ) maintains \( {96.2}\% \) (resp. over 100%) when increasing faulty edges achieve 2FP (i.e., 66 and 48, respectively). As for \( {\mathrm{{ASR}}}_{\min } \) ,the curve is much steeper than that of ASR though it still decreases in general. We can observe that there exist several intervals in the curve where \( {\mathrm{{ASR}}}_{\min } \) remains unchanged. For instance,when \( {42} \leq  \left| {F\left\lbrack  i\right\rbrack  }\right|  \leq  {46} \) ,the \( {\mathrm{{ASR}}}_{\text{min }} \) of \( {Q}_{4}^{3} \) is always equal to \( {92}\% \) . After the increasing faulty edges exceed 46,the \( {\mathrm{{ASR}}}_{\text{min }} \) of \( {Q}_{4}^{3} \) exhibits a cliff-like drop to \( {77}\% \) . The reason behind this phenomenon can be easily explained. In the for-loops of Procedures HP-Round and HP-Direct, some expected fault-free edges connecting two consecutive subgraphs are selected to construct the H-path, which determines whether algorithm HP-PEF can successfully be implemented. Thus, if the faulty edges we add into \( F\left\lbrack  i\right\rbrack \) aren’t concentrated between two consecutive subgraphs, then the success rate of HP-PEF will not be affected by the number of faulty edges. However, when the number of faulty edges exceeds a threshold value which makes that the procedures cannot successfully select the expected edges connecting two consecutive subgraphs,all the \( \mathrm{H} \) -paths that should pass through these two consecutive subgraphs cannot be successfully constructed.

It can be observed that the larger the values of \( n \) and \( k \) ,the smaller the decline rate of both ASR and \( {\mathrm{{ASR}}}_{\text{min }} \) . Therefore,it can be predicted that the algorithm HP-PEF will exhibit a more excellent fault-tolerant capacity for \( {Q}_{n}^{k} \) systems with larger \( n \) and \( k \) .

## VI. Concluding Remarks

We introduce a new edge-fault model named PEF model, for the purpose of embedding \( \mathrm{H} \) -paths into the \( k \) -ary \( n \) -cube \( {Q}_{n}^{k} \) containing a large number of faulty edges. Using the PEF model, we design, as well as implement, an efficient fault-tolerant H-path embedding algorithm for \( {Q}_{n}^{k} \) . Due to the recursive structure of \( {Q}_{n}^{k} \) and its edge symmetry property,we believe similar algorithms can be designed for other recursive, edge symmetric networks, such as generalized hypercubes and star graphs. Another promising direction of future work is to investigate the embedding of the H-path in \( k \) -ary \( n \) -cubes under the PEF model, so that the H-path passes through some prescribed linear forests, exponentially improving the recent results in [34].

## REFERENCES

[1] W. J. Dally and B. Towles, "Route packets, not wires: On-chip interconnection networks," in Proc. 38th Des. Automat. Conf., 2001, pp. 684-689.

[2] A. Jantsch and H. Tenhunen, Networks on Chip, Norwell, MA, USA: Kluwer, 2003.

[3] J. H. Lau, "Evolution, challenge, and outlook of TSV, 3D IC integration and 3D silicon integration," in Proc. IEEE Int. Symp. Adv. Packag. Mater., 2011, pp. 462-488.

[4] P. Bogdan, T. Dumitraş, and R. Marculescu, "Stochastic communication: A new paradigm for fault-tolerant networks-on-chip," VLSI Des., vol. 2007, p. 17, 2007.

[5] Y. Zhang et al., "A deterministic-path routing algorithm for tolerating many faults on very-large-scale network-on-chip," ACM Trans. Des. Automat. Electron. Syst., vol. 26, no. 1, pp. 1-26, Jan. 2021.

[6] E. Taheri, M. Isakov, A. Patooghy, and M. A. Kinsy, "Addressing a new class of reliability threats in 3-D network-on-chips," IEEE Trans. Comput. Aided Des. Integr. Circuits Syst., vol. 39, no. 7, pp. 1358-1371, Jul. 2020.

[7] M. Ebrahimi, M. Daneshtalab, and J. Plosila, "Fault-tolerant routing algorithm for 3D NoC using Hamiltonian path strategy," in Proc. Des. Autom. Test Eur. Conf. Exhib., 2013, pp. 1601-1604.

[8] C. Hu, C. M. Meyer, X. Jiang, and T. Watanabe, "A fault-tolerant Hamiltonian-based odd-even routing algorithm for network-on-chip," in Proc. IEEE 35th Int. Technol. Conf. Circuits/Syst. Circuits Commun., 2020, pp. 217-222.

[9] P. Bahrebar and D. Stroobandt, "Improving Hamiltonian-based routing methods for on-chip networks: A turn model approach," in Proc. Des. Autom. Test Eur. Conf. Exhib., 2014, pp. 1-4.

[10] P. Bahrebar and D. Stroobandt, "The Hamiltonian-based odd-even turn model for maximally adaptive routing in 2D mesh networks-on-chip," Comput. Electr. Eng., vol. 45, pp. 386-401, Jul. 2015.

[11] E. O. Amnah and W. L. Zuo, "Hamiltonian paths for designing deadlock-free multicasting wormhole-routing algorithms in 3-D meshes," J. Appl. Sci., vol. 7, no. 22, pp. 3410-3419, 2007.

[12] M. Ebrahimi, M. Daneshtalab, P. Liljeberg, and H. Tenhunen, "HAMUM - A novel routing protocol for unicast and multicast traffic in MPSoCs," in Proc. 18th Euromicro Int. Conf. Parallel Distrib. Netw. Based Process., 2010, pp. 525-532.

[13] X. Lin and L. M. Ni, "Multicast communication in multicomputer networks," IEEE Trans. Parallel Distrib. Syst., vol. 4, no. 10, pp. 1105-1117, Oct. 1993.

[14] W. J. Dally,"Performance analysis of \( k \) -ary \( n \) -cube interconnection networks," IEEE Trans. Comput., vol. 19, no. 6, pp. 775-785, Jun. 1990.

[15] W. J. Dally and C. L. Seitz, "The tours routing chip," Distrib. Comput., vol. 1, no. 3, pp. 187-196, Dec. 1986.

[16] D. H. Linder and J. C. Harden, "An adaptive and fault tolerant wormhole routing strategy for \( k \) -ary \( n \) -cubes," IEEE Trans. Comput.,vol. 40,no. 1, pp. 2-12, Jan. 1991.

[17] W. Luo and D. Xiang, "An efficient adaptive deadlock-free routing algorithm for torus networks," IEEE Trans. Parallel Distrib. Syst., vol. 23, no. 5, pp. 800-808, May 2012.

[18] P. Ren, X. Ren, S. Sane, M. A. Kinsy, and N. Zheng, "A deadlock-free and connectivity-guaranteed methodology for achieving fault-tolerance in on-chip networks," IEEE Trans. Comput., vol. 65, no. 2, pp. 353-366, Feb. 2016.

[19] H. Abu-Libdeh et al., "Symbiotic routing in future data centers," ACM SIGCOMM Comput. Commun. Rev., vol. 40, no. 4, pp. 51-62, Oct. 2010.

[20] T. Wang, Z. Su, Y. Xia, B. Qin, and M. Hamdi, "NovaCube: A low latency torus-based network architecture for data centers," in Proc. IEEE Glob. Commun. Conf., 2014, pp. 2252-2257.

[21] T. Wang, Z. Su, Y. Xia, and M. Hamdi, "CLOT: A cost-effective low-latency overlaid torus-based network architecture for data centers," in Proc. IEEE Conf. Commun., 2015, pp. 5479-5484.

[22] K. Chen et al., "WaveCube: A scalable, fault-tolerant, high-performance optical data center architecture," in Proc. IEEE Conf. Comput. Commun., 2015, pp. 1903-1911.

[23] S. Borkar et al., "iWarp: An integrated solution to high-speed parallel computing," in Proc. Int. Conf. Supercomput., 1988, pp. 330-339.

[24] M. D. Noakes, D. A. Wallach, and W. J. Dally, "The J-Machine multicomputer: An architectural evaluation," in Proc. 20th Ann. Int. Symp. Comput. Architecture, 1993, pp. 224-235.

[25] R. E. Kessler and J. L. Schwarzmeier, "Cray T3D: A new dimension for Cray research," in Proc. 38th IEEE Int. Comput. Conf., 1993, pp. 176-182.

[26] E. Anderson, J. Brooks, C. Grassl, and S. Scott, "Performance of the Cray T3E multiprocessor," in Proc. ACM/IEEE Conf. Supercomput., 1997, pp. 1-17.

[27] N. R. Adiga et al., "Blue Gene/L torus interconnection network," IBM J. Res. Dev., vol. 49, no. 2/3, pp. 265-276, Mar. 2005.

[28] Y. Lv, C.-K. Lin, J. Fan, and X. Jia, "Hamiltonian cycle and path embed-dings in 3-ary \( n \) -cubes based on \( {K}_{1,3} \) -structure faults," J. Parallel Distrib. Comput., vol. 120, pp. 148-158, 2018.

[29] S. Wang, S. Zhang, and Y. Yang, "Hamiltonian path embeddings in conditional faulty \( k \) -ary \( n \) -cubes," Inf. Sci.,vol. 268,pp. 463-488,Jun. 2014.

[30] Y. Xiang and I. A. Stewart,"Bipancyclicity in \( k \) -ary \( n \) -cubes with faulty edges under a conditional fault assumption," IEEE Trans. Parallel Distrib. Syst., vol. 22, no. 9, pp. 1506-1513, Sep. 2011.

[31] S. Zhang and X. Zhang, "Fault-free Hamiltonian cycles passing through prescribed edges in \( k \) -ary \( n \) -cubes with faulty edges," IEEE Trans. Parallel Distrib. Syst., vol. 26, no. 2, pp. 434-443, Feb. 2015.

[32] M.-C. Yang, J. J. M. Tan, and L.-H. Hsu, "Hamiltonian circuit and linear array embeddings in faulty \( k \) -ary \( n \) -cubes," J. Parallel Distrib. Comput., vol. 67, pp. 362-368, Apr. 2007.

[33] I. A. Stewart and Y. Xiang,"Embedding long paths in \( k \) -ary \( n \) -cubes with faulty nodes and links," IEEE Trans. Parallel Distrib. Syst., vol. 19, no. 8, pp. 1071-1085, Aug. 2008.

[34] Y. Yang and L. Zhang,"Hamiltonian paths of \( k \) -ary \( n \) -cubes avoiding faulty links and passing through prescribed linear forests," IEEE Trans. Parallel Distrib. Syst., vol. 33, no. 7, pp. 1752-1760, Jul. 2022.

[35] J. Yuan et al.,"The \( g \) -good-neighbor conditional diagnosability of \( k \) -ary \( n \) -cubes under the PMC model and MM model," IEEE Trans. Parallel Distrib. Syst., vol. 26, no. 4, pp. 1165-1177, Apr. 2015.

[36] L. Xu, L. Lin, S. Zhou, and S.-Y. Hsieh, "The extra connectivity, extra conditional diagnosability,and \( t/m \) -diagnosability of arrangement graphs," IEEE Trans. Rel., vol. 65, no. 3, pp. 1248-1262, Sep. 2016.

[37] R. Salamat, M. Khayambashi, M. Ebrahimi, and N. Bagherzadeh, "A resilient routing algorithm with formal reliability analysis for partially connected 3D-NoCs," IEEE Trans. Comput., vol. 65, no. 11, pp. 3265- 3279, Nov. 2016.

[38] A. Charif, A. Coelho, M. Ebrahimi, N. Bagherzadeh, and N.-E. Zergainoh, "First-Last: A cost-effective adaptive routing solution for TSV-based three-dimensional networks-on-chip," IEEE Trans. Comput., vol. 67, no. 10, pp. 1430-1444, Oct. 2018.

[39] H. Zhang, R.-X. Hao, X.-W. Qin, C.-K. Lin, and S.-Y. Hsieh, "The high faulty tolerant capability of the alternating group graphs," IEEE Trans. Parallel Distrib. Syst., vol. 34, no. 1, pp. 225-233, Jan. 2023.

[40] L.-H. Hsu and C.-K. Lin, Graph Theory and Interconnection Networks. Boca Raton, FL, USA: CRC Press, 2008.

[41] Y. A. Ashir and I. A. Stewart,"On embedding cycles in \( k \) -ary \( n \) -cubes," Parallel Process. Lett., vol. 7, no. 1, pp. 49-55, 1997.

[42] H.-C. Kim and J.-H. Park, "Fault Hamiltonicity of two-dimensional torus networks," in Proc. 5th Jpn. Korea Joint Workshop Algorithms Comput., 2000, pp. 110-117.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_12.jpg?x=889&y=1244&w=232&h=289"/>

<!-- Media -->

Hongbin Zhuang received the BEng degree from Huaqiao University, Xiamen, China, in 2019. He is currently working toward the doctoral degree with the College of Computer and Data Science, Fuzhou University, China. His research interests include design and analysis of algorithms, fault diagnosis, and fault-tolerant computing.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_12.jpg?x=892&y=1670&w=222&h=275"/>

<!-- Media -->

Xiao-Yan Li received the PhD degree in computer science from Soochow University, Suzhou, China, in 2019. She was a visiting scholar in the Department of Computer Science, the City University of Hong Kong, Hong Kong, from June 2018-June 2019. She is currently an associate professor with the College of Computer and Data Science, Fuzhou University, China. She has published more than 30 papers in research-related journals and conferences, such as IEEE Transactions on Parallel and Distributed Systems, IEEE/ACM Transactions on Networking, IEEE Transactions on Computers, Journal of Parallel and Distributed Computing, and Association for the Advancement of Artificial Intelligence. She has served some conferences as the Session Chair and Program Committee Member, including IEEE BIBM 2020, IEEE TrustCom 2020 Workshop, WWW 2021, AAAI 2022 Workshop. Her research interests include graph theory, data center networks, parallel and distributed systems, design and analysis of algorithms, and fault diagnosis.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_13.jpg?x=105&y=190&w=226&h=279"/>

<!-- Media -->

Jou-Ming Chang received the BS degree in applied mathematics from Chinese Culture University, Taipei, Taiwan, in 1987, the MS degree in information management from National Chiao Tung University, Hsinchu, Taiwan, in 1992, and the PhD degree in computer science and information engineering from National Central University, Zhongli, Taiwan, in 2001. He served as the Dean of the College of Management, National Taipei University of Business (NTUB), Taipei, in 2014. He is currently a Distinguished Professor in the Institute of Information and Decision Sciences, NTUB. He has published more than 150 research papers in refereed journals and conferences, including IEEE Transactions on Parallel and Distributed Systems, IEEE/ACM Transactions on Networking, IEEE Transactions on Computers. His major research interests include algorithm analysis and design, graph theory, and parallel and distributed computing.

<!-- Media -->

<img src="https://cdn.noedgeai.com/0193cad3-d1fd-725c-b228-f570603d2cf4_13.jpg?x=878&y=191&w=224&h=277"/>

<!-- Media -->

Dajin Wang received the BEng degree in computer engineering from Shanghai University of Science and Technology, Shanghai, China, in 1982, and the PhD degree in computer science from the Stevens Institute of Technology, Hoboken, USA, in 1990. He is currently a professor in the Department of Computer Science with Montclair State University. He received several university level awards for his scholarly accomplishments. He has held visiting positions in other universities, and has consulted in industry. He served as an associate editor of IEEE Transactions on Parallel and Distributed Systems from 2010 to 2014. He has published over one hundred papers in these areas. Many of his works appeared in premier journals including IEEE Transactions on Computers, IEEE Transactions on Parallel and Distributed Systems, IEEE Transactions on Systems, Man and Cybernetics, IEEE Transactions on Relibility, Journal of Parallel and Distributed Computing, and Parallel Computing. He has served on the program committees of influential conferences. His main research interests include interconnection networks, fault tolerant computing, algorithmic robotics, parallel processing, and wireless ad hoc and sensor networks.