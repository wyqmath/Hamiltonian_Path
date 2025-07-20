# Disjoint cycles through prescribed vertices in multidimensional tori

Amruta Shinde \( {}^{1} \) and Y. M. Borse \( {}^{2} \)

\( {}^{1,2} \) Department of Mathematics,Savitribai Phule Pune University,Pune 411 007,India

e-mail: \( {}^{1} \) samruta421@gmail.com; \( {}^{2} \) ymborse11@gmail.com

Abstract. For a positive integer \( r \) ,a graph \( G \) is spanning \( r \) -cyclable if for any given set \( F \) of \( r \) vertices,there exists \( r \) vertex-disjoint cycles that together span \( G \) and each cycle contains exactly one vertex from \( F \) . It is known that the hypercube \( {Q}_{n} \) and its variation,the crossed cube,are spanning \( r \) -cyclable for \( 1 \leq  r \leq  n - 1 \) . We prove that every \( n \) -dimensional torus,different from \( {C}_{3}\square {C}_{3} \) ,is spanning \( r \) -cyclable for \( 1 \leq  r \leq  {2n} - 1 \) .

2000 Mathematics Subject Classification: \( {68}\mathrm{R}{10},{05}\mathrm{C}{38} \) .

## 1. Introduction

All graphs under consideration are simple,finite and undirected. Let \( G = \left( {V,E}\right) \) be a graph with vertex set \( V \) and edge set \( E \) and let \( \delta \left( G\right) \) denote the minimum degree of \( G \) . The Cartesian product of two graphs \( {G}_{1} = \left( {{V}_{1},{E}_{1}}\right) \) and \( {G}_{2} = \left( {{V}_{2},{E}_{2}}\right) \) is a graph \( {G}_{1}\square {G}_{2} = \left( {V,E}\right) \) ,where \( V = {V}_{1} \times  {V}_{2} \) and \( E\left( {{G}_{1}\square {G}_{2}}\right)  = \left\{  {\left\langle  {\left( {{x}_{1},{y}_{1}}\right) ,\left( {{x}_{2},{y}_{2}}\right) }\right\rangle   : {x}_{1} = {x}_{2}}\right. \) and \( \left\langle  {{y}_{1},{y}_{2}}\right\rangle   \in  {E}_{2} \) ,or \( {y}_{1} = {y}_{2} \) and \( \left\langle  {{x}_{1},{x}_{2}}\right\rangle   \in  {E}_{1}\} \) . For a positive integer \( n \) ,the hypercube of dimension \( n \) ,denoted by \( {Q}_{n} \) ,is the Cartesian product of \( n \) copies of complete graph \( {K}_{2} \) . An \( n \) -dimensional torus is the Cartesian product of \( n \) cycles. Note that if \( n \) is even,then the hypercube \( {Q}_{n} \) is the Cartesian product of \( \frac{n}{2} \) cycles of length four. Hypercubes and multidimensional tori are popular interconnection networks in parallel computing [10].

A 2-factor of a graph is its spanning 2-regular subgraph. Note that each component of a 2-factor is a cycle. The theory of vertex-disjoint cycles and 2-factors of graphs is an extension of the theory of Hamiltonian cycles. One of the early results on this topic is due to Corrádi and Hajnal [3]. They proved that a graph with at least \( {3r} \) vertices and of minimum degree at least \( {2r} \) has \( r \) vertex-disjoint cycles.

Let \( r \) be a positive integer. A graph \( G \) is spanning \( r \) -cyclable if for any given \( r \) vertices \( {v}_{1},{v}_{2},\ldots ,{v}_{r} \) ,there exists \( r \) vertex-disjoint cycles \( {C}_{1},{C}_{2},\ldots ,{C}_{r} \) such that their union \( {C}_{1} \cup  {C}_{2} \cup  \cdots  \cup  {C}_{r} \) spans \( G \) ,and \( {C}_{i} \) contains the vertex \( {v}_{i} \) for \( 1 \leq  i \leq  r \) . Thus a graph \( G \) is spanning \( r \) -cyclable if it has a 2-factor \( F \) consisting of \( r \) cycles such that each cycle contains exactly one of the prescribed \( r \) vertices. Note that a Hamiltonian graph is 1-cyclable.

Egawa et al. [4] proved that a graph \( G \) with \( n \) vertices is spanning \( r \) -cyclable if \( {4r} \leq  n \leq  {6r} - 3 \) and \( \delta \left( G\right)  \geq  {3r} - 1 \) or \( n \geq  {6r} - 3 \) and \( \delta \left( G\right)  \geq  \frac{n}{2} \) and also obtained similar conditions when \( {3r} \leq  n \leq  {4r} \) . Ishigami and Jiang [7] considered the spanning \( r \) -cyclability of graphs in terms of small cycles. They proved that if a graph \( G \) has \( n \geq  c{r}^{2} \) vertices with sufficiently large constant \( c \) and \( \delta \left( G\right)  \geq  \left\lfloor  {\sqrt{n + \left( {\frac{9}{4}{r}^{2} - {4r} + 1}\right) } + \frac{3}{2}r - 1}\right\rfloor \) ,then there exists a vertex partition of \( G \) into cycles of length at most six,each containing exactly one of the \( r \) prescribed vertices. The problem of spanning \( r \) -cyclability of graphs is well studied in the literature; see the survey articles [2,6]. Lin et al. [11] considered the problem for the class of hypercubes and obtained the following optimum result.

Theorem 1.1 ([11]). For positive integers \( n \geq  2 \) and \( r \) such that \( 1 \leq  r \leq  n - 1 \) ,the hypercube \( {Q}_{n} \) is spanning \( r \) -cyclable.

A similar result for the class of crossed cubes \( C{Q}_{n} \) ,which is a variation of hypercubes,is proved in [9].

In this paper,we consider the problem of spanning \( r \) -cyclability for the class of multidimensional tori. Since an \( n \) -dimensional torus is a \( {2n} \) -regular graph,the upper bound on \( r \) is \( {2n} - 1 \) ; see Lemma 2.6. The following is the main result of the paper.

Theorem 1.2. Let \( n \) and \( r \) be integers with \( 1 \leq  r \leq  {2n} - 1 \) . Then every \( n \) -dimensional torus,which is not isomorphic to \( {C}_{3}\square {C}_{3} \) ,is spanning \( r \) -cyclable.

The proof proceeds by induction on \( n \) . In Section 2,we prove Theorem 1.2 for the special case \( {C}_{3}\square {C}_{3}\square {C}_{3} \) ,and settle the general case in Section 3.

## 2. Special case

For an integer \( k \geq  3 \) ,by \( {C}_{k} \) or a \( k \) -cycle,we mean a cycle of length \( k \) . If the vertices of \( {C}_{k} \) are labelled by \( 1,2,\ldots ,k \) with \( i \) adjacent to \( i + 1\left( {\;\operatorname{mod}\;k}\right) \) ,then we write \( {C}_{k} = \langle 1,2,\ldots ,k,1\rangle \) . Similarly, \( \langle 1,2,\ldots ,k\rangle \) denotes a path on \( k \) vertices from 1 to \( k \) . A path is said to be trivial if it consists of only one vertex.

In this section,we prove the spanning \( r \) -cyclability of the Cartesian product of \( n \) triangles for \( n = 2 \) and \( n = 3 \) .

The following is a special case of a result due to Kim and Park [8].

Lemma 2.1 ([8]). Let \( n \geq  2 \) be an integer and \( u \) be any vertex of an \( n \) -dimensional non-bipartite torus \( T \) . Then \( T - \{ u\} \) has a spanning cycle.

We need the following known results.

Lemma 2.2 ([12]). The Cartesian product of two non-trivial paths is a Hamiltonian graph if one of the two paths has an even number of vertices.

Lemma 2.3 ([1]). The Cartesian product of a cycle and a path on at least three vertices is a Hamiltonian graph.

Corollary 2.4. For a cycle \( C \) and a path \( P \) ,the graph \( C\square P \) is a Hamiltonian graph.

Proof. The statement obviously follows from Lemma 2.3 if \( P \) has at least three vertices. This also follows when \( P \) is a trivial path,as in this case, \( C\square P \) is just a cycle. Suppose \( P \) has two vertices. For any edge \( e \) of \( C \) ,by Lemma 2.2, the Cartesian product of the paths \( C - e \) and \( P \) has a spanning cycle,say \( Z \) . Clearly,the cycle \( Z \) also spans the graph \( C\square P \) . Hence \( C\square P \) is a Hamiltonian graph.

Corollary 2.5. For \( n \geq  1 \) ,every \( n \) -dimensional torus is a Hamiltonian graph.

Proof. The result obviously holds for \( n = 1 \) . Suppose \( n \geq  2 \) and assume that the result holds for the Cartesian product of \( n - 1 \) cycles. Let \( T = {C}_{{k}_{1}}\square {C}_{{k}_{2}}\square \cdots \square {C}_{{k}_{n}} \) . By induction, \( {C}_{{k}_{1}}\square {C}_{{k}_{2}}\square \cdots \square {C}_{{k}_{n - 1}} \) has a spanning cycle, say \( Z \) . By Corollary 2.4,for any edge \( e \) of \( Z \) ,the subgraph \( \left( {Z - e}\right) \square {C}_{{k}_{n}} \) of \( T \) has a spanning cycle \( {Z}^{\prime } \) . Then \( {Z}^{\prime } \) also spans \( T \) . Hence \( T \) is a Hamiltonian graph.

The following lemma shows that the bound on \( r \) in Theorem 1.2 is optimum.

Lemma 2.6. For \( n \geq  1 \) ,an \( n \) -dimensional torus is not spanning \( r \) -cyclable if \( r \geq  {2n} \) .

Proof. Note that it is sufficient to prove the result for \( r = {2n} \) only. Let \( T \) be an \( n \) -dimensional torus. Then \( T \) is a \( {2n} \) -regular graph. Let \( u \) be a vertex of \( T \) with neighbours \( {u}_{1},{u}_{2},\ldots ,{u}_{2n} \) . We set \( S = \left\{  {{u}_{1},{u}_{2},\ldots ,{u}_{{2n} - 1}}\right\} \) and \( F = S \cup  \{ u\} \) so that \( \left| F\right|  = {2n} \) . In the graph \( T - S,u \) is a pendant vertex and so it does not belong to any cycle. Thus we do not get \( {2n} \) vertex-disjoint cycles each containing exactly one vertex from the set \( F \) . Therefore \( T \) is not spanning \( {2n} \) -cyclable.

In the following lemma,we study the spanning \( r \) -cyclablity of the graph \( {C}_{3}\square {C}_{3} \) .

Lemma 2.7. The graph \( {C}_{3}\square {C}_{3} \) is spanning 1-cyclable and 2-cyclable but not 3-cyclable. Proof. By Corollary 2.5, \( T = {C}_{3}\square {C}_{3} \) is a Hamiltonian graph and so it is spanning 1-cyclable. Now we prove that \( T \) is spanning 2-cyclable. Let \( u \) and \( v \) be any two vertices of \( T \) . It is easy to see that \( T \) has a 3-cycle \( {Z}_{u} \) containing \( u \) but not \( v \) . It follows that \( T - V\left( {Z}_{u}\right) \) is isomorphic to \( {C}_{3}\square {K}_{2} \) and so,by Corollary 2.4,it has a spanning cycle \( {Z}_{v} \) . Thus \( {Z}_{u} \) and \( {Z}_{v} \) are vertex-disjoint cycles containing \( u \) and \( v \) ,respectively and further,their union spans \( T \) . Hence \( T \) is spanning 2-cyclable.

Assume that \( T \) is 3-cyclable. Consider three vertices \( x,y \) and \( z \) of \( T \) as shown in Figure 1. There exist three vertex-disjoint cycles \( {Z}_{x},{Z}_{y} \) and \( {Z}_{z} \) containing \( x,y \) and \( z \) ,respectively. Since \( T \) has exactly nine vertices,each of these cycles is a triangle. However,it is clear from the figure that every cycle containing \( y \) but avoiding \( x \) and \( z \) has length at least four,a contradiction. Hence \( T \) is not spanning 3-cyclable.

<!-- Media -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_2.jpg?x=722&y=659&w=204&h=199&r=0"/>

Figure 1. \( {C}_{3}\square {C}_{3} \)

<!-- Media -->

Lemma 2.8. The graph \( {C}_{3}\square {C}_{3}\square {C}_{3} \) is spanning \( r \) -cyclable for any integer \( r \) with \( 1 \leq  r \leq  5 \) .

Proof. Let \( T = {C}_{3}\square {C}_{3}\square {C}_{3} \) . Then \( T = H\square {C}_{3} \) ,where \( H = {C}_{3}\square {C}_{3} \) . The torus \( T \) is obtained by replacing each vertex of a 3-cycle by a copy of \( H \) ,say \( {H}_{1},{H}_{2} \) and \( {H}_{3} \) ,and replacing each edge by a perfect matching \( {M}_{i} \) between \( {H}_{i} \) and \( {H}_{i + 1\left( {\;\operatorname{mod}\;3}\right) } \) for \( 1 \leq  i \leq  3 \) .

Let \( F \) be a set consisting of \( r \) arbitrary vertices of \( T \) . We find a collection \( \mathcal{C} \) of \( r \) vertex-disjoint cycles whose union is a spanning subgraph of \( T \) and further,each cycle contains exactly one vertex from the set \( F \) . Let \( {F}_{i} = {H}_{i} \cap  F \) and \( \left| {F}_{i}\right|  = {r}_{i} \) for \( i = 1,2,3 \) . We may assume that \( {r}_{1} \geq  {r}_{2} \geq  {r}_{3} \) . If \( r = 1 \) ,then by Corollary 2.5, \( T \) is spanning 1-cyclable. Therefore we assume that \( r \geq  2 \) .

Case 1. \( {r}_{3} = 0 \) and \( {r}_{2} = 0 \) .

In this case, \( r = {r}_{1} \) and so, \( F = {F}_{1} \) . Hence \( {H}_{1} \) contains the set \( F \) . By Corollary 2.5, \( {H}_{1} \) has a Hamiltonian cycle \( Z \) . We split \( Z \) into \( r \) vertex-disjoint paths \( {P}_{1},{P}_{2},\ldots ,{P}_{r} \) such that each \( {P}_{i} \) contains exactly one vertex from \( F \) and the union of these paths spans \( Z \) . Some \( {P}_{i} \) may be trivial. We use thsese paths to obtain the cycles that span the graphs \( {H}_{2} \) and \( {H}_{3} \) also. By Corollary 2.4,the subgraph \( {P}_{i}\square {C}_{3} \) of \( T \) has a Hamiltonian cycle \( {Z}_{i} \) for \( i = 1,2,\ldots ,r \) . Obviously, each \( {Z}_{i} \) contains exactly one vertex from the set \( F \) . Thus \( \mathcal{C} = \left\{  {{Z}_{1},{Z}_{2},\ldots ,{Z}_{r}}\right\} \) is a collection of vertex-disjoint cycles that together span \( T \) ; see Figure 2.

<!-- Media -->

<!-- figureText: \( {H}_{1} \) \( {H}_{2} \) \( {H}_{3} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_2.jpg?x=480&y=1608&w=686&h=264&r=0"/>

Figure 2. Cycles in \( T \) when \( {r}_{1} \neq  0 \) and \( {r}_{2} = {r}_{3} = 0 \)

<!-- Media -->

Case 2. \( {r}_{3} = 0 \) but \( {r}_{2} \neq  0 \) .

Then \( {r}_{1} \leq  4 \) and \( {r}_{2} \) is 1 or 2 . By Lemma 2.7, \( {H}_{2} \) is spanning \( {r}_{2} \) -cyclable. Thus there is a collection \( {\mathcal{C}}_{2} \) of \( {r}_{2} \) vertex-disjoint cycles,each containing one vertex from the set \( {F}_{2} \) ,such that the union of these cycles spans \( {H}_{2} \) . Note that,the subgraph of \( T \) induced by vertices of \( {H}_{1} \) and \( {H}_{3} \) is isomorphic to \( {H}_{1}\square {K}_{2} \) . By Corollary 2.5, \( {H}_{1} \) has a spanning cycle \( Z \) . We use this cycle to obtain cycles that span \( {H}_{1} \) and \( {H}_{3} \) . If \( {r}_{1} = 1 \) ,then,by Corollary 2.4, \( Z\square {K}_{2} \) has a spanning cycle \( {Z}^{\prime } \) . Then \( {Z}^{\prime } \) spans \( {H}_{1} \) and \( {H}_{3} \) ,and contains the prescribed vertex of \( {F}_{1} \) . Hence \( \mathcal{C} = {\mathcal{C}}_{2} \cup  {Z}^{\prime } \) is a required collection of vertex-disjoint cycles whose union spans \( T \) . Now,suppose \( {r}_{1} \geq  2 \) . Note that we can choose a cycle \( Z \) so that \( Z \) can be split into \( r \) non-trivial subpaths \( {P}_{1},{P}_{2},\ldots ,{P}_{{r}_{1}} \) such that each path contains exactly one vertex from the set \( {F}_{1} \) and their union spans \( Z \) . By Lemma 2.2,the subgraph \( {P}_{i}\square {K}_{2} \) of \( T \) has a spanning cycle \( {Z}_{i} \) for \( i = 1,2,\ldots ,{r}_{1} \) . Since each \( {Z}_{i} \) contains a vertex from \( {F}_{1},\mathcal{C} = {\mathcal{C}}_{2} \cup  \left\{  {{Z}_{1},{Z}_{2},\ldots ,{Z}_{{r}_{1}}}\right\} \) is a required collection of vertex-disjoint cycles; see Figure 3.

<!-- Media -->

<!-- figureText: \( {H}_{2} \) \( {P}_{1} \) \( {H}_{1} \) \( {H}_{3} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_3.jpg?x=477&y=577&w=691&h=270&r=0"/>

Figure 3. Cycles in \( T \) when \( {r}_{3} = 0 \) and \( {r}_{2} = 2 \)

<!-- Media -->

Case 3. \( {r}_{3} \neq  0 \) .

As \( r \leq  5,1 \leq  {r}_{3} \leq  {r}_{2} \leq  2 \) and \( {r}_{2} \leq  {r}_{1} \leq  3 \) . Suppose \( {r}_{1} \leq  2 \) . Then,by Lemma 2.7, \( {H}_{i} \) has a collection \( {\mathcal{C}}_{i} \) of \( {r}_{i} \) vertex-disjoint cycles whose union spans \( {H}_{i} \) and each cycle contains exactly one vertex from the set \( {F}_{i} \) for \( i = 1,2,3 \) . Thus \( {\mathcal{C}}_{1} \cup  {\mathcal{C}}_{2} \cup  {\mathcal{C}}_{3} \) is a required collection of cycles that proves \( r \) -cyclability of \( T \) .

<!-- Media -->

<!-- figureText: \( u \circledast \) \( u \circledast \) W ( \( u \circledast \) \( u \odot \) W . \( v \) @ \( W \) (c) (d) (a) (b) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_3.jpg?x=305&y=1173&w=1041&h=256&r=0"/>

Figure 4. Cycles in \( {H}_{1} - \{ u\} \)

<!-- Media -->

Suppose \( {r}_{1} = 3 \) . Then \( {r}_{2} = {r}_{3} = 1 \) and so, \( r = 5 \) . Let \( {F}_{1} = \{ u,v,w\} ,{F}_{2} = \{ x\} \) and \( {F}_{3} = \{ y\} \) . As \( {F}_{1} \) has three vertices, \( x \) and \( y \) do not correspond to at least one of them. We may assume that \( u \) is such a vertex. Let \( {u}^{\prime } \) and \( {u}^{\prime \prime } \) be the vertices corresponding to \( u \) in \( {H}_{2} \) and \( {H}_{3} \) ,respectively. Then they are different from \( x \) and \( y \) . Then \( Z = \left\langle  {u,{u}^{\prime },{u}^{\prime \prime },u}\right\rangle \) is a 3-cycle in \( T \) . By Lemma 2.1,the subgraphs \( {H}_{2} - \left\{  {u}^{\prime }\right\} \) and \( {H}_{3} - \left\{  {u}^{\prime \prime }\right\} \) of \( T \) contain spanning cycles \( {Z}_{x} \) and \( {Z}_{y} \) , respectively. It is easy to see that there exists two vertex-disjoint cycles in \( {H}_{1} - \{ u\} \) ,one is a 3-cycle \( {Z}_{w} \) and other is a 5-cycle \( {Z}_{v} \) ,containing \( v \) and \( w \) ,respectively; see Figure 4. As \( {H}_{1} \) has nine vertices, \( {Z}_{v} \cup  {Z}_{w} \) spans \( {H}_{1} - \{ u\} \) . Thus \( \mathcal{C} = \left\{  {Z,{Z}_{v},{Z}_{w},{Z}_{x},{Z}_{y}}\right\} \) is a required collection of five vertex-disjoint cycles that together span \( T \) .

## 3. General case

In this section, we prove Theorem 1.2. For convenience, we restate the theorem here.

Theorem 3.1. For integers \( n \) and \( r \) with \( 1 \leq  r \leq  {2n} - 1 \) ,every \( n \) -dimensional torus,not isomorphic to \( {C}_{3}\square {C}_{3} \) ,is spanning \( r \) -cyclable.

Proof. The proof proceeds by induction on \( n \) . The result holds trivially for \( n = 1 \) . Suppose \( n \geq  2 \) . Assume that the result holds for \( n - 1 \) . Let \( T \) be an \( n \) -dimensional torus. Then \( T \) is the Cartesian product of \( n \) cycles, say \( {C}_{{k}_{1}},{C}_{{k}_{2}},\ldots ,{C}_{{k}_{n}} \) . We may assume that \( {k}_{1} \geq  {k}_{2} \geq  \cdots  \geq  {k}_{n} \geq  3 \) . Let \( H = {C}_{{k}_{1}}\square {C}_{{k}_{2}}\square \cdots \square {C}_{{k}_{n - 1}} \) . Then \( T = H\square {C}_{{k}_{n}} \) . If \( H \) is isomorphic to \( {C}_{3}\square {C}_{3} \) ,then \( T = {C}_{3}\square {C}_{3}\square {C}_{3} \) and so,by Lemma 2.8, \( T \) is spanning \( r \) -cyclable for \( 1 \leq  r \leq  5 = {2n} - 1 \) .

Suppose \( H \) is not isomorphic to \( {C}_{3}\square {C}_{3} \) . Label the vertices of the cycle \( {C}_{{k}_{n}} \) so that \( {C}_{{k}_{n}} = \left\langle  {1,2,\ldots ,{k}_{n},1}\right\rangle \) . Note that the graph \( T \) is obtained by replacing vertex \( i \) of \( {C}_{{k}_{n}} \) by a copy \( {H}_{i} \) of \( H \) and replacing each edge \( \langle i,i + 1\rangle \) by the perfect matching \( {M}_{i} \) between \( {H}_{i} \) and \( {H}_{i + 1} \) ,where the sum is taken modulo \( {k}_{n} \) . Thus

\[T = {H}_{1} \cup  {H}_{2} \cup  \cdots  \cup  {H}_{{k}_{n}} \cup  \left( {{M}_{1} \cup  {M}_{2} \cup  \cdots  \cup  {M}_{{k}_{n}}}\right) .\]

Let \( F = \left\{  {{v}_{1},{v}_{2},\ldots ,{v}_{r}}\right\} \) be an arbitrary set of \( r \) vertices of \( T \) . We prove that \( T \) has a collection \( \mathcal{C} \) of \( r \) vertex-disjoint cycles that together span \( T \) and each cycle contains exactly one \( {v}_{i} \) . Let \( {F}_{i} = F \cap  V\left( {H}_{i}\right) \) and \( \left| {F}_{i}\right|  = {r}_{i} \) for \( i = 1,2,\ldots ,{k}_{n} \) . Then \( {r}_{1} + {r}_{2} + \cdots  + {r}_{{k}_{n}} = r \) . We may assume that \( {r}_{1} \geq  {r}_{i} \) for \( i = 1,2,\ldots ,{k}_{n} \) . Then \( 1 \leq  {r}_{1} \leq  r \leq  {2n} - 1 \) .

We consider three cases depending upon \( {r}_{1} \) is \( {2n} - 1 \) or \( {2n} - 2 \) or less than \( {2n} - 2 \) .

Case 1. \( {r}_{1} = {2n} - 1 \) .

In this case, \( r = {r}_{1} \) and \( {H}_{1} \) contains all the vertices belonging to the set \( F \) . So, \( {F}_{1} = F \) . Let \( {r}^{\prime } = 2\left( {n - 1}\right)  - 1 \) . By induction, \( {H}_{1} \) is spanning \( {r}^{\prime } \) -cyclable. Hence \( {H}_{1} \) has \( {r}^{\prime } \) vertex-disjoint cycles \( {Z}_{1},{Z}_{2},\ldots ,{Z}_{{r}^{\prime }} \) whose union spans \( {H}_{1} \) and each cycle contains one vertex from \( F - \left\{  {{v}_{1},{v}_{2}}\right\} \) . Of these \( {r}^{\prime } \) cycles, \( {v}_{1} \) and \( {v}_{2} \) belong to a single cycle or two cycles. Accordingly, we make two subcases.

## Subcase 1.1. \( {v}_{1} \) and \( {v}_{2} \) belong to two cycles.

We may assume that \( {Z}_{1} \) contains \( {v}_{1} \) and \( {Z}_{2} \) contains \( {v}_{2} \) . Let \( x \) and \( y \) be the vertices of \( F - \left\{  {{v}_{1},{v}_{2}}\right\} \) belonging to \( {Z}_{1} \) and \( {Z}_{2} \) ,respectively. To seperate out \( x \) from \( {v}_{1} \) ,and \( y \) from \( {v}_{2} \) ,we split these cycles into paths. More precisely,for \( i = 1,2 \) ,we split the cycle \( {Z}_{i} \) into two vertex-disjoint subpaths \( {P}_{i} \) and \( {Q}_{i} \) such that \( {P}_{i} \cup  {Q}_{i} \) spans \( {Z}_{i} \) and further, \( {v}_{1},{v}_{2},x \) ,and \( y \) belong to \( {P}_{1},{P}_{2},{Q}_{1} \) and \( {Q}_{2} \) ,respectively; see Figure 5. By Corollary 2.4,the subgraphs \( {P}_{1}\square {C}_{{k}_{n}},{P}_{2}\square {C}_{{k}_{n}},{Q}_{1}\square {C}_{{k}_{n}} \) and \( {Q}_{2}\square {C}_{{k}_{n}} \) of \( T \) have spanning cycles \( {Z}_{{v}_{1}},{Z}_{{v}_{2}},{Z}_{x} \) and \( {Z}_{y} \) containing \( {v}_{1},{v}_{2},x \) and \( y \) ,respectively. Similarly,we extend the cycles \( {Z}_{3},{Z}_{4},\ldots ,{Z}_{{r}^{\prime }} \) so as to cover the corresponding vertices of the subgraphs \( {H}_{2},{H}_{3},\ldots ,{H}_{{k}_{n}} \) . By Corollary 2.5,the subgraph \( {Z}_{i}\square {C}_{{k}_{n}} \) of \( T \) has a spanning cycle \( {Z}_{i}^{\prime } \) for \( i = 3,4,\ldots ,{r}^{\prime } \) . Each \( {Z}_{i}^{\prime } \) contains only one vertex from \( F - \left\{  {{v}_{1},{v}_{2}}\right\} \) . Thus \( \mathcal{C} = \left\{  {{Z}_{{v}_{1}},{Z}_{{v}_{2}},{Z}_{x},{Z}_{y},{Z}_{3}^{\prime },{Z}_{4}^{\prime },\ldots ,{Z}_{{r}^{\prime }}^{\prime }}\right\} \) is a required collection of \( r \) vertex-disjoint cycles spanning \( T \) .

<!-- Media -->

<!-- figureText: \( {P}_{1} \) | ---------------@ \( {Z}_{x} \) \( {Z}_{{r}^{\prime }}^{\prime } \) \( {H}_{{k}_{n - 1}} \) \( {H}_{{k}_{n}} \) \( {Q}_{1} \) \( {P}_{2} \) \( {Q}_{2} \) \( {Z}_{y} \) \( {H}_{1} \) \( {H}_{2} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_4.jpg?x=145&y=1504&w=1356&h=404&r=0"/>

Figure 5. Cycles in \( T \) when \( {v}_{1} \) is in \( {Z}_{1} \) and \( {v}_{2} \) in \( {Z}_{2} \)

<!-- Media -->

Subcase 1.2. \( \;{v}_{1} \) and \( {v}_{2} \) belong to the cycle \( {Z}_{i} \) for some \( i \) .

Without loss of generality,we may assume that the cycle \( {Z}_{1} \) contains the vertices \( {v}_{1},{v}_{2} \) and one more vertex \( x \) of \( F \) . Let \( P,Q,R \) be vertex-disjoint subpaths in \( {Z}_{1} \) containing \( {v}_{1},{v}_{2},x \) ,respectively; see Figure 6. By Corollary 2.4, the subgraphs \( P\square {C}_{{k}_{n}},Q\square {C}_{{k}_{n}} \) and \( R\square {C}_{{k}_{n}} \) of \( T \) have spanning cycles \( {Z}_{{v}_{1}},{Z}_{{v}_{2}} \) and \( {Z}_{x} \) containing \( {v}_{1},{v}_{2} \) and \( x \) ,respectively. Similarly,we extend the cycles \( {Z}_{i} \) with \( i \neq  1 \) to cover the corresponding vertices of \( {H}_{2},{H}_{3},\ldots ,{H}_{{k}_{n}} \) . By Corollary 2.5,the subgraph \( {Z}_{i}\square {C}_{{k}_{n}} \) of \( T \) has a spanning cycle \( {Z}_{i}^{\prime } \) for \( i = 2,3,\ldots ,{k}_{n} \) . Thus \( \mathcal{C} = \left\{  {{Z}_{{v}_{1}},{Z}_{{v}_{2}},{Z}_{x},{Z}_{2}^{\prime },{Z}_{3}^{\prime },\ldots ,{Z}_{{r}^{\prime }}^{\prime }}\right\} \) is a collection of \( r \) vertex-disjoint cycles whose union spans the graph \( T \) .

<!-- Media -->

<!-- figureText: \( {Z}_{x} \) \( {Z}_{{r}^{\prime }}^{\prime } \) \( {H}_{{k}_{n - 1}} \) \( {H}_{{k}_{n}} \) \( {Z}_{2} \) \( {Z}_{2}^{\prime } \) \( {H}_{1} \) \( {H}_{2} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_5.jpg?x=146&y=501&w=1356&h=398&r=0"/>

Figure 6. Cycles in \( T \) when \( {v}_{1},{v}_{2} \) are in \( {Z}_{1} \)

<!-- Media -->

Case 2. \( {r}_{1} = {2n} - 2 \) .

In this case, \( r = {r}_{1} \) or \( r = {r}_{1} + 1 \) . We may assume that \( {F}_{1} = \left\{  {{v}_{1},{v}_{2},\ldots ,{v}_{{2n} - 2}}\right\} \) . Let \( {r}_{1}^{\prime } = {r}_{1} - 1 = 2\left( {n - 1}\right)  - 1 \) . By induction, \( {H}_{1} \) is spanning \( {r}_{1}^{\prime } \) -cyclable. Thus \( {H}_{1} \) contains \( {r}_{1}^{\prime } \) vertex-disjoint cycles \( {Z}_{1},{Z}_{2},\ldots ,{Z}_{{r}_{1}^{\prime }} \) that together span \( {H}_{1} \) and each cycle contains exactly one vertex from the set \( {F}_{1} - \left\{  {v}_{1}\right\} \) . The remaining vertex \( {v}_{1} \) of \( F \) belongs to one of these cycles. Without loss of generality,assume that \( {Z}_{1} \) contains two vertices of \( {F}_{1} \) ,say \( {v}_{1} \) and \( {v}_{2} \) . We split the cycle \( {Z}_{1} \) into two vertex-disjoint subpaths \( P \) and \( Q \) in a way that \( P \) contains \( {v}_{1} \) and \( Q \) contains \( {v}_{2} \) ,and \( P \cup  Q \) spans \( {Z}_{1} \) .

Suppose \( r = {r}_{1} \) . Then \( {F}_{i} = \varnothing \) for \( i = 2,3,\ldots ,{k}_{n} \) . By Corollary 2.4 and Corollary 2.5,the subgraphs \( P\square {C}_{{k}_{n}} \) . \( Q\square {C}_{{k}_{n}},{Z}_{j}\square {C}_{{k}_{n}} \) of \( T \) for \( j = 2,3,\ldots ,{r}_{1}^{\prime } \) are vertex-disjoint Hamiltonian graphs. The collection \( \mathcal{C} \) consisting of one Hamiltonian cycle from each of these \( 2 + \left( {{r}_{1}^{\prime } - 1}\right)  = {r}_{1} = r \) graphs forms a required collection of cycles for spanning \( r \) -cyclability of the graph \( T \) .

Suppose \( r = {r}_{1} + 1 \) . Then \( {r}_{t} = 1 \) for some \( t \neq  1 \) and \( {r}_{j} = 0 \) for \( j \notin  \{ 1,t\} \) . Then \( {F}_{t} = F \smallsetminus  {F}_{1} \) consists of only one vertex,say \( w \) . As \( {k}_{n} \geq  3 \) ,either \( {r}_{2} = 0 \) or \( {r}_{{k}_{n}} = 0 \) . Due to symmetry,we may assume that \( {r}_{2} = 0 \) . Hence \( t \geq  3 \) . Let \( {L}_{1} = \langle 1,2\rangle \) and \( {L}_{2} = \left\langle  {3,4,\ldots ,{k}_{n}}\right\rangle \) be subpaths in the cycle \( {C}_{{k}_{n}} \) . Then \( {L}_{1} \cup  {L}_{2} \) spans \( {C}_{{k}_{n}} \) . By Corollary 2.4, the subgraph \( {Z}_{i}\square {L}_{1} \) of \( T \) has a Hamiltonian cycle \( {Z}_{i}^{\prime } \) for \( i = 2,3,\ldots ,{r}_{1}^{\prime } \) . Then each \( {Z}_{i}^{\prime } \) contains exactly one vertex from \( {F}_{1} - \left\{  {{v}_{1},{v}_{2}}\right\} \) . We split the cycle \( {Z}_{1} \) into two vertex-disjoint subpaths one containing \( {v}_{1} \) and the other \( {v}_{2} \) .

Subcase 2.1. Suppose \( {Z}_{1} \) is not a triangle.

We can choose subpaths \( P \) and \( Q \) of \( {Z}_{1} \) as non-trivial paths. As \( {L}_{1} \) contains even number of vertices,by Lemma 2.2, the subgraphs \( P\square {L}_{1},Q\square {L}_{1} \) of \( T \) have spanning cycles,say \( {Z}_{1}^{\prime } \) and \( {Z}_{1}^{\prime \prime } \) ,respectively. Then \( {v}_{1} \) belongs to \( {Z}_{1}^{\prime } \) and \( {v}_{2} \) belongs to \( {Z}_{1}^{\prime \prime } \) . By Corollary 2.5, \( {H}_{t} \) has a Hamiltonian cycle \( C \) . Then,by Corollary 2.4,the subgraph \( C\square {L}_{2} \) of \( T \) has a spanning cycle \( Z \) . Hence the cycle \( Z \) spans \( {H}_{3},{H}_{4},\ldots ,{H}_{{k}_{n}} \) and contains only vertex \( w \) from the set \( F \) . Thus, \( \mathcal{C} = \left\{  {Z,{Z}_{1}^{\prime },{Z}_{1}^{\prime \prime },{Z}_{2}^{\prime },{Z}_{3}^{\prime },\ldots ,{Z}_{{r}_{1}^{\prime }}^{\prime }}\right\} \) is a collection of \( 3 + {r}_{1}^{\prime } - 1 = {r}_{1} + 1 = r \) vertex-disjoint cycles whose union spans \( T \) ; see Figure 7.

Subcase 2.2. Suppose \( {Z}_{1} \) is a triangle.

Let \( {Z}_{1} = \left\langle  {{v}_{1},{v}_{2},x,{v}_{1}}\right\rangle \) be a 3-cycle in \( {H}_{1} \) . We may assume that the vertex \( w \) belonging to \( {F}_{t} \) does not correspond to the vertex \( {v}_{1} \) . Split the cycle \( {Z}_{1} \) into two vertex-disjoint subpaths \( P \) and \( Q \) ,where \( P \) consists of the vertex \( {v}_{1} \) only and \( Q = {Z}_{1} - \left\{  {v}_{1}\right\} \) . Then \( P\square {C}_{{k}_{n}} \) is a cycle,say \( {Z}_{1}^{\prime } \) ,in \( T \) containing \( {v}_{1} \) . Let \( {v}_{1}^{\prime } \) be the vertex of \( {H}_{t} \) corresponding to \( {v}_{1} \) . Then \( {v}_{1}^{\prime } \neq  w \) and it belongs \( {Z}_{1}^{\prime } \) . Note that \( Q\square {L}_{1} \) is a 4-cycle \( {Z}_{1}^{\prime \prime } \) containing \( {v}_{2} \) .

<!-- Media -->

<!-- figureText: \( {L}_{1} \) \( {L}_{2} \) \( t \) \( C \) \( Z \) \( {H}_{t} \) \( {H}_{{k}_{n}} \) 2 3 \( {Z}_{1}^{\prime } \) \( {Z}_{2}^{\prime } \) \( {Z}_{{r}^{\prime }}^{\prime } \) \( {Z}_{{r}_{1}^{\prime }} \) \( {H}_{1} \) \( {H}_{2} \) \( {H}_{3} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_6.jpg?x=240&y=280&w=1167&h=451&r=0"/>

Figure 7. Cycles in \( T \) when \( {Z}_{1} \) is not a triangle

<!-- Media -->

Suppose \( n = 2 \) . Then \( r = 3 \) and \( H \) is a cycle. Hence \( {H}_{1} \) is a cycle. Therefore \( H = {Z}_{1} = {C}_{3} \) and so \( T = {C}_{3}\square {C}_{3} \) , a contradiction. Suppose \( n \geq  3 \) . Then \( H \) is the Cartesian product of \( n - 1 \geq  2 \) cycles. Since \( {H}_{t} \) contains a triangle corresponding to \( {Z}_{1} \) ,it is a non-bipartite(n - 1)-dimensional torus. By Lemma 2.1,there is a cycle \( C \) in \( {H}_{t} \) containing all vertices of \( {H}_{t} \) except \( {v}_{1}^{\prime } \) . Then,by Corollary 2.4,the subgraph \( C\square {L}_{2} \) of \( T \) has a Hamiltonian cycle,say \( Z \) . Then \( Z \) contains \( w \) . Thus \( \mathcal{C} = \left\{  {Z,{Z}_{1}^{\prime },{Z}_{1}^{\prime \prime },{Z}_{2}^{\prime },{Z}_{3}^{\prime },\ldots ,{Z}_{{r}_{1}^{\prime }}}\right\} \) is a required collection of \( r \) vertex-disjoint cycles of \( T \) ; see Figure 8.

<!-- Media -->

<!-- figureText: \( {Z}_{1}^{\prime } \) \( C \) \( Z \) \( {H}_{t} \) \( {H}_{{k}_{n}} \) \( {Z}_{2} \) \( {Z}_{2}^{{}^{\prime }} \) \( {H}_{1} \) \( {H}_{2} \) \( {H}_{3} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_6.jpg?x=240&y=1116&w=1168&h=347&r=0"/>

Figure 8. Cycles in \( T \) when \( {Z}_{1} \) is a triangle

<!-- Media -->

Case 3. \( 1 \leq  {r}_{1} \leq  {2n} - 3 \) .

Note that \( {r}_{i} \leq  {r}_{1} \leq  {2n} - 3 = 2\left( {n - 1}\right)  - 1 \) for all \( i \) . We form a collection \( \mathcal{C} \) of \( r \) vertex-disjoint cycles each containing one vertex from the set \( F \) and the union of these cycles spans \( T \) .

If \( {r}_{i} \neq  0 \) for some \( i \) ,then,by induction,we get a collection \( {\mathcal{C}}_{i} \) of \( {r}_{i} \) vertex-disjoint cycles that span \( {H}_{i} \) and each cycle contains one vertex from the set \( {F}_{i} \) . In addition,if \( {r}_{i + 1} \) is also non-zero,then we include the collection \( {\mathcal{C}}_{i} \) as i is to form the collection \( \mathcal{C} \) . Thus if \( {r}_{i} \neq  0 \) for all \( i \) ,then \( \mathcal{C} = {\mathcal{C}}_{1} \cup  {\mathcal{C}}_{2} \cup  \cdots  \cup  {\mathcal{C}}_{{k}_{n}} \) is a required collection.

Suppose \( {r}_{i} \neq  0 \) but \( {r}_{i + 1} = 0 \) for some \( i \) . In this case,we modify the cycles in \( {\mathcal{C}}_{i} \) to span the subgraph \( {H}_{i + 1} \) and include this modified collection to form the collection \( \mathcal{C} \) . We modify the collection \( {\mathcal{C}}_{i} \) as follows. Suppose for some \( i \) ,we have \( {r}_{i} \neq  0 \) but \( {r}_{j} = 0 \) for \( j = i + 1,i + 2,\ldots ,i + t \) and \( {r}_{i + t + 1} \neq  0 \) for some integer \( t \geq  1 \) . Let \( {\mathcal{C}}_{i} = \left\{  {{Z}_{1},{Z}_{2},\ldots ,{Z}_{{r}_{i}}}\right\} \) and let \( L = \langle i,i + 1,\ldots ,i + t\rangle \) be a subpath of cycle \( {C}_{{k}_{n}} \) from \( i \) to \( i + t \) . By Corollary 2.4,the subgraph \( {Z}_{k}\square L \) of \( T \) has a spanning cycle,say \( {Z}_{k}^{\prime } \) ,for \( 1 \leq  k \leq  {r}_{i} \) . Thus \( {\mathcal{C}}_{i}^{\prime } = \left\{  {{Z}_{1}^{\prime },{Z}_{2}^{\prime },\ldots ,{Z}_{{r}_{i}}^{\prime }}\right\} \) is a collection of vertex-disjoint cycles in \( T \) such that each cycle of it contains exactly one vertex from the set \( {F}_{i} \) ,and further \( {Z}_{1}^{\prime } \cup  {Z}_{2}^{\prime } \cup  \cdots  \cup  {Z}_{{r}_{i}}^{\prime } \) is a spanning subgraph of \( {H}_{i} \cup  {H}_{i + 1} \cup  \cdots  \cup  {H}_{i + t} \) . In this case,we include the collection \( {\mathcal{C}}_{i}^{\prime } \) to form the collection \( \mathcal{C} \) .

<!-- Media -->

<!-- figureText: \( {Z}_{1} \) \( {z}_{1}^{\prime } \) \( {H}_{3} \) \( {H}_{j + 1} \) \( {H}_{{k}_{n}} \) \( {Z}_{2}^{\prime } \) \( {Z}_{{r}_{1}} \) \( {Z}_{{r}_{1}}^{\prime } \) \( {H}_{1} \) \( {H}_{2} \) -->

<img src="https://cdn.noedgeai.com/01969098-4273-7339-875e-383544b06c24_7.jpg?x=242&y=279&w=1163&h=354&r=0"/>

Figure 9. \( {H}_{1} \) and \( {H}_{j + 1} \) contains vertices of \( F \)

<!-- Media -->

As \( {r}_{1} \) is non-zero,we include the collection \( {\mathcal{C}}_{1} \) or its modified collection \( {\mathcal{C}}_{1}^{\prime } \) to form \( \mathcal{C} \) . If \( {r}_{j} \) from the list of \( \left\{  {{r}_{2},{r}_{3},\ldots ,{r}_{{k}_{n}}}\right\} \) is non-zero,then we include \( {\mathcal{C}}_{j} \) or \( {\mathcal{C}}_{j}^{\prime } \) to \( \mathcal{C} \) . Continuing this way,we form the required collection \( \mathcal{C} \) ; see Figure 9.

This completes the proof.

## Acknowledgement

The first author acknowledges the Council of Scientific and Industrial Research, India, for Senior Research Fellowship, file number 09/137(0572)/2017-EMR-I. The second author is supported by DST-SERB, Government of India by Project MTR/2018/000447.

## References

[1] V. Batagelj and T. Pisanski, Hamiltonian cycles in the Cartesian product of a tree and a cycle, Discrete Math., 38 (1982) 311-312.

[2] S. Chiba and T. Yamashita, Degree conditions for the existence of vertex-disjoint cycles and paths: A Survey, Graphs Combin, 34 (2018) 1-83.

[3] K. Corrádi and A. Hajnal, On the maximal number of independent circuits in graph, Acta Math. Acad. Sci. Hungar., 14 (1963) 423-439.

[4] Y. Egawa, H. Enomoto, R. Faudree, H. Li and I. Schiermeyer, Two-factors each component of which contains a specified vertex, J. Graph Theory, 43 (2003) 188-198.

[5] R. Gould, Recent advances on the Hamiltonian problem: Survey III, Graphs Combin., 30 (2003) 1-46.

[6] R. Gould, A look at cycles containing specified elements of a graph, Discrete Math., 309 (2009) 6299-6311.

[7] Y. Ishigami and T. Jiang, Vertex-disjoint cycles containing prescribed vertices, J. Graph Theory, 42 (2003) 276-296.

[8] H. Kim and J. Park, Paths and cycles in \( d \) -dimensional tori with faults,in Proc. of Workshop on Algorithms and Computation WAAC'01, Pusan, Korea (2001) 67-74.

[9] T. Kung, C. Hung, C. Lin, H. Chen, C. Lin and L. Hsu, A Framework of cycle-based clustering on the crossed cube architecture, 2016 10th International Conference on Innovative Mobile and Internet Services in Ubiquitous Computing (IMIS), Fukuoka, 1 (2016) 430-434.

[10] F. Leighton, Introduction to Parallel Algorithms and Architectures: Arrays, Trees, Hypercubes, M. Kaufmann Publishers Inc., San Mateo, California (1992).

[11] C. Lin, J. Tana, L. Hsu and T. Kung, Disjoint cycles in hypercubes with prescribed vertices in each cycle, Discrete Appl. Math., 161 (2013) 2992-3004.

[12] S. Mane and B. Waphare, Regular connected bipancyclic spanning subgraphs of hypercubes, Comput. Math. Appl., 62(9) (2011) 3551-3554.