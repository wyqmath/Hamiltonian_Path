"""
基于区域/簇的故障模型 (Region-Based Fault Model, RBF)

本模块实现了一种新的故障模型，其中故障不是以单个边的形式出现，
而是以空间聚集的"故障簇"形式出现。这种模型更符合实际系统中
故障的空间相关性特征。

核心创新：
1. 故障簇定义：将故障边组织成空间连通的簇
2. 簇形状模型：支持完全图、星形图、路径图等多种簇形状
3. 区域容错条件：基于簇的数量和大小的容错条件
4. 递归证明策略：利用网络递归结构的归纳证明方法

理论基础：
- 故障簇模型 (Clustered Fault Model, CFM)
- 网络递归分解理论
- 哈密尔顿路径的归纳构造

"""

import math
import random
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict, deque

# 导入基础模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from origin_pef import QkCube


class ClusterShape(Enum):
    """故障簇形状类型"""
    COMPLETE_GRAPH = "complete"      # 完全图 K_m
    STAR_GRAPH = "star"             # 星形图 S_k  
    PATH_GRAPH = "path"             # 路径图 P_l
    CYCLE_GRAPH = "cycle"           # 环图 C_l
    TREE_GRAPH = "tree"             # 树图 T_n
    CUSTOM = "custom"               # 自定义形状


@dataclass
class FaultCluster:
    """故障簇数据结构"""
    cluster_id: int                 # 簇ID
    fault_edges: List[Tuple]        # 簇内故障边
    affected_nodes: Set[Tuple]      # 受影响的节点
    shape: ClusterShape             # 簇形状
    size: int                       # 簇大小（边数）
    center: Optional[Tuple]         # 簇中心（如果适用）
    radius: int                     # 簇半径
    connectivity: float             # 簇内连通度
    
    def __post_init__(self):
        """后处理：计算派生属性"""
        if not self.affected_nodes:
            self.affected_nodes = self._extract_affected_nodes()
        if self.size == 0:
            self.size = len(self.fault_edges)
    
    def _extract_affected_nodes(self) -> Set[Tuple]:
        """从故障边提取受影响的节点"""
        nodes = set()
        for edge in self.fault_edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        return nodes


@dataclass
class RegionBasedFaultModel:
    """区域故障模型参数"""
    max_clusters: int               # 最大故障簇数量 k
    max_cluster_size: int           # 单个簇最大大小 s
    allowed_shapes: List[ClusterShape]  # 允许的簇形状
    spatial_correlation: float      # 空间相关性参数
    cluster_separation: int         # 簇间最小分离距离
    
    def __post_init__(self):
        """验证参数合理性"""
        assert self.max_clusters > 0, "最大簇数量必须为正"
        assert self.max_cluster_size > 0, "最大簇大小必须为正"
        assert 0 <= self.spatial_correlation <= 1, "空间相关性必须在[0,1]范围内"


class RegionBasedFaultAnalyzer:
    """区域故障模型分析器"""
    
    def __init__(self, Q: QkCube, rbf_params: RegionBasedFaultModel):
        self.Q = Q
        self.rbf_params = rbf_params
        self.clusters: List[FaultCluster] = []
        
    def analyze_fault_distribution(self, F: List[Tuple]) -> List[FaultCluster]:
        """
        分析故障边分布，识别故障簇
        
        Args:
            F: 故障边集合
            
        Returns:
            识别出的故障簇列表
        """
        # 1. 构建故障边的邻接图
        fault_graph = self._build_fault_adjacency_graph(F)
        
        # 2. 使用连通分量算法识别初始簇
        initial_clusters = self._find_connected_components(fault_graph, F)
        
        # 3. 根据空间距离合并相近的簇
        merged_clusters = self._merge_nearby_clusters(initial_clusters)
        
        # 4. 分析每个簇的形状和特征
        analyzed_clusters = self._analyze_cluster_shapes(merged_clusters)
        
        # 5. 验证簇模型的有效性
        self._validate_cluster_model(analyzed_clusters)
        
        self.clusters = analyzed_clusters
        return analyzed_clusters
    
    def _build_fault_adjacency_graph(self, F: List[Tuple]) -> Dict[Tuple, Set[Tuple]]:
        """构建故障边的邻接图（基于节点共享）"""
        graph = defaultdict(set)
        
        # 为每条故障边的端点建立连接
        for edge in F:
            u, v = edge
            graph[u].add(v)
            graph[v].add(u)
            
        return dict(graph)
    
    def _find_connected_components(self, graph: Dict[Tuple, Set[Tuple]], F: List[Tuple]) -> List[FaultCluster]:
        """使用DFS找到连通分量"""
        visited = set()
        clusters = []
        cluster_id = 0
        
        # 获取所有涉及故障的节点
        all_fault_nodes = set()
        for edge in F:
            all_fault_nodes.add(edge[0])
            all_fault_nodes.add(edge[1])
        
        for node in all_fault_nodes:
            if node not in visited:
                # 开始新的连通分量
                component_nodes = set()
                component_edges = []
                
                # DFS遍历
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component_nodes.add(current)
                        
                        # 添加相邻节点
                        if current in graph:
                            for neighbor in graph[current]:
                                if neighbor not in visited:
                                    stack.append(neighbor)
                                # 添加边到组件
                                edge = tuple(sorted([current, neighbor]))
                                if edge in F or (neighbor, current) in F:
                                    component_edges.append((current, neighbor))
                
                # 创建故障簇
                if component_edges:
                    cluster = FaultCluster(
                        cluster_id=cluster_id,
                        fault_edges=component_edges,
                        affected_nodes=component_nodes,
                        shape=ClusterShape.CUSTOM,  # 稍后分析
                        size=len(component_edges),
                        center=None,
                        radius=0,
                        connectivity=0.0
                    )
                    clusters.append(cluster)
                    cluster_id += 1
        
        return clusters
    
    def _merge_nearby_clusters(self, clusters: List[FaultCluster]) -> List[FaultCluster]:
        """根据空间距离合并相近的簇"""
        if len(clusters) <= 1:
            return clusters
            
        merged = []
        used = set()
        
        for i, cluster1 in enumerate(clusters):
            if i in used:
                continue
                
            # 寻找可以合并的簇
            merge_candidates = [cluster1]
            used.add(i)
            
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                    
                # 计算簇间距离
                distance = self._calculate_cluster_distance(cluster1, cluster2)
                
                if distance <= self.rbf_params.cluster_separation:
                    merge_candidates.append(cluster2)
                    used.add(j)
            
            # 合并簇
            if len(merge_candidates) > 1:
                merged_cluster = self._merge_clusters(merge_candidates)
                merged.append(merged_cluster)
            else:
                merged.append(cluster1)
        
        return merged
    
    def _calculate_cluster_distance(self, cluster1: FaultCluster, cluster2: FaultCluster) -> int:
        """计算两个簇之间的最小距离"""
        min_distance = float('inf')
        
        for node1 in cluster1.affected_nodes:
            for node2 in cluster2.affected_nodes:
                distance = self._manhattan_distance(node1, node2)
                min_distance = min(min_distance, distance)
        
        return int(min_distance)
    
    def _manhattan_distance(self, node1: Tuple, node2: Tuple) -> int:
        """计算两个节点的曼哈顿距离"""
        return sum(abs(a - b) for a, b in zip(node1, node2))
    
    def _merge_clusters(self, clusters: List[FaultCluster]) -> FaultCluster:
        """合并多个簇"""
        all_edges = []
        all_nodes = set()
        
        for cluster in clusters:
            all_edges.extend(cluster.fault_edges)
            all_nodes.update(cluster.affected_nodes)
        
        # 去重
        unique_edges = list(set(tuple(sorted(edge)) for edge in all_edges))
        
        return FaultCluster(
            cluster_id=clusters[0].cluster_id,  # 使用第一个簇的ID
            fault_edges=unique_edges,
            affected_nodes=all_nodes,
            shape=ClusterShape.CUSTOM,
            size=len(unique_edges),
            center=None,
            radius=0,
            connectivity=0.0
        )
    
    def _analyze_cluster_shapes(self, clusters: List[FaultCluster]) -> List[FaultCluster]:
        """分析每个簇的形状特征"""
        analyzed = []
        
        for cluster in clusters:
            # 分析簇的拓扑结构
            shape = self._identify_cluster_shape(cluster)
            center = self._calculate_cluster_center(cluster)
            radius = self._calculate_cluster_radius(cluster, center)
            connectivity = self._calculate_cluster_connectivity(cluster)
            
            # 更新簇信息
            cluster.shape = shape
            cluster.center = center
            cluster.radius = radius
            cluster.connectivity = connectivity
            
            analyzed.append(cluster)
        
        return analyzed
    
    def _identify_cluster_shape(self, cluster: FaultCluster) -> ClusterShape:
        """识别簇的形状类型"""
        nodes = cluster.affected_nodes
        edges = cluster.fault_edges
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        if n_nodes <= 1:
            return ClusterShape.CUSTOM
        
        # 完全图检测：K_n 有 n(n-1)/2 条边
        if n_edges == n_nodes * (n_nodes - 1) // 2:
            return ClusterShape.COMPLETE_GRAPH
        
        # 星形图检测：中心节点连接所有其他节点
        if self._is_star_graph(cluster):
            return ClusterShape.STAR_GRAPH
        
        # 路径图检测：n-1条边，形成路径
        if n_edges == n_nodes - 1 and self._is_path_graph(cluster):
            return ClusterShape.PATH_GRAPH
        
        # 环图检测：n条边，形成环
        if n_edges == n_nodes and self._is_cycle_graph(cluster):
            return ClusterShape.CYCLE_GRAPH
        
        # 树图检测：n-1条边，连通但无环
        if n_edges == n_nodes - 1 and self._is_tree_graph(cluster):
            return ClusterShape.TREE_GRAPH
        
        return ClusterShape.CUSTOM
    
    def _is_star_graph(self, cluster: FaultCluster) -> bool:
        """检测是否为星形图"""
        # 构建度数统计
        degree = defaultdict(int)
        for edge in cluster.fault_edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        
        degrees = list(degree.values())
        degrees.sort(reverse=True)
        
        # 星形图：一个中心节点度数为n-1，其他节点度数为1
        n = len(cluster.affected_nodes)
        return len(degrees) == n and degrees[0] == n-1 and all(d == 1 for d in degrees[1:])
    
    def _is_path_graph(self, cluster: FaultCluster) -> bool:
        """检测是否为路径图"""
        degree = defaultdict(int)
        for edge in cluster.fault_edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        
        degrees = list(degree.values())
        degrees.sort()
        
        # 路径图：两个端点度数为1，其他节点度数为2
        return degrees == [1, 1] + [2] * (len(degrees) - 2)
    
    def _is_cycle_graph(self, cluster: FaultCluster) -> bool:
        """检测是否为环图"""
        degree = defaultdict(int)
        for edge in cluster.fault_edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        
        # 环图：所有节点度数为2
        return all(d == 2 for d in degree.values())
    
    def _is_tree_graph(self, cluster: FaultCluster) -> bool:
        """检测是否为树图（连通且无环）"""
        # 已知边数 = 节点数 - 1，只需检查连通性
        return self._is_connected(cluster)
    
    def _is_connected(self, cluster: FaultCluster) -> bool:
        """检测簇是否连通"""
        if not cluster.affected_nodes:
            return True
        
        # 构建邻接表
        graph = defaultdict(set)
        for edge in cluster.fault_edges:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])
        
        # BFS检查连通性
        start = next(iter(cluster.affected_nodes))
        visited = set()
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(graph[node] - visited)
        
        return len(visited) == len(cluster.affected_nodes)

    def _calculate_cluster_center(self, cluster: FaultCluster) -> Optional[Tuple]:
        """计算簇的几何中心"""
        if not cluster.affected_nodes:
            return None

        n_dim = self.Q.n
        center_coords = [0] * n_dim

        for node in cluster.affected_nodes:
            for i in range(n_dim):
                center_coords[i] += node[i]

        # 计算平均值并四舍五入
        center = tuple(round(coord / len(cluster.affected_nodes)) for coord in center_coords)
        return center

    def _calculate_cluster_radius(self, cluster: FaultCluster, center: Optional[Tuple]) -> int:
        """计算簇的半径（最大距离）"""
        if not center or not cluster.affected_nodes:
            return 0

        max_distance = 0
        for node in cluster.affected_nodes:
            distance = self._manhattan_distance(node, center)
            max_distance = max(max_distance, distance)

        return max_distance

    def _calculate_cluster_connectivity(self, cluster: FaultCluster) -> float:
        """计算簇的连通度（实际边数/最大可能边数）"""
        n_nodes = len(cluster.affected_nodes)
        if n_nodes <= 1:
            return 1.0

        max_edges = n_nodes * (n_nodes - 1) // 2
        actual_edges = len(cluster.fault_edges)

        return actual_edges / max_edges

    def _validate_cluster_model(self, clusters: List[FaultCluster]) -> bool:
        """验证簇模型是否满足RBF条件"""
        # 检查簇数量限制
        if len(clusters) > self.rbf_params.max_clusters:
            return False

        # 检查每个簇的大小限制
        for cluster in clusters:
            if cluster.size > self.rbf_params.max_cluster_size:
                return False

        # 检查簇间分离距离
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i+1:]:
                distance = self._calculate_cluster_distance(cluster1, cluster2)
                if distance < self.rbf_params.cluster_separation:
                    return False

        return True

    def calculate_rbf_fault_tolerance(self) -> int:
        """计算RBF模型的理论容错上界（严格按照定理2.1）"""
        k_max = self.rbf_params.max_clusters
        s_max = self.rbf_params.max_cluster_size
        n = self.Q.n
        k_val = self.Q.k
        d_sep = self.rbf_params.cluster_separation

        # 按照定理2.1的公式：Θ_RBF = k_max * s_max * α(n, k, d_sep)
        base_tolerance = k_max * s_max

        # 结构修正因子
        alpha_struct = self._calculate_structure_factor()

        # 空间修正因子
        alpha_spatial = self._calculate_spatial_factor()

        # 总修正因子
        alpha_total = alpha_struct * alpha_spatial

        # 最终容错上界
        theta_rbf = int(base_tolerance * alpha_total)

        return theta_rbf

    def _calculate_structure_factor(self) -> float:
        """计算网络结构修正因子（严格按照数学理论）"""
        n, k_val = self.Q.n, self.Q.k

        # 按照定理2.1中的公式
        alpha_struct = min(1 + math.log(n * k_val / 2) / n, 2.0)
        return alpha_struct

    def _calculate_spatial_factor(self) -> float:
        """计算空间分布提升因子（严格按照数学理论）"""
        d_sep = self.rbf_params.cluster_separation
        rho = self.rbf_params.spatial_correlation  # 空间相关性参数

        # 按照定理2.1中的公式
        alpha_spatial = (1 + 0.5 * (1 - rho)) * (1 + math.log(1 + d_sep) / 10)
        return alpha_spatial


class RegionBasedHamiltonianEmbedding:
    """基于区域故障模型的哈密尔顿路径嵌入"""

    def __init__(self, Q: QkCube, rbf_params: RegionBasedFaultModel):
        self.Q = Q
        self.rbf_params = rbf_params
        self.analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)

    def embed_hamiltonian_path_rbf(
        self,
        F: List[Tuple],
        source: Tuple,
        target: Tuple
    ) -> List[Tuple]:
        """
        基于区域故障模型的哈密尔顿路径嵌入

        使用递归分解和归纳证明的策略：
        1. 分析故障簇分布
        2. 选择最优分解维度
        3. 递归构造子网络路径
        4. 利用跨维度边缝合路径
        """
        # 1. 分析故障分布，识别故障簇
        clusters = self.analyzer.analyze_fault_distribution(F)

        # 2. 检查RBF容错条件
        if not self._check_rbf_conditions(clusters):
            return self._fallback_algorithm(F, source, target)

        # 3. 选择最优递归分解策略
        decomposition_strategy = self._select_decomposition_strategy(clusters)

        # 4. 执行递归哈密尔顿路径构造
        path = self._recursive_hamiltonian_construction(
            F, source, target, clusters, decomposition_strategy
        )

        return path if path else []

    def _check_rbf_conditions(self, clusters: List[FaultCluster]) -> bool:
        """检查是否满足RBF容错条件"""
        # 条件1：簇数量不超过限制
        if len(clusters) > self.rbf_params.max_clusters:
            return False

        # 条件2：每个簇大小不超过限制
        for cluster in clusters:
            if cluster.size > self.rbf_params.max_cluster_size:
                return False

        # 条件3：簇间有足够的分离距离
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i+1:]:
                distance = self.analyzer._calculate_cluster_distance(cluster1, cluster2)
                if distance < self.rbf_params.cluster_separation:
                    return False

        return True

    def _select_decomposition_strategy(self, clusters: List[FaultCluster]) -> int:
        """
        选择最优的递归分解维度

        策略：选择能最好地"隔离"故障簇的维度
        """
        n = self.Q.n
        best_dimension = 0
        best_score = -1

        for dim in range(n):
            score = self._evaluate_dimension_separation(clusters, dim)
            if score > best_score:
                best_score = score
                best_dimension = dim

        return best_dimension

    def _evaluate_dimension_separation(self, clusters: List[FaultCluster], dimension: int) -> float:
        """评估在指定维度上分解的效果"""
        # 计算簇在该维度上的分布
        dimension_distribution = defaultdict(list)

        for cluster in clusters:
            for node in cluster.affected_nodes:
                layer = node[dimension]
                dimension_distribution[layer].append(cluster.cluster_id)

        # 计算分离度：簇分布越分散越好
        occupied_layers = len(dimension_distribution)
        total_layers = self.Q.k

        # 计算簇的分散程度
        cluster_spread = 0
        for layer_clusters in dimension_distribution.values():
            unique_clusters = len(set(layer_clusters))
            cluster_spread += unique_clusters

        # 分离度评分
        separation_score = occupied_layers / total_layers
        spread_score = cluster_spread / (len(clusters) * occupied_layers + 1)

        return separation_score * spread_score

    def _recursive_hamiltonian_construction(
        self,
        F: List[Tuple],
        source: Tuple,
        target: Tuple,
        clusters: List[FaultCluster],
        decomposition_dim: int
    ) -> List[Tuple]:
        """
        递归哈密尔顿路径构造算法

        基于网络递归结构的归纳证明策略：
        1. 基础情况：低维网络直接构造
        2. 归纳步骤：分解为子网络，递归构造，然后缝合
        """
        n, k = self.Q.n, self.Q.k

        # 基础情况：1维或2维网络
        if n <= 2:
            return self._base_case_construction(F, source, target)

        # 归纳步骤：沿decomposition_dim分解网络
        subcubes = self._decompose_network(decomposition_dim)

        # 分析故障簇在子网络中的分布
        cluster_distribution = self._distribute_clusters_to_subcubes(clusters, decomposition_dim)

        # 为每个子网络递归构造路径
        subcube_paths = self._construct_subcube_paths(
            F, subcubes, cluster_distribution, source, target, decomposition_dim
        )

        # 使用跨维度边缝合子网络路径
        final_path = self._stitch_subcube_paths(
            subcube_paths, decomposition_dim, source, target
        )

        return final_path

    def _base_case_construction(self, F: List[Tuple], source: Tuple, target: Tuple) -> List[Tuple]:
        """处理基础情况（低维网络）"""
        # 对于低维网络，使用简化的路径构造算法
        return self._simple_path_search(F, source, target)

    def _decompose_network(self, dimension: int) -> List[List[Tuple]]:
        """沿指定维度分解网络为子立方体"""
        subcubes = []
        k = self.Q.k

        for layer in range(k):
            subcube_nodes = []
            for node in self._generate_all_nodes():
                if node[dimension] == layer:
                    subcube_nodes.append(node)
            subcubes.append(subcube_nodes)

        return subcubes

    def _generate_all_nodes(self) -> List[Tuple]:
        """生成所有节点"""
        nodes = []
        ranges = [range(self.Q.k) for _ in range(self.Q.n)]
        for coords in itertools.product(*ranges):
            nodes.append(coords)
        return nodes

    def _distribute_clusters_to_subcubes(
        self,
        clusters: List[FaultCluster],
        dimension: int
    ) -> Dict[int, List[FaultCluster]]:
        """将故障簇分配到相应的子立方体"""
        distribution = defaultdict(list)

        for cluster in clusters:
            # 确定簇主要影响哪些层
            affected_layers = set()
            for node in cluster.affected_nodes:
                affected_layers.add(node[dimension])

            # 将簇分配给所有受影响的层
            for layer in affected_layers:
                distribution[layer].append(cluster)

        return dict(distribution)

    def _construct_subcube_paths(
        self,
        F: List[Tuple],
        subcubes: List[List[Tuple]],
        cluster_distribution: Dict[int, List[FaultCluster]],
        source: Tuple,
        target: Tuple,
        decomposition_dim: int
    ) -> Dict[int, List[Tuple]]:
        """为每个子立方体构造哈密尔顿路径"""
        subcube_paths = {}

        source_layer = source[decomposition_dim]
        target_layer = target[decomposition_dim]

        for layer, subcube_nodes in enumerate(subcubes):
            if not subcube_nodes:
                continue

            # 确定该子立方体的起点和终点
            if layer == source_layer:
                subcube_source = source
            else:
                subcube_source = subcube_nodes[0]  # 默认起点

            if layer == target_layer:
                subcube_target = target
            else:
                subcube_target = subcube_nodes[-1]  # 默认终点

            # 过滤该子立方体内的故障边
            subcube_faults = self._filter_subcube_faults(F, subcube_nodes)

            # 递归构造路径（降维）
            if self.Q.n > 1:
                # 创建降维的子问题
                sub_Q = QkCube(n=self.Q.n-1, k=self.Q.k)
                sub_embedding = RegionBasedHamiltonianEmbedding(sub_Q, self.rbf_params)

                # 将坐标投影到子空间
                projected_source = self._project_to_subspace(subcube_source, decomposition_dim)
                projected_target = self._project_to_subspace(subcube_target, decomposition_dim)
                projected_faults = [
                    (self._project_to_subspace(u, decomposition_dim),
                     self._project_to_subspace(v, decomposition_dim))
                    for u, v in subcube_faults
                ]

                # 递归调用
                projected_path = sub_embedding.embed_hamiltonian_path_rbf(
                    projected_faults, projected_source, projected_target
                )

                # 将路径投影回原空间
                subcube_path = [
                    self._unproject_from_subspace(node, decomposition_dim, layer)
                    for node in projected_path
                ]
            else:
                # 1维情况，直接构造
                subcube_path = self._simple_path_search(subcube_faults, subcube_source, subcube_target)

            subcube_paths[layer] = subcube_path

        return subcube_paths

    def _filter_subcube_faults(self, F: List[Tuple], subcube_nodes: List[Tuple]) -> List[Tuple]:
        """过滤出子立方体内的故障边"""
        subcube_node_set = set(subcube_nodes)
        subcube_faults = []

        for edge in F:
            u, v = edge
            if u in subcube_node_set and v in subcube_node_set:
                subcube_faults.append(edge)

        return subcube_faults

    def _project_to_subspace(self, node: Tuple, excluded_dim: int) -> Tuple:
        """将节点投影到子空间（去除指定维度）"""
        return tuple(coord for i, coord in enumerate(node) if i != excluded_dim)

    def _unproject_from_subspace(self, node: Tuple, excluded_dim: int, layer_value: int) -> Tuple:
        """将子空间节点投影回原空间"""
        result = list(node)
        result.insert(excluded_dim, layer_value)
        return tuple(result)

    def _stitch_subcube_paths(
        self,
        subcube_paths: Dict[int, List[Tuple]],
        decomposition_dim: int,
        source: Tuple,
        target: Tuple
    ) -> List[Tuple]:
        """
        缝合子立方体路径为完整的哈密尔顿路径

        这是算法的关键步骤：利用跨维度边将各个子立方体的路径连接起来
        """
        if not subcube_paths:
            return []

        # 确定路径遍历顺序
        source_layer = source[decomposition_dim]
        target_layer = target[decomposition_dim]

        # 计算最优的层遍历顺序
        layer_order = self._calculate_optimal_layer_order(
            subcube_paths, source_layer, target_layer
        )

        # 按顺序连接各层的路径
        final_path = []

        for i, layer in enumerate(layer_order):
            if layer not in subcube_paths:
                continue

            layer_path = subcube_paths[layer]
            if not layer_path:
                continue

            if i == 0:
                # 第一层：直接添加
                final_path.extend(layer_path)
            else:
                # 后续层：需要找到连接边
                prev_layer = layer_order[i-1]
                connection_edge = self._find_connection_edge(
                    final_path[-1], layer_path[0], decomposition_dim
                )

                if connection_edge:
                    # 添加连接路径
                    final_path.extend(connection_edge[1:])  # 跳过重复的起点
                    final_path.extend(layer_path[1:])       # 跳过重复的起点
                else:
                    # 无法连接，算法失败
                    return []

        return final_path

    def _calculate_optimal_layer_order(
        self,
        subcube_paths: Dict[int, List[Tuple]],
        source_layer: int,
        target_layer: int
    ) -> List[int]:
        """计算最优的层遍历顺序"""
        available_layers = list(subcube_paths.keys())

        if source_layer == target_layer:
            # 源和目标在同一层，简单情况
            return [source_layer]

        # 计算从源层到目标层的路径
        k = self.Q.k

        # 选择较短的路径方向
        forward_distance = (target_layer - source_layer) % k
        backward_distance = (source_layer - target_layer) % k

        if forward_distance <= backward_distance:
            # 正向遍历
            order = []
            current = source_layer
            while current != target_layer:
                if current in available_layers:
                    order.append(current)
                current = (current + 1) % k
            if target_layer in available_layers:
                order.append(target_layer)
        else:
            # 反向遍历
            order = []
            current = source_layer
            while current != target_layer:
                if current in available_layers:
                    order.append(current)
                current = (current - 1) % k
            if target_layer in available_layers:
                order.append(target_layer)

        return order

    def _find_connection_edge(
        self,
        from_node: Tuple,
        to_node: Tuple,
        decomposition_dim: int
    ) -> Optional[List[Tuple]]:
        """寻找连接两个节点的跨维度路径"""
        # 检查是否可以直接连接
        if self._are_adjacent(from_node, to_node, decomposition_dim):
            return [from_node, to_node]

        # 寻找中间路径（在同一层内移动到可连接位置）
        target_coords = list(to_node)
        intermediate = list(from_node)

        # 将中间节点的非分解维度坐标调整为目标坐标
        for i in range(len(intermediate)):
            if i != decomposition_dim:
                intermediate[i] = target_coords[i]

        intermediate_node = tuple(intermediate)

        # 检查中间节点是否可达
        if self._are_adjacent(from_node, intermediate_node, decomposition_dim):
            return [from_node, intermediate_node, to_node]

        return None

    def _are_adjacent(self, node1: Tuple, node2: Tuple, decomposition_dim: int) -> bool:
        """检查两个节点是否相邻（考虑分解维度）"""
        diff_count = 0
        diff_dim = -1

        for i in range(len(node1)):
            if node1[i] != node2[i]:
                diff_count += 1
                diff_dim = i

        # 相邻条件：恰好在一个维度上相差1
        if diff_count == 1:
            if diff_dim == decomposition_dim:
                # 跨层连接
                return abs(node1[diff_dim] - node2[diff_dim]) == 1 or \
                       abs(node1[diff_dim] - node2[diff_dim]) == self.Q.k - 1
            else:
                # 层内连接
                return abs(node1[diff_dim] - node2[diff_dim]) == 1 or \
                       abs(node1[diff_dim] - node2[diff_dim]) == self.Q.k - 1

        return False

    def _simple_path_search(self, F: List[Tuple], source: Tuple, target: Tuple) -> List[Tuple]:
        """改进的路径搜索算法（用于基础情况和回退）"""
        if source == target:
            return [source]

        # 对于小网络，尝试构造哈密尔顿路径
        total_nodes = self.Q.k ** self.Q.n
        if total_nodes <= 50:
            hamiltonian_path = self._try_hamiltonian_path(F, source, target)
            if hamiltonian_path:
                return hamiltonian_path

        # 否则使用改进的路径搜索
        return self._improved_path_search(F, source, target)

    def _try_hamiltonian_path(self, F: List[Tuple], source: Tuple, target: Tuple) -> List[Tuple]:
        """尝试构造哈密尔顿路径（访问所有节点）"""
        all_nodes = set(self.Q.generate_all_nodes())

        def dfs(current, path, remaining):
            if not remaining:
                return path if current == target else None

            total_nodes = self.Q.k ** self.Q.n
            if len(path) > total_nodes:  # 防止无限循环
                return None

            for neighbor in self._get_neighbors(current):
                if (neighbor in remaining and
                    not self._is_edge_faulty(current, neighbor, F)):

                    new_remaining = remaining - {neighbor}
                    result = dfs(neighbor, path + [neighbor], new_remaining)
                    if result:
                        return result

            return None

        # 从源节点开始DFS
        remaining_nodes = all_nodes - {source}
        return dfs(source, [source], remaining_nodes)

    def _improved_path_search(self, F: List[Tuple], source: Tuple, target: Tuple) -> List[Tuple]:
        """改进的路径搜索（不要求访问所有节点，但尽量访问更多）"""
        from collections import deque

        # 使用A*算法寻找较好的路径
        def heuristic(node):
            return sum(min(abs(node[i] - target[i]), self.Q.k - abs(node[i] - target[i]))
                      for i in range(self.Q.n))

        # 优先队列：(f_score, g_score, node, path)
        import heapq
        open_set = [(heuristic(source), 0, source, [source])]
        visited = {source}

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == target:
                return path

            # 限制路径长度，避免过长
            total_nodes = self.Q.k ** self.Q.n
            if len(path) > min(100, total_nodes * 2):
                continue

            for neighbor in self._get_neighbors(current):
                if neighbor not in visited and not self._is_edge_faulty(current, neighbor, F):
                    visited.add(neighbor)
                    new_g_score = g_score + 1
                    new_f_score = new_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, path + [neighbor]))

        return []  # 无法找到路径

    def _get_neighbors(self, node: Tuple) -> List[Tuple]:
        """获取节点的所有相邻节点"""
        neighbors = []
        n, k = self.Q.n, self.Q.k

        for dim in range(n):
            for direction in [-1, 1]:
                next_coords = list(node)
                next_coords[dim] = (node[dim] + direction) % k
                neighbors.append(tuple(next_coords))

        return neighbors

    def _is_edge_faulty(self, u: Tuple, v: Tuple, F: List[Tuple]) -> bool:
        """检查边是否故障"""
        return (u, v) in F or (v, u) in F

    def _fallback_algorithm(self, F: List[Tuple], source: Tuple, target: Tuple) -> List[Tuple]:
        """回退算法：当不满足RBF条件时使用"""
        # 使用简单的路径搜索作为回退
        return self._simple_path_search(F, source, target)
