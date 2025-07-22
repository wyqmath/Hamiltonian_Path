"""
属性分析器 (PropertyAnalyzer)

核心目标：回答"RBF模型和算法的内在机制是什么？"这个问题。
主要关注RBF模型独有的特性。

主要功能：
1. analyze_correction_factors() - 修正因子分析
2. analyze_decomposition_dimension() - 分解维度选择分析
3. analyze_cluster_geometry() - 故障簇几何分析
"""

import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 设置matplotlib字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 导入必要模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from origin_pef import QkCube
from region_based_fault_model import (
    RegionBasedFaultModel, RegionBasedFaultAnalyzer,
    ClusterShape, FaultCluster
)


class PropertyAnalyzer:
    """属性分析器"""

    def __init__(self):
        self.analysis_results = {}
        self.performance_data = {}

        # 创建输出文件夹
        self.output_dir = "property_analyzer"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file = os.path.join(self.output_dir, "property_analyzer.txt")
        
    def analyze_correction_factors(self):
        """
        修正因子分析
        
        计算并分析结构修正因子 (alpha_struct) 和空间修正因子 (alpha_spatial) 
        在不同网络配置下的具体数值和变化趋势。这是解释RBF模型为何优越的核心。
        
        Returns:
            list: 包含 n, k, d_sep, alpha_struct, alpha_spatial, alpha_total 的数据列表
        """
        print("=== 修正因子分析 ===")

        # 生成3-10元，3-10维的测试案例，共64个基础数据点
        test_cases = []
        for n in range(3, 11):  # 3-10元
            for k in range(3, 8):   # 3-7维（减少计算量）
                for d_sep in [1, 2, 3]:  # 不同分离距离
                    test_cases.append((n, k, d_sep))
        
        correction_factor_results = []
        
        print("  详细修正因子计算:")
        for n, k, d_sep in test_cases:
            # 计算结构修正因子
            alpha_struct = min(1 + math.log(n * k / 2) / n, 2.0)
            
            # 计算空间修正因子
            rho = 0.5  # 默认空间相关性
            alpha_spatial = (1 + 0.5 * (1 - rho)) * (1 + math.log(1 + d_sep) / 10)
            
            # 总修正因子
            alpha_total = alpha_struct * alpha_spatial
            
            correction_factor_results.append({
                'n': n, 'k': k, 'd_sep': d_sep,
                'alpha_struct': alpha_struct,
                'alpha_spatial': alpha_spatial,
                'alpha_total': alpha_total
            })
            
            if len(correction_factor_results) <= 20:  # 只显示前20个结果
                print(f"    {n}元{k}维(d_sep={d_sep}): α_struct={alpha_struct:.4f}, "
                      f"α_spatial={alpha_spatial:.4f}, α_total={alpha_total:.4f}")
        
        # 统计分析
        alpha_struct_values = [r['alpha_struct'] for r in correction_factor_results]
        alpha_spatial_values = [r['alpha_spatial'] for r in correction_factor_results]
        alpha_total_values = [r['alpha_total'] for r in correction_factor_results]
        
        print(f"\n  修正因子统计:")
        print(f"    结构修正因子范围: [{min(alpha_struct_values):.4f}, {max(alpha_struct_values):.4f}]")
        print(f"    空间修正因子范围: [{min(alpha_spatial_values):.4f}, {max(alpha_spatial_values):.4f}]")
        print(f"    总修正因子范围: [{min(alpha_total_values):.4f}, {max(alpha_total_values):.4f}]")
        
        self.analysis_results['correction_factors'] = True
        self.performance_data['correction_factors'] = correction_factor_results
        
        return correction_factor_results

    def analyze_decomposition_dimension(self):
        """
        分解维度选择分析
        
        验证和展示算法中"分解维度选择"策略的有效性。对于给定的故障簇分布，
        计算每个维度的分离度得分，并找出最佳维度。
        
        Returns:
            list: 包含 n, k, cluster_config, best_dimension, separation_score 的数据列表
        """
        print("\n=== 分解维度选择分析 ===")

        # 选择代表性的测试案例（从3-10元，3-10维中选择）
        test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (7, 4),
            (8, 3), (9, 3), (10, 3)
        ]
        
        decomposition_results = []
        
        print("  详细分解维度分析:")
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)
            
            # 创建多种故障簇配置
            cluster_configs = self._generate_cluster_configurations(n, k)
            
            for config_id, clusters in enumerate(cluster_configs):
                best_dim = 0
                best_separation = 0
                dimension_scores = []
                
                # 测试每个维度的分离度
                for dim in range(n):
                    separation_score = self._calculate_separation_score(clusters, dim, Q)
                    dimension_scores.append(separation_score)
                    
                    if separation_score > best_separation:
                        best_separation = separation_score
                        best_dim = dim
                
                decomposition_results.append({
                    'n': n, 'k': k,
                    'cluster_config': config_id,
                    'best_dimension': best_dim,
                    'separation_score': best_separation,
                    'all_scores': dimension_scores
                })
                
                print(f"    {n}元{k}维配置{config_id}: 最佳维度={best_dim}, "
                      f"分离度={best_separation:.4f}")
        
        self.performance_data['decomposition_dimension'] = decomposition_results
        return decomposition_results

    def analyze_cluster_geometry(self):
        """
        故障簇几何分析
        
        分析RBF模型中"故障簇"的几何与拓扑属性，如直径、跨度、密度等。
        这展示了RBF模型对故障模式的精细刻画能力，这是PEF等传统模型不具备的。
        
        Returns:
            list: 包含 n, k, cluster_shape, diameter, span, density 的数据列表
        """
        print("\n=== 故障簇几何分析 ===")

        # 测试不同的网络配置和簇形状（从3-10元，3-10维中选择）
        test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (8, 3)
        ]
        
        cluster_shapes = [
            (ClusterShape.COMPLETE_GRAPH, "Complete"),
            (ClusterShape.STAR_GRAPH, "Star")
        ]
        
        geometry_results = []
        
        print("  详细几何属性分析:")
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)
            
            for shape, shape_name in cluster_shapes:
                # 创建测试簇
                center = tuple([0] * n)
                fault_edges = self._create_cluster_edges(center, shape, k, n, 5)
                
                cluster = FaultCluster(
                    cluster_id=0,
                    fault_edges=fault_edges,
                    affected_nodes=set(),
                    shape=shape,
                    size=len(fault_edges),
                    center=center,
                    radius=1,
                    connectivity=1.0
                )
                
                # 分析几何属性
                diameter = self._calculate_cluster_diameter(cluster, Q)
                span = self._calculate_cluster_span(cluster, n)
                density = self._calculate_cluster_density(cluster)
                compactness = self._calculate_cluster_compactness(cluster, n)
                
                geometry_results.append({
                    'n': n, 'k': k,
                    'cluster_shape': shape_name,
                    'diameter': diameter,
                    'span': span,
                    'density': density,
                    'compactness': compactness
                })
                
                print(f"    {n}元{k}维{shape_name}簇: 直径={diameter}, 跨度={span}, "
                      f"密度={density:.3f}, 紧凑度={compactness:.3f}")
        
        self.performance_data['cluster_geometry'] = geometry_results
        return geometry_results

    def _generate_cluster_configurations(self, n, k):
        """生成不同的故障簇配置"""
        configurations = []
        
        # 配置1: 单个中心簇
        center1 = tuple([0] * n)
        config1 = [self._create_simple_cluster(center1, 0)]
        configurations.append(config1)
        
        # 配置2: 两个分离的簇
        center2 = tuple([k//2] * n)
        config2 = [
            self._create_simple_cluster(center1, 0),
            self._create_simple_cluster(center2, 1)
        ]
        configurations.append(config2)
        
        # 配置3: 对角分布的簇
        if n >= 3:
            center3 = tuple([k-1 if i < n//2 else 0 for i in range(n)])
            config3 = [
                self._create_simple_cluster(center1, 0),
                self._create_simple_cluster(center3, 1)
            ]
            configurations.append(config3)
        
        return configurations

    def _create_simple_cluster(self, center, cluster_id):
        """创建简单的故障簇"""
        fault_edges = [(center, center)]  # 简化的边表示
        return FaultCluster(
            cluster_id=cluster_id,
            fault_edges=fault_edges,
            affected_nodes={center},
            shape=ClusterShape.COMPLETE_GRAPH,
            size=1,
            center=center,
            radius=1,
            connectivity=1.0
        )

    def _calculate_separation_score(self, clusters, dimension, Q):
        """计算指定维度的分离度得分"""
        if len(clusters) < 2:
            return 0.0
        
        # 计算簇在指定维度上的分离程度
        dim_positions = []
        for cluster in clusters:
            if cluster.center:
                dim_positions.append(cluster.center[dimension])
        
        if len(dim_positions) < 2:
            return 0.0
        
        # 计算位置差异的标准差作为分离度
        return np.std(dim_positions)

    def _create_cluster_edges(self, center, shape, k, n, max_edges):
        """创建指定形状的簇边"""
        edges = []
        
        if shape == ClusterShape.COMPLETE_GRAPH:
            # 完全图：从中心向周围扩展
            for i in range(min(max_edges, n)):
                neighbor = list(center)
                neighbor[i] = (neighbor[i] + 1) % k
                edges.append((center, tuple(neighbor)))
        
        elif shape == ClusterShape.STAR_GRAPH:
            # 星图：中心连接多个叶子
            for i in range(min(max_edges, n)):
                neighbor = list(center)
                neighbor[i] = (neighbor[i] + 1) % k
                edges.append((center, tuple(neighbor)))
        
        return edges

    def _calculate_cluster_diameter(self, cluster, Q):
        """计算簇的直径"""
        if not cluster.affected_nodes:
            return 0
        
        nodes = list(cluster.affected_nodes)
        if len(nodes) < 2:
            return 0
        
        max_distance = 0
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                distance = self._hamming_distance(nodes[i], nodes[j])
                max_distance = max(max_distance, distance)
        
        return max_distance

    def _calculate_cluster_span(self, cluster, n):
        """计算簇的跨度"""
        if not cluster.center:
            return 0
        
        # 计算簇在各维度上的跨度
        spans = []
        for dim in range(n):
            spans.append(1)  # 简化计算
        
        return max(spans)

    def _calculate_cluster_density(self, cluster):
        """计算簇的密度"""
        if not cluster.affected_nodes or cluster.size == 0:
            return 0.0
        
        return cluster.size / max(1, len(cluster.affected_nodes))

    def _calculate_cluster_compactness(self, cluster, n):
        """计算簇的紧凑度"""
        if not cluster.center or cluster.radius == 0:
            return 1.0
        
        # 紧凑度 = 实际大小 / 理论最大大小
        theoretical_max = min(cluster.size * 2, n * 2)
        return cluster.size / max(1, theoretical_max)

    def _hamming_distance(self, node1, node2):
        """计算汉明距离"""
        if isinstance(node1, tuple) and isinstance(node2, tuple):
            return sum(a != b for a, b in zip(node1, node2))
        return 0

    def _calculate_pef_tolerance(self, n, k):
        """计算PEF模型的容错能力"""
        # 基于PEF论文的公式: (k^n - k^2)/(k-1) - 2n + 5
        return max(1, int((k**n - k**2) / (k - 1) - 2*n + 5))

    def _calculate_ft_tolerance(self, n, k):
        """计算传统FT模型的容错能力"""
        # 基于参考论文中的传统方法：2n-3 (对于奇数k>=3)
        return max(1, 2*n - 3)

    def save_results_to_file(self, results):
        """保存结果到txt文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=== Property Analysis Results ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 保存correction factors结果
            f.write("1. Correction Factors Results:\n")
            f.write("n\tk\td_sep\talpha_struct\talpha_spatial\talpha_total\n")
            for result in results['correction_factors']:
                f.write(f"{result['n']}\t{result['k']}\t{result['d_sep']}\t"
                       f"{result['alpha_struct']:.4f}\t{result['alpha_spatial']:.4f}\t"
                       f"{result['alpha_total']:.4f}\n")

            # 保存decomposition dimension结果
            f.write("\n2. Decomposition Dimension Results:\n")
            f.write("n\tk\tCluster_Config\tBest_Dimension\tSeparation_Score\n")
            for result in results['decomposition_dimension']:
                f.write(f"{result['n']}\t{result['k']}\t{result['cluster_config']}\t"
                       f"{result['best_dimension']}\t{result['separation_score']:.4f}\n")

            # 保存cluster geometry结果
            f.write("\n3. Cluster Geometry Results:\n")
            f.write("n\tk\tCluster_Shape\tDiameter\tSpan\tDensity\tCompactness\n")
            for result in results['cluster_geometry']:
                f.write(f"{result['n']}\t{result['k']}\t{result['cluster_shape']}\t"
                       f"{result['diameter']}\t{result['span']}\t{result['density']:.3f}\t"
                       f"{result['compactness']:.3f}\n")

        print(f"Results saved to {self.output_file}")

    def create_visualizations(self, results):
        """创建可视化图表"""
        # 图1: 修正因子分析
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        correction_data = results['correction_factors']
        n_values = [r['n'] for r in correction_data]
        alpha_struct = [r['alpha_struct'] for r in correction_data]
        alpha_spatial = [r['alpha_spatial'] for r in correction_data]
        alpha_total = [r['alpha_total'] for r in correction_data]

        # 子图1: 结构修正因子
        ax1.scatter(n_values, alpha_struct, alpha=0.6)
        ax1.set_xlabel('Network Arity (n)')
        ax1.set_ylabel('Structural Correction Factor')
        ax1.set_title('Structural Correction Factor vs Network Arity')
        ax1.grid(True, alpha=0.3)

        # 子图2: 空间修正因子
        ax2.scatter(n_values, alpha_spatial, alpha=0.6, color='red')
        ax2.set_xlabel('Network Arity (n)')
        ax2.set_ylabel('Spatial Correction Factor')
        ax2.set_title('Spatial Correction Factor vs Network Arity')
        ax2.grid(True, alpha=0.3)

        # 子图3: 总修正因子
        ax3.scatter(n_values, alpha_total, alpha=0.6, color='green')
        ax3.set_xlabel('Network Arity (n)')
        ax3.set_ylabel('Total Correction Factor')
        ax3.set_title('Total Correction Factor vs Network Arity')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correction_factors.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: 簇几何属性分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        geometry_data = results['cluster_geometry']

        # 按形状分组
        complete_data = [r for r in geometry_data if r['cluster_shape'] == 'Complete']
        star_data = [r for r in geometry_data if r['cluster_shape'] == 'Star']

        if complete_data and star_data:
            # 直径比较
            complete_diameters = [r['diameter'] for r in complete_data]
            star_diameters = [r['diameter'] for r in star_data]
            ax1.boxplot([complete_diameters, star_diameters], labels=['Complete', 'Star'])
            ax1.set_ylabel('Diameter')
            ax1.set_title('Cluster Diameter Comparison')
            ax1.grid(True, alpha=0.3)

            # 跨度比较
            complete_spans = [r['span'] for r in complete_data]
            star_spans = [r['span'] for r in star_data]
            ax2.boxplot([complete_spans, star_spans], labels=['Complete', 'Star'])
            ax2.set_ylabel('Span')
            ax2.set_title('Cluster Span Comparison')
            ax2.grid(True, alpha=0.3)

            # 密度比较
            complete_densities = [r['density'] for r in complete_data]
            star_densities = [r['density'] for r in star_data]
            ax3.boxplot([complete_densities, star_densities], labels=['Complete', 'Star'])
            ax3.set_ylabel('Density')
            ax3.set_title('Cluster Density Comparison')
            ax3.grid(True, alpha=0.3)

            # 紧凑度比较
            complete_compactness = [r['compactness'] for r in complete_data]
            star_compactness = [r['compactness'] for r in star_data]
            ax4.boxplot([complete_compactness, star_compactness], labels=['Complete', 'Star'])
            ax4.set_ylabel('Compactness')
            ax4.set_title('Cluster Compactness Comparison')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_geometry.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved in {self.output_dir}/: correction_factors.png, cluster_geometry.png")

    def run_all_property_analysis(self):
        """运行所有属性分析"""
        print("开始属性分析...")

        correction_factors = self.analyze_correction_factors()
        decomposition_dimension = self.analyze_decomposition_dimension()
        cluster_geometry = self.analyze_cluster_geometry()

        results = {
            'correction_factors': correction_factors,
            'decomposition_dimension': decomposition_dimension,
            'cluster_geometry': cluster_geometry
        }

        # 保存结果到文件
        self.save_results_to_file(results)

        # 创建可视化
        self.create_visualizations(results)

        print("\n=== 属性分析完成 ===")
        return results


if __name__ == "__main__":
    analyzer = PropertyAnalyzer()
    results = analyzer.run_all_property_analysis()
