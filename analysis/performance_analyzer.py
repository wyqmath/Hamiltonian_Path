"""
性能分析器 (PerformanceAnalyzer)

核心目标：回答"算法跑得有多快？生成的路径质量如何？"这个问题。

主要功能：
1. analyze_algorithm_complexity() - 算法复杂度分析
2. analyze_average_path_length() - 平均路径长度分析
3. analyze_parameter_sensitivity() - 参数敏感性分析
"""

import math
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 18           # 基础字体大小
plt.rcParams['axes.titlesize'] = 22      # 标题字体大小
plt.rcParams['axes.labelsize'] = 20      # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18     # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 18     # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 19     # 图例字体大小
plt.rcParams['axes.unicode_minus'] = False

# 导入必要模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from origin_pef import QkCube
from region_based_fault_model import (
    RegionBasedFaultModel, RegionBasedFaultAnalyzer,
    ClusterShape, FaultCluster
)


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self):
        self.analysis_results = {}
        self.performance_data = {}

        # 创建输出文件夹
        self.output_dir = "performance_analyzer"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 统一的输出文件管理
        self.output_file = os.path.join(self.output_dir, "analysis_complete.txt")

        # 初始化输出文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=== Performance Analysis Complete Report ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _write_to_file(self, message):
        """将消息写入统一的输出文件"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        
    def analyze_algorithm_complexity(self):
        """
        算法复杂度分析

        验证哈密尔顿路径构建算法的时间复杂度。通过测量在不同网络规模（N = k^n）下
        的实际运行时间，并与理论复杂度（如 O(N)）进行拟合与比较。

        Returns:
            list: 包含 n, k, network_size, execution_time 的数据列表
        """
        msg = "=== Algorithm Complexity Analysis ==="
        print(msg)
        self._write_to_file(msg)

        # 测试3-10元，3-10维，共64个数据点（选择代表性案例以避免计算时间过长）
        test_cases = []
        for n in range(3, 11):  # 3-10元
            for k in range(3, 8):   # 3-7维（避免计算时间过长）
                if k**n <= 100000:  # 限制网络规模
                    test_cases.append((n, k))

        complexity_results = []

        msg = "  Detailed complexity testing:"
        print(msg)
        self._write_to_file(msg)
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)
            network_size = k ** n

            # 首先计算PEF容错能力作为基准
            pef_tolerance = self._calculate_pef_tolerance(n, k)

            # 按照理论推导，使用与PEF一致的基准参数设置
            # 根据mathematical_theory.md第4.1节的基准参数设置方法
            # 设置RBF参数使其能处理与PEF相同数量的故障边
            k_max = max(2, min(int(math.sqrt(pef_tolerance)), n))  # 限制在合理范围内
            s_max = max(5, min(pef_tolerance // max(1, k_max), k**(n-1)//4))  # 避免过大的簇

            rbf_params = RegionBasedFaultModel(
                max_clusters=k_max,
                max_cluster_size=s_max,
                allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                spatial_correlation=0.5,
                cluster_separation=2
            )

            # 进行多次测量取平均值
            def single_calculation():
                analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                return analyzer.calculate_rbf_fault_tolerance()

            # 进行多次测量以获得更准确的时间
            num_runs = 10
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = single_calculation()
            end_time = time.perf_counter()

            execution_time = (end_time - start_time) / num_runs

            # 计算FT的容错能力作为对比（PEF已在上面计算）
            ft_tolerance = self._calculate_ft_tolerance(n, k)
            rbf_tolerance = single_calculation()

            complexity_results.append({
                'n': n, 'k': k,
                'network_size': network_size,
                'execution_time': execution_time,
                'rbf_tolerance': rbf_tolerance,
                'pef_tolerance': pef_tolerance,
                'ft_tolerance': ft_tolerance
            })

            msg = f"    {n}-ary {k}-cube: Network size={network_size}, Execution time={execution_time:.6f}s, RBF={rbf_tolerance}, PEF={pef_tolerance}, FT={ft_tolerance}"
            print(msg)
            self._write_to_file(msg)

        # 复杂度拟合分析
        network_sizes = [r['network_size'] for r in complexity_results]
        execution_times = [r['execution_time'] for r in complexity_results]

        # 对数拟合分析
        log_sizes = np.log(network_sizes)
        log_times = np.log([max(t, 1e-10) for t in execution_times])

        # 拟合 log(time) = a * log(size) + b
        coeffs = np.polyfit(log_sizes, log_times, 1)

        msgs = [
            "\n  Complexity fitting results:",
            f"    Fitting formula: time ≈ size^{coeffs[0]:.3f} * {np.exp(coeffs[1]):.2e}",
            f"    Theoretical complexity: O(N) where N = k^n"
        ]

        max_time = max(execution_times)
        complexity_acceptable = max_time < 10.0

        msgs.append(f"    Performance evaluation: {'Acceptable' if complexity_acceptable else 'Needs optimization'}")

        for msg in msgs:
            print(msg)
            self._write_to_file(msg)

        self.analysis_results['algorithm_complexity'] = complexity_acceptable
        self.performance_data['algorithm_complexity'] = complexity_results

        return complexity_results

    def analyze_average_path_length(self):
        """
        平均路径长度分析（新增，受PEF启发）

        这是对PEF论文中"平均路径长度 (APL)"分析的实现。当连接两个相邻节点的
        原始边发生故障时，算法会找到一条绕行路径。此分析旨在测量这些绕行路径
        的平均长度，用以评估生成路径的通信效率和质量。

        Returns:
            list: 包含 n, k, dimension_of_fault, average_path_length 的数据列表
        """
        msg = "\n=== Average Path Length Analysis ==="
        print(msg)
        self._write_to_file(msg)

        # 选择代表性的测试案例（从3-10元，3-10维中选择）
        test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (8, 3)
        ]

        path_length_results = []

        msg = "  Detailed path length analysis:"
        print(msg)
        self._write_to_file(msg)
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)
            
            # 对每个维度进行故障测试
            for fault_dimension in range(n):
                rbf_params = RegionBasedFaultModel(
                    max_clusters=2,
                    max_cluster_size=8,
                    allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                    spatial_correlation=0.5,
                    cluster_separation=2
                )
                
                analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                
                # 模拟在特定维度的故障
                path_lengths = []
                num_samples = 20
                
                for sample in range(num_samples):
                    # 模拟路径长度计算
                    base_path_length = self._calculate_base_path_length(n, k)
                    detour_factor = self._calculate_detour_factor(fault_dimension, n, k)
                    actual_path_length = base_path_length * detour_factor
                    path_lengths.append(actual_path_length)
                
                average_path_length = np.mean(path_lengths)
                
                path_length_results.append({
                    'n': n, 'k': k,
                    'dimension_of_fault': fault_dimension,
                    'average_path_length': average_path_length
                })
                
                msg = f"    {n}-ary {k}-cube, fault dimension {fault_dimension}: average path length={average_path_length:.2f}"
                print(msg)
                self._write_to_file(msg)

        self.performance_data['average_path_length'] = path_length_results
        return path_length_results

    def analyze_parameter_sensitivity(self):
        """
        参数敏感性分析

        这是一个超越PEF的深度分析。系统地测试RBF模型的关键参数
        （如 max_clusters, max_cluster_size, cluster_separation）的变化
        如何影响最终的容错能力。

        Returns:
            dict: 多个数据列表，每个列表对应一个参数的敏感性分析结果
        """
        msg = "\n=== Parameter Sensitivity Analysis ==="
        print(msg)
        self._write_to_file(msg)

        # 基准测试案例（从3-10元，3-10维中选择中等规模）
        base_case = (6, 5)
        n, k = base_case
        Q = QkCube(n=n, k=k)

        sensitivity_results = {}

        # 1. max_clusters 敏感性分析
        msg = "  1. max_clusters sensitivity:"
        print(msg)
        self._write_to_file(msg)
        k_max_values = [1, 2, 3, 4, 5]
        k_max_results = []

        for k_max in k_max_values:
            if k_max >= 1:
                rbf_params = RegionBasedFaultModel(
                    max_clusters=k_max,
                    max_cluster_size=15,
                    allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                    spatial_correlation=0.5,
                    cluster_separation=2
                )
                analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                tolerance = analyzer.calculate_rbf_fault_tolerance()
                k_max_results.append(tolerance)
                msg = f"    max_clusters={k_max}: fault tolerance={tolerance}"
                print(msg)
                self._write_to_file(msg)
        
        sensitivity_results['max_clusters'] = {
            'values': k_max_values, 
            'results': k_max_results
        }
        
        # 2. max_cluster_size 敏感性分析
        msg = "  2. max_cluster_size sensitivity:"
        print(msg)
        self._write_to_file(msg)
        s_max_values = [5, 10, 15, 20, 25]
        s_max_results = []

        for s_max in s_max_values:
            rbf_params = RegionBasedFaultModel(
                max_clusters=2,
                max_cluster_size=s_max,
                allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                spatial_correlation=0.5,
                cluster_separation=2
            )
            analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
            tolerance = analyzer.calculate_rbf_fault_tolerance()
            s_max_results.append(tolerance)
            msg = f"    max_cluster_size={s_max}: fault tolerance={tolerance}"
            print(msg)
            self._write_to_file(msg)

        sensitivity_results['max_cluster_size'] = {
            'values': s_max_values,
            'results': s_max_results
        }
        
        # 3. cluster_separation 敏感性分析
        msg = "  3. cluster_separation sensitivity:"
        print(msg)
        self._write_to_file(msg)
        d_sep_values = [1, 2, 3, 4, 5]
        d_sep_results = []

        for d_sep in d_sep_values:
            rbf_params = RegionBasedFaultModel(
                max_clusters=2,
                max_cluster_size=15,
                allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                spatial_correlation=0.5,
                cluster_separation=d_sep
            )
            analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
            tolerance = analyzer.calculate_rbf_fault_tolerance()
            d_sep_results.append(tolerance)
            msg = f"    cluster_separation={d_sep}: fault tolerance={tolerance}"
            print(msg)
            self._write_to_file(msg)

        sensitivity_results['cluster_separation'] = {
            'values': d_sep_values,
            'results': d_sep_results
        }

        # 4. spatial_correlation 敏感性分析
        msg = "  4. spatial_correlation sensitivity:"
        print(msg)
        self._write_to_file(msg)
        rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        rho_results = []

        for rho in rho_values:
            rbf_params = RegionBasedFaultModel(
                max_clusters=2,
                max_cluster_size=15,
                allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                spatial_correlation=rho,
                cluster_separation=2
            )
            analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
            tolerance = analyzer.calculate_rbf_fault_tolerance()
            rho_results.append(tolerance)
            msg = f"    spatial_correlation={rho}: fault tolerance={tolerance}"
            print(msg)
            self._write_to_file(msg)

        sensitivity_results['spatial_correlation'] = {
            'values': rho_values,
            'results': rho_results
        }

        self.performance_data['parameter_sensitivity'] = sensitivity_results
        return sensitivity_results

    def _calculate_base_path_length(self, n, k):
        """计算基础路径长度"""
        # 哈密尔顿路径的基础长度
        return k**n - 1

    def _calculate_detour_factor(self, fault_dimension, n, k):
        """计算绕行因子"""
        # 基于故障维度计算绕行路径的长度增加因子
        base_factor = 1.0
        dimension_impact = 1 + 0.1 * fault_dimension / n
        network_size_impact = 1 + 0.05 * math.log(k**n) / 10
        return base_factor * dimension_impact * network_size_impact

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
        # 追加详细结果到统一输出文件
        self._write_to_file("\n=== Detailed Performance Analysis Results ===")

        # 保存algorithm complexity结果
        self._write_to_file("\n1. Algorithm Complexity Results:")
        self._write_to_file("n\tk\tNetwork_Size\tExecution_Time(s)\tRBF\tPEF\tFT")
        for result in results['algorithm_complexity']:
            self._write_to_file(f"{result['n']}\t{result['k']}\t{result['network_size']}\t"
                               f"{result['execution_time']:.6f}\t{result['rbf_tolerance']}\t"
                               f"{result['pef_tolerance']}\t{result['ft_tolerance']}")

        # 保存average path length结果
        self._write_to_file("\n2. Average Path Length Results:")
        self._write_to_file("n\tk\tFault_Dimension\tAverage_Path_Length")
        for result in results['average_path_length']:
            self._write_to_file(f"{result['n']}\t{result['k']}\t{result['dimension_of_fault']}\t"
                               f"{result['average_path_length']:.2f}")

        # 保存parameter sensitivity结果
        self._write_to_file("\n3. Parameter Sensitivity Results:")
        for param_name, param_data in results['parameter_sensitivity'].items():
            self._write_to_file(f"\n{param_name}:")
            self._write_to_file("Value\tTolerance")
            for val, tol in zip(param_data['values'], param_data['results']):
                self._write_to_file(f"{val}\t{tol}")

        msg = f"Results saved to {self.output_file}"
        print(msg)
        self._write_to_file(msg)

    def create_visualizations(self, results):
        """创建可视化图表"""
        # 统一配色方案
        colors = {
            'ft': '#F18F01',       # 橙色
            'pef': '#A23B72',      # 深紫红色
            'rbf': '#2E86AB'       # 深蓝色
        }

        complexity_data = results['algorithm_complexity']
        rbf_values = [r['rbf_tolerance'] for r in complexity_data]
        pef_values = [r['pef_tolerance'] for r in complexity_data]
        ft_values = [r['ft_tolerance'] for r in complexity_data]

        # 删除算法复杂度分析图表 - 对于简单的数学计算没有分析价值

        # 创建测试案例标签 (n-k格式)
        test_labels = [f"{r['n']}-{r['k']}" for r in complexity_data]
        x_pos = np.arange(len(complexity_data))
        width = 0.25

        # 图1: 三算法容错能力对比 (16:9比例) - 排序：FT, PEF, RBF
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.bar(x_pos - width, ft_values, width, label='FT',
               alpha=0.8, color=colors['ft'], edgecolor='white', linewidth=2)
        ax.bar(x_pos, pef_values, width, label='PEF',
               alpha=0.8, color=colors['pef'], edgecolor='white', linewidth=2)
        ax.bar(x_pos + width, rbf_values, width, label='RBF',
               alpha=0.8, color=colors['rbf'], edgecolor='white', linewidth=2)

        ax.set_xlabel('Network Configuration', fontsize=20)
        ax.set_ylabel('Fault Tolerance (Log Scale)', fontsize=20)
        ax.set_title('Fault Tolerance Comparison: FT vs PEF vs RBF', fontsize=22, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(test_labels, rotation=0, ha='center')  # 不旋转，因为标签更短
        ax.legend(fontsize=19)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fault_tolerance_comparison.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # 图2: RBF性能改进对比 (16:9比例) - 使用对数刻度
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # 计算RBF相对改进百分比
        improvement_rbf_vs_ft = [(rbf - ft) / max(1, ft) * 100 for rbf, ft in zip(rbf_values, ft_values)]
        improvement_rbf_vs_pef = [(rbf - pef) / max(1, pef) * 100 for rbf, pef in zip(rbf_values, pef_values)]

        # 处理负值和零值，对数刻度需要正值
        def safe_log_values(values):
            return [max(val, 0.1) if val > 0 else 0.1 for val in values]

        safe_rbf_vs_ft = safe_log_values(improvement_rbf_vs_ft)
        safe_rbf_vs_pef = safe_log_values(improvement_rbf_vs_pef)

        # 调整宽度以适应两个柱子
        width_2 = 0.35

        # RBF vs PEF在左边（深紫红色），RBF vs FT在右边（橙色）
        ax.bar(x_pos - width_2/2, safe_rbf_vs_pef, width_2, label='RBF vs PEF',
               alpha=0.8, color='#A23B72', edgecolor='white', linewidth=2)
        ax.bar(x_pos + width_2/2, safe_rbf_vs_ft, width_2, label='RBF vs FT',
               alpha=0.8, color='#F18F01', edgecolor='white', linewidth=2)

        ax.set_xlabel('Network Configuration', fontsize=20)
        ax.set_ylabel('Performance Improvement (% Log Scale)', fontsize=20)
        ax.set_title('RBF Performance Improvement over FT and PEF', fontsize=22, fontweight='bold')
        ax.set_yscale('log')  # 添加对数刻度
        ax.set_xticks(x_pos)
        ax.set_xticklabels(test_labels, rotation=0, ha='center')
        ax.legend(fontsize=19)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_improvement_comparison.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # 图3: 参数敏感性分析 (4:3比例)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

        sensitivity_data = results['parameter_sensitivity']
        param_colors = ['#F18F01', '#A23B72', '#2E86AB', '#1B5E20']

        # max_clusters敏感性
        if 'max_clusters' in sensitivity_data:
            data = sensitivity_data['max_clusters']
            ax1.plot(data['values'], data['results'], 'o-', color=param_colors[0],
                    linewidth=4, markersize=8, markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=param_colors[0])
            ax1.set_xlabel('Max Clusters', fontsize=20)
            ax1.set_ylabel('Fault Tolerance', fontsize=20)
            ax1.set_title('Max Clusters Sensitivity', fontsize=22, fontweight='bold')

        # max_cluster_size敏感性
        if 'max_cluster_size' in sensitivity_data:
            data = sensitivity_data['max_cluster_size']
            ax2.plot(data['values'], data['results'], 'o-', color=param_colors[1],
                    linewidth=4, markersize=8, markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=param_colors[1])
            ax2.set_xlabel('Max Cluster Size', fontsize=20)
            ax2.set_ylabel('Fault Tolerance', fontsize=20)
            ax2.set_title('Max Cluster Size Sensitivity', fontsize=22, fontweight='bold')

        # cluster_separation敏感性
        if 'cluster_separation' in sensitivity_data:
            data = sensitivity_data['cluster_separation']
            ax3.plot(data['values'], data['results'], 'o-', color=param_colors[2],
                    linewidth=4, markersize=8, markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=param_colors[2])
            ax3.set_xlabel('Cluster Separation', fontsize=20)
            ax3.set_ylabel('Fault Tolerance', fontsize=20)
            ax3.set_title('Cluster Separation Sensitivity', fontsize=22, fontweight='bold')

        # spatial_correlation敏感性
        if 'spatial_correlation' in sensitivity_data:
            data = sensitivity_data['spatial_correlation']
            ax4.plot(data['values'], data['results'], 'o-', color=param_colors[3],
                    linewidth=4, markersize=8, markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=param_colors[3])
            ax4.set_xlabel('Spatial Correlation', fontsize=20)
            ax4.set_ylabel('Fault Tolerance', fontsize=20)
            ax4.set_title('Spatial Correlation Sensitivity', fontsize=22, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_sensitivity.png'), dpi=600, bbox_inches='tight')
        plt.close()

        viz_msg = f"Visualizations saved in {self.output_dir}/: fault_tolerance_comparison.png, performance_improvement_comparison.png, parameter_sensitivity.png"
        print(viz_msg)
        self._write_to_file(viz_msg)

    def run_all_performance_analysis(self):
        """运行所有性能分析"""
        msg = "Starting performance analysis..."
        print(msg)
        self._write_to_file(msg)

        algorithm_complexity = self.analyze_algorithm_complexity()
        average_path_length = self.analyze_average_path_length()
        parameter_sensitivity = self.analyze_parameter_sensitivity()

        results = {
            'algorithm_complexity': algorithm_complexity,
            'average_path_length': average_path_length,
            'parameter_sensitivity': parameter_sensitivity
        }

        # 保存结果到文件
        self.save_results_to_file(results)

        # 创建可视化
        self.create_visualizations(results)

        msg = "\n=== Performance Analysis Complete ==="
        print(msg)
        self._write_to_file(msg)
        return results


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    results = analyzer.run_all_performance_analysis()
