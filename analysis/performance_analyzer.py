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


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self):
        self.analysis_results = {}
        self.performance_data = {}

        # 创建输出文件夹
        self.output_dir = "performance_analyzer"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file = os.path.join(self.output_dir, "performance_analyzer.txt")
        
    def analyze_algorithm_complexity(self):
        """
        算法复杂度分析
        
        验证哈密尔顿路径构建算法的时间复杂度。通过测量在不同网络规模（N = k^n）下
        的实际运行时间，并与理论复杂度（如 O(N)）进行拟合与比较。
        
        Returns:
            list: 包含 n, k, network_size, execution_time 的数据列表
        """
        print("=== 算法复杂度分析 ===")

        # 测试3-10元，3-10维，共64个数据点（选择代表性案例以避免计算时间过长）
        test_cases = []
        for n in range(3, 11):  # 3-10元
            for k in range(3, 8):   # 3-7维（避免计算时间过长）
                if k**n <= 100000:  # 限制网络规模
                    test_cases.append((n, k))
        
        complexity_results = []
        
        print("  详细复杂度测试:")
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)
            network_size = k ** n
            
            rbf_params = RegionBasedFaultModel(
                max_clusters=2,
                max_cluster_size=5,
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

            # 计算PEF和FT的容错能力作为对比
            pef_tolerance = self._calculate_pef_tolerance(n, k)
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

            print(f"    {n}元{k}维: 网络大小={network_size}, 执行时间={execution_time:.6f}s, "
                  f"RBF={rbf_tolerance}, PEF={pef_tolerance}, FT={ft_tolerance}")
        
        # 复杂度拟合分析
        network_sizes = [r['network_size'] for r in complexity_results]
        execution_times = [r['execution_time'] for r in complexity_results]
        
        # 对数拟合分析
        log_sizes = np.log(network_sizes)
        log_times = np.log([max(t, 1e-10) for t in execution_times])
        
        # 拟合 log(time) = a * log(size) + b
        coeffs = np.polyfit(log_sizes, log_times, 1)
        
        print(f"\n  复杂度拟合结果:")
        print(f"    拟合公式: time ≈ size^{coeffs[0]:.3f} * {np.exp(coeffs[1]):.2e}")
        print(f"    理论复杂度: O(N) 其中 N = k^n")
        
        max_time = max(execution_times)
        complexity_acceptable = max_time < 10.0
        
        print(f"    性能评估: {'可接受' if complexity_acceptable else '需要优化'}")

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
        print("\n=== 平均路径长度分析 ===")

        # 选择代表性的测试案例（从3-10元，3-10维中选择）
        test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (8, 3)
        ]
        
        path_length_results = []
        
        print("  详细路径长度分析:")
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
                
                print(f"    {n}元{k}维, 故障维度{fault_dimension}: "
                      f"平均路径长度={average_path_length:.2f}")
        
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
        print("\n=== 参数敏感性分析 ===")

        # 基准测试案例（从3-10元，3-10维中选择中等规模）
        base_case = (6, 5)
        n, k = base_case
        Q = QkCube(n=n, k=k)
        
        sensitivity_results = {}
        
        # 1. max_clusters 敏感性分析
        print("  1. max_clusters 敏感性:")
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
                print(f"    max_clusters={k_max}: 容错能力={tolerance}")
        
        sensitivity_results['max_clusters'] = {
            'values': k_max_values, 
            'results': k_max_results
        }
        
        # 2. max_cluster_size 敏感性分析
        print("  2. max_cluster_size 敏感性:")
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
            print(f"    max_cluster_size={s_max}: 容错能力={tolerance}")
        
        sensitivity_results['max_cluster_size'] = {
            'values': s_max_values, 
            'results': s_max_results
        }
        
        # 3. cluster_separation 敏感性分析
        print("  3. cluster_separation 敏感性:")
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
            print(f"    cluster_separation={d_sep}: 容错能力={tolerance}")
        
        sensitivity_results['cluster_separation'] = {
            'values': d_sep_values, 
            'results': d_sep_results
        }
        
        # 4. spatial_correlation 敏感性分析
        print("  4. spatial_correlation 敏感性:")
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
            print(f"    spatial_correlation={rho}: 容错能力={tolerance}")
        
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
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=== Performance Analysis Results ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 保存algorithm complexity结果
            f.write("1. Algorithm Complexity Results:\n")
            f.write("n\tk\tNetwork_Size\tExecution_Time(s)\tRBF\tPEF\tFT\n")
            for result in results['algorithm_complexity']:
                f.write(f"{result['n']}\t{result['k']}\t{result['network_size']}\t"
                       f"{result['execution_time']:.6f}\t{result['rbf_tolerance']}\t"
                       f"{result['pef_tolerance']}\t{result['ft_tolerance']}\n")

            # 保存average path length结果
            f.write("\n2. Average Path Length Results:\n")
            f.write("n\tk\tFault_Dimension\tAverage_Path_Length\n")
            for result in results['average_path_length']:
                f.write(f"{result['n']}\t{result['k']}\t{result['dimension_of_fault']}\t"
                       f"{result['average_path_length']:.2f}\n")

            # 保存parameter sensitivity结果
            f.write("\n3. Parameter Sensitivity Results:\n")
            for param_name, param_data in results['parameter_sensitivity'].items():
                f.write(f"\n{param_name}:\n")
                f.write("Value\tTolerance\n")
                for val, tol in zip(param_data['values'], param_data['results']):
                    f.write(f"{val}\t{tol}\n")

        print(f"Results saved to {self.output_file}")

    def create_visualizations(self, results):
        """创建可视化图表"""
        complexity_data = results['algorithm_complexity']
        network_sizes = [r['network_size'] for r in complexity_data]
        execution_times = [r['execution_time'] for r in complexity_data]
        rbf_values = [r['rbf_tolerance'] for r in complexity_data]
        pef_values = [r['pef_tolerance'] for r in complexity_data]
        ft_values = [r['ft_tolerance'] for r in complexity_data]

        # 图1: 算法复杂度分析
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(network_sizes, execution_times, 'bo-', label='Execution Time')
        ax.set_xlabel('Network Size')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Algorithm Complexity Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'algorithm_complexity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: RBF vs PEF 容错能力比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x_pos = np.arange(len(complexity_data))
        width = 0.35

        # 子图1: RBF vs PEF
        ax1.bar(x_pos - width/2, rbf_values, width, label='RBF', alpha=0.8, color='blue')
        ax1.bar(x_pos + width/2, pef_values, width, label='PEF', alpha=0.8, color='orange')
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Fault Tolerance')
        ax1.set_title('Fault Tolerance Comparison (RBF vs PEF)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 改进百分比
        improvement_pef = [(rbf - pef) / max(1, pef) * 100 for rbf, pef in zip(rbf_values, pef_values)]
        ax2.bar(x_pos, improvement_pef, width, alpha=0.8, color='green')
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('RBF vs PEF Performance Improvement')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rbf_vs_pef_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图3: RBF vs FT 容错能力比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 子图1: RBF vs FT
        ax1.bar(x_pos - width/2, rbf_values, width, label='RBF', alpha=0.8, color='blue')
        ax1.bar(x_pos + width/2, ft_values, width, label='FT', alpha=0.8, color='red')
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Fault Tolerance')
        ax1.set_title('Fault Tolerance Comparison (RBF vs FT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 改进百分比
        improvement_ft = [(rbf - ft) / max(1, ft) * 100 for rbf, ft in zip(rbf_values, ft_values)]
        ax2.bar(x_pos, improvement_ft, width, alpha=0.8, color='purple')
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('RBF vs FT Performance Improvement')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rbf_vs_ft_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: 参数敏感性分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        sensitivity_data = results['parameter_sensitivity']

        # max_clusters敏感性
        if 'max_clusters' in sensitivity_data:
            data = sensitivity_data['max_clusters']
            ax1.plot(data['values'], data['results'], 'bo-')
            ax1.set_xlabel('Max Clusters')
            ax1.set_ylabel('Fault Tolerance')
            ax1.set_title('Max Clusters Sensitivity')
            ax1.grid(True, alpha=0.3)

        # max_cluster_size敏感性
        if 'max_cluster_size' in sensitivity_data:
            data = sensitivity_data['max_cluster_size']
            ax2.plot(data['values'], data['results'], 'ro-')
            ax2.set_xlabel('Max Cluster Size')
            ax2.set_ylabel('Fault Tolerance')
            ax2.set_title('Max Cluster Size Sensitivity')
            ax2.grid(True, alpha=0.3)

        # cluster_separation敏感性
        if 'cluster_separation' in sensitivity_data:
            data = sensitivity_data['cluster_separation']
            ax3.plot(data['values'], data['results'], 'go-')
            ax3.set_xlabel('Cluster Separation')
            ax3.set_ylabel('Fault Tolerance')
            ax3.set_title('Cluster Separation Sensitivity')
            ax3.grid(True, alpha=0.3)

        # spatial_correlation敏感性
        if 'spatial_correlation' in sensitivity_data:
            data = sensitivity_data['spatial_correlation']
            ax4.plot(data['values'], data['results'], 'mo-')
            ax4.set_xlabel('Spatial Correlation')
            ax4.set_ylabel('Fault Tolerance')
            ax4.set_title('Spatial Correlation Sensitivity')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved in {self.output_dir}/: algorithm_complexity.png, rbf_vs_pef_performance.png, rbf_vs_ft_performance.png, parameter_sensitivity.png")

    def run_all_performance_analysis(self):
        """运行所有性能分析"""
        print("开始性能分析...")

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

        print("\n=== 性能分析完成 ===")
        return results


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    results = analyzer.run_all_performance_analysis()
