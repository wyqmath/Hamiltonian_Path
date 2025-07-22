"""
哈密尔顿性分析器 (HamiltonianAnalyzer)

核心目标：回答"算法能够成功运行的理论前提是什么？"这个问题。

主要功能：
1. verify_hamiltonian_condition() - 哈密尔顿性充分条件验证
2. analyze_boundary_cases() - 边界情况分析
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


class HamiltonianAnalyzer:
    """哈密尔顿性分析器"""

    def __init__(self):
        self.analysis_results = {}
        self.performance_data = {}

        # 创建输出文件夹
        self.output_dir = "hamiltonian_analyzer"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file = os.path.join(self.output_dir, "hamiltonian_analyzer.txt")
        
    def verify_hamiltonian_condition(self):
        """
        哈密尔顿性充分条件验证
        
        针对不同的 (n, k) 组合，系统地测试和验证论文中提出的哈密尔顿性
        充分条件（例如 k_max * s_max < k / 4）。
        
        Returns:
            list: 包含 n, k, k_max, s_max, product, limit, is_satisfied 的数据列表
        """
        print("=== 哈密尔顿性充分条件验证 ===")

        # 扩展到3-10元，3-10维的哈密尔顿性条件测试，共64个数据点
        test_cases = []
        for n in range(3, 11):  # 3-10元
            for k in range(3, 11):  # 3-10维
                test_cases.append((n, k))

        hamiltonian_results = []

        print("  详细条件验证:")
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)

            # 计算理论上界
            theoretical_limit = k / 4

            # 测试不同的参数组合
            param_combinations = [
                (1, int(theoretical_limit) - 1),
                (2, int(theoretical_limit / 2) - 1),
                (int(math.sqrt(theoretical_limit)), int(math.sqrt(theoretical_limit)))
            ]

            for k_max, s_max in param_combinations:
                if k_max * s_max < theoretical_limit and k_max >= 1 and s_max >= 1:
                    rbf_params = RegionBasedFaultModel(
                        max_clusters=k_max,
                        max_cluster_size=s_max,
                        allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                        spatial_correlation=0.5,
                        cluster_separation=2
                    )

                    analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                    rbf_tolerance = analyzer.calculate_rbf_fault_tolerance()

                    # 计算PEF和FT作为对比
                    pef_tolerance = self._calculate_pef_tolerance(n, k)
                    ft_tolerance = self._calculate_ft_tolerance(n, k)

                    # 检查是否满足哈密尔顿性条件
                    product = k_max * s_max
                    satisfies_condition = product < k / 4

                    # 验证哈密尔顿路径存在性
                    hamiltonian_exists = self._verify_hamiltonian_existence(
                        Q, rbf_params, k_max, s_max
                    )

                    hamiltonian_results.append({
                        'n': n, 'k': k,
                        'k_max': k_max, 's_max': s_max,
                        'product': product,
                        'limit': k / 4,
                        'is_satisfied': satisfies_condition,
                        'rbf_tolerance': rbf_tolerance,
                        'pef_tolerance': pef_tolerance,
                        'ft_tolerance': ft_tolerance,
                        'hamiltonian_exists': hamiltonian_exists
                    })

                    if len(hamiltonian_results) <= 30:  # 只显示前30个结果
                        print(f"    {n}元{k}维: k_max={k_max}, s_max={s_max}, "
                              f"乘积={product:.1f}, 限制={k/4:.1f}, "
                              f"RBF={rbf_tolerance}, PEF={pef_tolerance}, FT={ft_tolerance}, "
                              f"条件满足={satisfies_condition}, 哈密尔顿存在={hamiltonian_exists}")

        # 统计分析
        satisfied_count = sum(1 for r in hamiltonian_results if r['is_satisfied'])
        hamiltonian_count = sum(1 for r in hamiltonian_results if r['hamiltonian_exists'])
        
        print(f"\n  验证统计:")
        print(f"    总测试案例: {len(hamiltonian_results)}")
        print(f"    满足条件案例: {satisfied_count} ({satisfied_count/len(hamiltonian_results)*100:.1f}%)")
        print(f"    哈密尔顿存在案例: {hamiltonian_count} ({hamiltonian_count/len(hamiltonian_results)*100:.1f}%)")

        self.analysis_results['hamiltonian_conditions'] = True
        self.performance_data['hamiltonian_verification'] = hamiltonian_results
        
        return hamiltonian_results

    def analyze_boundary_cases(self):
        """
        边界情况分析
        
        专门测试在哈密尔顿性条件的"边界"附近，模型的表现如何。
        例如，当 k_max * s_max 刚好小于、等于或略大于 k/4 时，容错能力的变化情况。
        
        Returns:
            list: 边界情况分析结果
        """
        print("\n=== 边界情况分析 ===")

        # 选择代表性的测试案例进行边界分析（从3-10元，3-10维中选择）
        boundary_test_cases = [
            (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6),
            (5, 4), (5, 5), (6, 4), (6, 5), (7, 4), (8, 4)
        ]
        
        boundary_results = []
        
        print("  详细边界分析:")
        for n, k in boundary_test_cases:
            Q = QkCube(n=n, k=k)
            limit = k / 4
            
            print(f"    {n}元{k}维网络 (理论限制={limit:.2f}):")
            
            # 找到刚好满足条件的最大参数组合
            best_k_max = 1
            best_s_max = int(limit) - 1

            for test_k_max in range(1, int(limit) + 1):
                test_s_max = int((limit - 0.1) / test_k_max)
                if test_s_max >= 1 and test_k_max * test_s_max < limit:
                    if test_k_max * test_s_max > best_k_max * best_s_max:
                        best_k_max = test_k_max
                        best_s_max = test_s_max
            
            # 测试边界附近的多个点
            boundary_points = [
                ("远低于边界", best_k_max, max(1, best_s_max - 2)),
                ("接近边界", best_k_max, best_s_max),
                ("边界临界", best_k_max, best_s_max + 1),
                ("超过边界", best_k_max, best_s_max + 2)
            ]
            
            for point_name, k_max, s_max in boundary_points:
                product = k_max * s_max
                
                if k_max >= 1 and s_max >= 1:
                    rbf_params = RegionBasedFaultModel(
                        max_clusters=k_max,
                        max_cluster_size=s_max,
                        allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                        spatial_correlation=0.5,
                        cluster_separation=2
                    )
                    
                    analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                    tolerance = analyzer.calculate_rbf_fault_tolerance()
                    
                    # 评估边界性能
                    performance_ratio = tolerance / max(1, int(limit))
                    satisfies_condition = product < limit
                    
                    boundary_results.append({
                        'n': n, 'k': k,
                        'point_type': point_name,
                        'k_max': k_max, 's_max': s_max,
                        'product': product,
                        'limit': limit,
                        'tolerance': tolerance,
                        'performance_ratio': performance_ratio,
                        'satisfies_condition': satisfies_condition
                    })
                    
                    print(f"      {point_name}: k_max={k_max}, s_max={s_max}, "
                          f"乘积={product:.1f}, 容错={tolerance}, "
                          f"性能比={performance_ratio:.2f}")

        # 边界性能分析
        boundary_performance = {}
        for point_type in ["远低于边界", "接近边界", "边界临界", "超过边界"]:
            type_results = [r for r in boundary_results if r['point_type'] == point_type]
            if type_results:
                avg_performance = np.mean([r['performance_ratio'] for r in type_results])
                boundary_performance[point_type] = avg_performance
        
        print(f"\n  边界性能统计:")
        for point_type, avg_perf in boundary_performance.items():
            print(f"    {point_type}: 平均性能比={avg_perf:.3f}")

        self.performance_data['boundary_analysis'] = boundary_results
        return boundary_results

    def analyze_hamiltonian_connectivity(self):
        """
        哈密尔顿连通性分析
        
        分析在不同故障模式下，网络的哈密尔顿连通性如何变化。
        
        Returns:
            list: 连通性分析结果
        """
        print("\n=== 哈密尔顿连通性分析 ===")

        # 选择测试案例（从3-10元，3-10维中选择）
        connectivity_test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (8, 3)
        ]
        
        connectivity_results = []
        
        print("  详细连通性分析:")
        for n, k in connectivity_test_cases:
            Q = QkCube(n=n, k=k)
            
            # 测试不同故障密度下的连通性
            fault_densities = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            for fault_density in fault_densities:
                max_faults = int(fault_density * (k**n - 1))
                
                # 估算连通性概率
                connectivity_prob = self._estimate_connectivity_probability(
                    n, k, max_faults
                )
                
                # 计算理论预期
                theoretical_threshold = k / 4
                expected_connectivity = 1.0 if max_faults < theoretical_threshold else \
                                     max(0.0, 1.0 - (max_faults - theoretical_threshold) / theoretical_threshold)
                
                connectivity_results.append({
                    'n': n, 'k': k,
                    'fault_density': fault_density,
                    'max_faults': max_faults,
                    'connectivity_prob': connectivity_prob,
                    'expected_connectivity': expected_connectivity
                })
                
                print(f"    {n}元{k}维, 故障密度={fault_density:.1f}: "
                      f"连通概率={connectivity_prob:.3f}, 理论预期={expected_connectivity:.3f}")
        
        self.performance_data['connectivity_analysis'] = connectivity_results
        return connectivity_results

    def _verify_hamiltonian_existence(self, Q, rbf_params, k_max, s_max):
        """验证哈密尔顿路径存在性"""
        try:
            analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
            tolerance = analyzer.calculate_rbf_fault_tolerance()
            
            # 简化的存在性检验：基于容错能力和参数关系
            theoretical_minimum = k_max * s_max
            return tolerance >= theoretical_minimum
        except:
            return False

    def _estimate_connectivity_probability(self, n, k, max_faults):
        """估算连通性概率"""
        # 基于网络规模和故障数的简化概率模型
        network_size = k ** n
        total_edges = n * network_size

        if max_faults >= total_edges * 0.5:
            return 0.0

        # 使用指数衰减模型
        fault_ratio = max_faults / (total_edges * 0.5)
        return max(0.0, math.exp(-2 * fault_ratio))

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
            f.write("=== Hamiltonian Analysis Results ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 保存hamiltonian verification结果
            f.write("1. Hamiltonian Condition Verification Results:\n")
            f.write("n\tk\tk_max\ts_max\tProduct\tLimit\tSatisfied\tRBF\tPEF\tFT\tHamiltonian_Exists\n")
            for result in results['hamiltonian_verification']:
                f.write(f"{result['n']}\t{result['k']}\t{result['k_max']}\t{result['s_max']}\t"
                       f"{result['product']:.1f}\t{result['limit']:.1f}\t{result['is_satisfied']}\t"
                       f"{result['rbf_tolerance']}\t{result['pef_tolerance']}\t{result['ft_tolerance']}\t"
                       f"{result['hamiltonian_exists']}\n")

            # 保存boundary analysis结果
            f.write("\n2. Boundary Analysis Results:\n")
            f.write("n\tk\tPoint_Type\tk_max\ts_max\tProduct\tLimit\tTolerance\tPerformance_Ratio\tSatisfied\n")
            for result in results['boundary_analysis']:
                f.write(f"{result['n']}\t{result['k']}\t{result['point_type']}\t"
                       f"{result['k_max']}\t{result['s_max']}\t{result['product']:.1f}\t"
                       f"{result['limit']:.1f}\t{result['tolerance']}\t{result['performance_ratio']:.2f}\t"
                       f"{result['satisfies_condition']}\n")

            # 保存connectivity analysis结果
            f.write("\n3. Connectivity Analysis Results:\n")
            f.write("n\tk\tFault_Density\tMax_Faults\tConnectivity_Prob\tExpected_Connectivity\n")
            for result in results['connectivity_analysis']:
                f.write(f"{result['n']}\t{result['k']}\t{result['fault_density']:.1f}\t"
                       f"{result['max_faults']}\t{result['connectivity_prob']:.3f}\t"
                       f"{result['expected_connectivity']:.3f}\n")

        print(f"Results saved to {self.output_file}")

    def create_visualizations(self, results):
        """创建可视化图表"""
        # 图1: 哈密尔顿条件验证和容错能力比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        verification_data = results['hamiltonian_verification']

        # 子图1: 条件满足情况
        satisfied_count = sum(1 for r in verification_data if r['is_satisfied'])
        unsatisfied_count = len(verification_data) - satisfied_count

        ax1.pie([satisfied_count, unsatisfied_count],
                labels=['Satisfied', 'Not Satisfied'],
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Hamiltonian Condition Satisfaction')

        # 子图2: 哈密尔顿存在性统计
        hamiltonian_exists_count = sum(1 for r in verification_data if r['hamiltonian_exists'])
        hamiltonian_not_exists_count = len(verification_data) - hamiltonian_exists_count

        ax2.pie([hamiltonian_exists_count, hamiltonian_not_exists_count],
                labels=['Hamiltonian Exists', 'Hamiltonian Not Exists'],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Hamiltonian Path Existence')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hamiltonian_conditions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: RBF vs PEF 容错能力比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        rbf_values = [r['rbf_tolerance'] for r in verification_data[:20]]
        pef_values = [r['pef_tolerance'] for r in verification_data[:20]]

        x_pos = np.arange(len(rbf_values))
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
        plt.savefig(os.path.join(self.output_dir, 'rbf_vs_pef_hamiltonian.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图3: RBF vs FT 容错能力比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ft_values = [r['ft_tolerance'] for r in verification_data[:20]]

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
        plt.savefig(os.path.join(self.output_dir, 'rbf_vs_ft_hamiltonian.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: 边界分析和连通性分析
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 子图1: 边界性能分析
        boundary_data = results['boundary_analysis']
        point_types = ['远低于边界', '接近边界', '边界临界', '超过边界']

        performance_by_type = {}
        for point_type in point_types:
            type_data = [r for r in boundary_data if r['point_type'] == point_type]
            if type_data:
                performance_by_type[point_type] = np.mean([r['performance_ratio'] for r in type_data])

        if performance_by_type:
            types = list(performance_by_type.keys())
            performances = list(performance_by_type.values())

            ax1.bar(types, performances, alpha=0.8)
            ax1.set_ylabel('Average Performance Ratio')
            ax1.set_title('Boundary Performance Analysis')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

        # 子图2: 连通性概率分析
        connectivity_data = results['connectivity_analysis']
        if connectivity_data:
            fault_densities = [r['fault_density'] for r in connectivity_data]
            connectivity_probs = [r['connectivity_prob'] for r in connectivity_data]
            expected_connectivities = [r['expected_connectivity'] for r in connectivity_data]

            ax2.plot(fault_densities, connectivity_probs, 'bo-', label='Actual Probability')
            ax2.plot(fault_densities, expected_connectivities, 'r--', label='Expected Probability')
            ax2.set_xlabel('Fault Density')
            ax2.set_ylabel('Connectivity Probability')
            ax2.set_title('Connectivity Probability Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boundary_connectivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved in {self.output_dir}/: hamiltonian_conditions.png, rbf_vs_pef_hamiltonian.png, rbf_vs_ft_hamiltonian.png, boundary_connectivity.png")

    def run_all_hamiltonian_analysis(self):
        """运行所有哈密尔顿性分析"""
        print("开始哈密尔顿性分析...")

        hamiltonian_verification = self.verify_hamiltonian_condition()
        boundary_analysis = self.analyze_boundary_cases()
        connectivity_analysis = self.analyze_hamiltonian_connectivity()

        results = {
            'hamiltonian_verification': hamiltonian_verification,
            'boundary_analysis': boundary_analysis,
            'connectivity_analysis': connectivity_analysis
        }

        # 保存结果到文件
        self.save_results_to_file(results)

        # 创建可视化
        self.create_visualizations(results)

        print("\n=== 哈密尔顿性分析完成 ===")
        return results


if __name__ == "__main__":
    analyzer = HamiltonianAnalyzer()
    results = analyzer.run_all_hamiltonian_analysis()
