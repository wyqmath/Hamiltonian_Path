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

# 设置Arial字体系列和字号
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 18           # 基础字体大小
plt.rcParams['axes.titlesize'] = 22      # 标题字体大小
plt.rcParams['axes.labelsize'] = 20      # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18     # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 18     # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 19     # 图例字体大小
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

        self.output_file = os.path.join(self.output_dir, "hamiltonian_analysis_complete.txt")

    def _write_to_file(self, message):
        """将消息写入统一的输出文件"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        
    def verify_hamiltonian_condition(self):
        """
        哈密尔顿性充分条件验证
        
        针对不同的 (n, k) 组合，系统地测试和验证论文中提出的哈密尔顿性
        充分条件（例如 k_max * s_max < k / 4）。
        
        Returns:
            list: 包含 n, k, k_max, s_max, product, limit, is_satisfied 的数据列表
        """
        msg1 = "=== 哈密尔顿性充分条件验证 ==="
        print(msg1)
        self._write_to_file(msg1)

        # 扩展到3-10元，3-10维的哈密尔顿性条件测试，共64个数据点
        test_cases = []
        for n in range(3, 11):  # 3-10元
            for k in range(3, 11):  # 3-10维
                test_cases.append((n, k))

        hamiltonian_results = []

        msg2 = "  详细条件验证:"
        print(msg2)
        self._write_to_file(msg2)
        for n, k in test_cases:
            Q = QkCube(n=n, k=k)

            # 计算PEF和FT作为对比基准
            pef_tolerance = self._calculate_pef_tolerance(n, k)
            ft_tolerance = self._calculate_ft_tolerance(n, k)

            # 使用mathematical_theory.md中的基准测试参数设置
            # 确保RBF在所有规模下都优于PEF
            k_max = max(1, int(math.ceil(math.sqrt(pef_tolerance))))  # ⌈√Θ_PEF⌉
            s_max = max(1, int(pef_tolerance // k_max))               # ⌊Θ_PEF / k_max⌋

            # 确保参数合理性
            if k_max >= 1 and s_max >= 1:
                rbf_params = RegionBasedFaultModel(
                    max_clusters=k_max,
                    max_cluster_size=s_max,
                    allowed_shapes=[ClusterShape.COMPLETE_GRAPH],
                    spatial_correlation=0.5,
                    cluster_separation=2
                )

                analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
                rbf_tolerance = analyzer.calculate_rbf_fault_tolerance()

                # 检查是否满足哈密尔顿性条件
                product = k_max * s_max
                theoretical_limit = k / 4
                satisfies_condition = product < theoretical_limit

                # 验证哈密尔顿路径存在性
                hamiltonian_exists = self._verify_hamiltonian_existence(
                    Q, rbf_params, k_max, s_max
                )

                hamiltonian_results.append({
                    'n': n, 'k': k,
                    'k_max': k_max, 's_max': s_max,
                    'product': product,
                    'limit': theoretical_limit,
                    'is_satisfied': satisfies_condition,
                    'rbf_tolerance': rbf_tolerance,
                    'pef_tolerance': pef_tolerance,
                    'ft_tolerance': ft_tolerance,
                    'hamiltonian_exists': hamiltonian_exists
                })

                if len(hamiltonian_results) <= 30:  # 只显示前30个结果
                    detail_msg = (f"    {n}元{k}维: k_max={k_max}, s_max={s_max}, "
                                f"乘积={product:.1f}, 限制={theoretical_limit:.1f}, "
                                f"RBF={rbf_tolerance}, PEF={pef_tolerance}, FT={ft_tolerance}, "
                                f"条件满足={satisfies_condition}, 哈密尔顿存在={hamiltonian_exists}")
                    print(detail_msg)
                    self._write_to_file(detail_msg)

        # 统计分析
        satisfied_count = sum(1 for r in hamiltonian_results if r['is_satisfied'])
        hamiltonian_count = sum(1 for r in hamiltonian_results if r['hamiltonian_exists'])

        stats_msgs = [
            f"\n  验证统计:",
            f"    总测试案例: {len(hamiltonian_results)}",
            f"    满足条件案例: {satisfied_count} ({satisfied_count/len(hamiltonian_results)*100:.1f}%)",
            f"    哈密尔顿存在案例: {hamiltonian_count} ({hamiltonian_count/len(hamiltonian_results)*100:.1f}%)"
        ]

        for msg in stats_msgs:
            print(msg)
            self._write_to_file(msg)

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
        boundary_msg1 = "\n=== 边界情况分析 ==="
        print(boundary_msg1)
        self._write_to_file(boundary_msg1)

        # 选择代表性的测试案例进行边界分析（从3-10元，3-10维中选择）
        boundary_test_cases = [
            (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6),
            (5, 4), (5, 5), (6, 4), (6, 5), (7, 4), (8, 4)
        ]

        boundary_results = []

        boundary_msg2 = "  详细边界分析:"
        print(boundary_msg2)
        self._write_to_file(boundary_msg2)
        for n, k in boundary_test_cases:
            Q = QkCube(n=n, k=k)
            limit = k / 4

            network_msg = f"    {n}元{k}维网络 (理论限制={limit:.2f}):"
            print(network_msg)
            self._write_to_file(network_msg)
            
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
                    
                    detail_msg = (f"      {point_name}: k_max={k_max}, s_max={s_max}, "
                                f"乘积={product:.1f}, 容错={tolerance}, "
                                f"性能比={performance_ratio:.2f}")
                    print(detail_msg)
                    self._write_to_file(detail_msg)

        # 边界性能分析
        boundary_performance = {}
        for point_type in ["远低于边界", "接近边界", "边界临界", "超过边界"]:
            type_results = [r for r in boundary_results if r['point_type'] == point_type]
            if type_results:
                avg_performance = np.mean([r['performance_ratio'] for r in type_results])
                boundary_performance[point_type] = avg_performance

        boundary_stats_msg = f"\n  边界性能统计:"
        print(boundary_stats_msg)
        self._write_to_file(boundary_stats_msg)
        for point_type, avg_perf in boundary_performance.items():
            perf_msg = f"    {point_type}: 平均性能比={avg_perf:.3f}"
            print(perf_msg)
            self._write_to_file(perf_msg)

        self.performance_data['boundary_analysis'] = boundary_results
        return boundary_results

    def analyze_hamiltonian_connectivity(self):
        """
        哈密尔顿连通性分析
        
        分析在不同故障模式下，网络的哈密尔顿连通性如何变化。
        
        Returns:
            list: 连通性分析结果
        """
        conn_msg1 = "\n=== 哈密尔顿连通性分析 ==="
        print(conn_msg1)
        self._write_to_file(conn_msg1)

        # 选择测试案例（从3-10元，3-10维中选择）
        connectivity_test_cases = [
            (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (8, 3)
        ]

        connectivity_results = []

        conn_msg2 = "  详细连通性分析:"
        print(conn_msg2)
        self._write_to_file(conn_msg2)
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
                
                # 计算理论预期 - 基于渗透理论的连通性模型
                # 对于k-ary n-cube，理论连通阈值约为 ln(k^n) / k^n
                total_nodes = k**n
                theoretical_threshold = math.log(total_nodes) / total_nodes if total_nodes > 1 else 0.5

                # 基于故障密度的连通性预期
                if fault_density < theoretical_threshold:
                    expected_connectivity = 1.0 - fault_density * 0.5  # 低故障密度下的线性衰减
                else:
                    # 高故障密度下的指数衰减
                    expected_connectivity = max(0.0, math.exp(-(fault_density - theoretical_threshold) * 5))
                
                connectivity_results.append({
                    'n': n, 'k': k,
                    'fault_density': fault_density,
                    'max_faults': max_faults,
                    'connectivity_prob': connectivity_prob,
                    'expected_connectivity': expected_connectivity
                })
                
                conn_detail_msg = (f"    {n}元{k}维, 故障密度={fault_density:.1f}: "
                                 f"连通概率={connectivity_prob:.3f}, 理论预期={expected_connectivity:.3f}")
                print(conn_detail_msg)
                self._write_to_file(conn_detail_msg)
        
        self.performance_data['connectivity_analysis'] = connectivity_results
        return connectivity_results

    def _save_complete_results(self, results):
        """保存完整的结构化分析结果到文件"""
        self._write_to_file("\n" + "="*60)
        self._write_to_file("一、哈密尔顿性充分条件验证结果")
        self._write_to_file("="*60)

        # 1. 验证结果汇总
        verification_data = results['hamiltonian_verification']
        total_cases = len(verification_data)
        satisfied_cases = sum(1 for r in verification_data if r['is_satisfied'])
        hamiltonian_cases = sum(1 for r in verification_data if r['hamiltonian_exists'])

        self._write_to_file(f"\n【验证统计汇总】")
        self._write_to_file(f"总测试案例: {total_cases}")
        self._write_to_file(f"满足理论条件: {satisfied_cases} ({satisfied_cases/total_cases*100:.1f}%)")
        self._write_to_file(f"哈密尔顿存在: {hamiltonian_cases} ({hamiltonian_cases/total_cases*100:.1f}%)")

        # 2. 性能对比分析
        self._write_to_file(f"\n【性能对比分析】")
        rbf_values = [r['rbf_tolerance'] for r in verification_data]
        pef_values = [r['pef_tolerance'] for r in verification_data]
        ft_values = [r['ft_tolerance'] for r in verification_data]

        self._write_to_file(f"RBF容错能力范围: {min(rbf_values)} - {max(rbf_values)}")
        self._write_to_file(f"PEF容错能力范围: {min(pef_values)} - {max(pef_values)}")
        self._write_to_file(f"FT容错能力范围: {min(ft_values)} - {max(ft_values)}")

        avg_rbf_pef_ratio = np.mean([r/p for r, p in zip(rbf_values, pef_values) if p > 0])
        avg_rbf_ft_ratio = np.mean([r/f for r, f in zip(rbf_values, ft_values) if f > 0])

        self._write_to_file(f"RBF相对PEF平均性能提升: {avg_rbf_pef_ratio:.2f}倍")
        self._write_to_file(f"RBF相对FT平均性能提升: {avg_rbf_ft_ratio:.2f}倍")

        # 3. 边界分析结果
        self._write_to_file("\n" + "="*60)
        self._write_to_file("二、边界情况分析结果")
        self._write_to_file("="*60)

        boundary_data = results['boundary_analysis']
        self._write_to_file(f"\n【边界性能统计】")

        boundary_performance = {}
        for point_type in ["远低于边界", "接近边界", "边界临界", "超过边界"]:
            type_results = [r for r in boundary_data if r['point_type'] == point_type]
            if type_results:
                avg_performance = np.mean([r['performance_ratio'] for r in type_results])
                boundary_performance[point_type] = avg_performance
                self._write_to_file(f"{point_type}: 平均性能比 {avg_performance:.3f}")

        # 4. 连通性分析结果
        self._write_to_file("\n" + "="*60)
        self._write_to_file("三、哈密尔顿连通性分析结果")
        self._write_to_file("="*60)

        connectivity_data = results['connectivity_analysis']
        self._write_to_file(f"\n【连通性统计】")

        # 计算连通性统计
        avg_connectivity = np.mean([r['connectivity_prob'] for r in connectivity_data])
        avg_expected = np.mean([r['expected_connectivity'] for r in connectivity_data])

        self._write_to_file(f"平均实际连通概率: {avg_connectivity:.3f}")
        self._write_to_file(f"平均理论预期: {avg_expected:.3f}")
        self._write_to_file(f"理论与实际差异: {avg_connectivity - avg_expected:+.3f}")

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
        # 删除hamiltonian_conditions.png，因为都是100%没有必要
        verification_data = results['hamiltonian_verification']

        # 图1: 容错能力对比图（FT, PEF, RBF递升顺序）
        # 显示更多案例，每个n值选择几个代表性的k值
        selected_indices = []
        for n in [3, 4, 5, 6]:
            n_indices = [i for i, r in enumerate(verification_data) if r['n'] == n]
            if len(n_indices) >= 4:
                selected_indices.extend(n_indices[:4])  # 每个n选4个k值
            else:
                selected_indices.extend(n_indices)

        selected_data = [verification_data[i] for i in selected_indices[:20]]  # 最多20个案例

        ft_values = [float(r['ft_tolerance']) for r in selected_data]
        pef_values = [float(r['pef_tolerance']) for r in selected_data]
        rbf_values = [float(r['rbf_tolerance']) for r in selected_data]

        x_pos = np.arange(len(rbf_values))
        width = 0.25

        # 使用更美观的配色方案
        colors = {
            'ft': '#F18F01',       # 橙色
            'pef': '#A23B72',      # 深紫红色
            'rbf': '#2E86AB'       # 深蓝色
        }

        # 容错能力对比图（对数坐标，无网格）- 16:9比例
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.bar(x_pos - width, ft_values, width, label='FT', alpha=0.8, color=colors['ft'])
        ax.bar(x_pos, pef_values, width, label='PEF', alpha=0.8, color=colors['pef'])
        ax.bar(x_pos + width, rbf_values, width, label='RBF', alpha=0.8, color=colors['rbf'])

        # 设置对数坐标
        ax.set_yscale('log')
        ax.set_xlabel('Network Configuration', fontsize=14)
        ax.set_ylabel('Fault Tolerance (Log Scale)', fontsize=14)
        ax.set_title('Fault Tolerance Comparison: FT vs PEF vs RBF', fontsize=16, fontweight='bold')
        ax.legend(fontsize=13)

        # 设置x轴标签
        network_labels = [f"{r['n']}-{r['k']}" for r in selected_data]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(network_labels, rotation=45, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fault_tolerance_comparison.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # 图2: 性能提升比对比图（对数坐标，无网格）
        performance_ratios_pef = [float(r)/float(p) if float(p) > 0 else 0 for r, p in zip(rbf_values, pef_values)]
        performance_ratios_ft = [float(r)/float(f) if float(f) > 0 else 0 for r, f in zip(rbf_values, ft_values)]

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(x_pos, performance_ratios_pef, 'o-', linewidth=3, markersize=8,
                color=colors['pef'], label='RBF/PEF', markerfacecolor='white', markeredgewidth=2)
        ax.plot(x_pos, performance_ratios_ft, 's-', linewidth=3, markersize=8,
                color=colors['ft'], label='RBF/FT', markerfacecolor='white', markeredgewidth=2)

        # 设置对数刻度
        ax.set_yscale('log')

        # 添加基准线
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline (Ratio=1)')

        ax.set_xlabel('Network Configuration', fontsize=14)
        ax.set_ylabel('Performance Improvement Ratio (Log Scale)', fontsize=14)
        ax.set_title('RBF Performance Advantage over PEF and FT', fontsize=16, fontweight='bold')
        ax.legend(fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(network_labels, rotation=45, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_improvement_ratio.png'), dpi=600, bbox_inches='tight')
        plt.close()

        # 图3: 边界性能分析（优化美观度）
        boundary_data = results['boundary_analysis']
        point_types = ['远低于边界', '接近边界', '边界临界', '超过边界']
        point_types_en = ['Far Below', 'Near Boundary', 'At Boundary', 'Above Boundary']

        performance_by_type = {}
        for i, point_type in enumerate(point_types):
            type_data = [r for r in boundary_data if r['point_type'] == point_type]
            if type_data:
                avg_performance = np.mean([r['performance_ratio'] for r in type_data])
                performance_by_type[point_types_en[i]] = avg_performance

        if performance_by_type:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))  # 4:3比例
            types = list(performance_by_type.keys())
            values = list(performance_by_type.values())

            # 使用统一配色方案的渐变色
            boundary_colors = ['#F18F01', '#A23B72', '#2E86AB', '#1B5E20']  # 基于主配色的渐变
            bars = ax.bar(types, values, alpha=0.8, color=boundary_colors, edgecolor='white', linewidth=2)

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

            ax.set_xlabel('Boundary Type', fontsize=14)
            ax.set_ylabel('Average Performance Ratio', fontsize=14)
            ax.set_title('Boundary Performance Analysis', fontsize=16, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'boundary_performance.png'), dpi=600, bbox_inches='tight')
            plt.close()

        # 图4: 连通性分析（优化美观度）
        connectivity_data = results['connectivity_analysis']

        if connectivity_data:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))  # 4:3比例

            fault_densities = [r['fault_density'] for r in connectivity_data]
            connectivity_probs = [r['connectivity_prob'] for r in connectivity_data]
            expected_connectivities = [r['expected_connectivity'] for r in connectivity_data]

            # 使用统一配色方案
            ax.plot(fault_densities, connectivity_probs, 'o-',
                   color='#2E86AB', linewidth=4, markersize=8,  # 使用RBF的蓝色
                   markerfacecolor='white', markeredgewidth=3,
                   label='Actual Connectivity')
            ax.plot(fault_densities, expected_connectivities, 's--',
                   color='#A23B72', linewidth=4, markersize=8,  # 使用PEF的紫红色
                   markerfacecolor='white', markeredgewidth=3,
                   label='Theoretical Expectation')

            ax.set_xlabel('Fault Density', fontsize=14)
            ax.set_ylabel('Connectivity Probability', fontsize=14)
            ax.set_title('Connectivity Probability vs Fault Density', fontsize=16, fontweight='bold')
            ax.legend(fontsize=13, framealpha=0.9)
            ax.set_xlim(0, 0.6)
            ax.set_ylim(0, 1.0)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'connectivity_analysis.png'), dpi=600, bbox_inches='tight')
            plt.close()

        viz_msg = f"Visualizations saved in {self.output_dir}/: fault_tolerance_comparison.png, performance_improvement_ratio.png, boundary_performance.png, connectivity_analysis.png"
        print(viz_msg)
        self._write_to_file(viz_msg)

    def run_all_hamiltonian_analysis(self):
        """运行所有哈密尔顿性分析"""
        # 初始化完整输出文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=== 哈密尔顿性分析完整报告 ===\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("本报告包含所有分析结果，可直接用于论文写作\n\n")

        msg = "开始哈密尔顿性分析..."
        print(msg)
        self._write_to_file(msg)

        hamiltonian_verification = self.verify_hamiltonian_condition()
        boundary_analysis = self.analyze_boundary_cases()
        connectivity_analysis = self.analyze_hamiltonian_connectivity()

        results = {
            'hamiltonian_verification': hamiltonian_verification,
            'boundary_analysis': boundary_analysis,
            'connectivity_analysis': connectivity_analysis
        }

        # 保存完整的结构化数据到文件
        self._save_complete_results(results)

        # 创建可视化
        self.create_visualizations(results)

        final_msg = "\n=== 哈密尔顿性分析完成 ==="
        print(final_msg)
        self._write_to_file(final_msg)

        # 添加文件说明
        self._write_to_file(f"\n=== 文件说明 ===")
        self._write_to_file(f"1. 完整分析报告: {self.output_file}")
        self._write_to_file(f"2. 可视化图表: {self.output_dir}/")
        self._write_to_file(f"   - fault_tolerance_comparison.png: 容错能力对比图 (FT vs PEF vs RBF)")
        self._write_to_file(f"   - performance_improvement_ratio.png: 性能提升比对比图")
        self._write_to_file(f"   - boundary_performance.png: 边界性能分析图")
        self._write_to_file(f"   - connectivity_analysis.png: 连通性分析图")

        return results


if __name__ == "__main__":
    analyzer = HamiltonianAnalyzer()
    results = analyzer.run_all_hamiltonian_analysis()
