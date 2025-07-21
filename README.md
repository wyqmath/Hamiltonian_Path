# 基于区域/簇的故障模型 (Region-Based Fault Model)

## 概述

本项目实现了一种创新的故障模型——基于区域/簇的故障模型（RBF），用于k元n维立方体网络中的哈密尔顿路径嵌入问题。与传统的分区边故障（PEF）模型不同，RBF模型将故障视为空间聚集的"簇"，更符合实际系统中故障的空间相关性特征。

## 核心创新

- **故障簇模型**：故障形成空间连通的簇，支持多种簇形状（完全图、星形图、路径图等）
- **区域感知算法**：自动识别故障簇并分析其拓扑结构
- **递归路径构造**：利用网络递归结构和最优维度选择构建哈密尔顿路径

## 理论优势

- **容错能力提升**：RBF模型容错上界 Θ_RBF = k_max × s_max × α(n, k, d_sep)
- **实际提升效果**：在各种网络配置下容错能力提升54.5%-95.8%
- **数学严谨性**：所有理论公式与实现完全一致（误差 < 1e-10）

## 文件结构

```
Hamiltonian_Path/
├── region_based_fault_model.py    # 核心算法实现
├── mathematical_theory.md         # 完整的数学理论推导和验证
├── comprehensive_theory_test.py   # 理论验证测试程序
└── README.md                      # 本文档
```

## 核心组件

- **FaultCluster**: 故障簇数据结构，包含簇内故障边、受影响节点、形状类型等
- **RegionBasedFaultAnalyzer**: 故障分析器，负责识别故障簇和计算容错上界
- **RegionBasedHamiltonianEmbedding**: 哈密尔顿路径嵌入算法的核心实现

## 使用示例

```python
from region_based_fault_model import *

# 创建网络和RBF参数
Q = QkCube(n=3, k=5)
rbf_params = RegionBasedFaultModel(
    max_clusters=3,
    max_cluster_size=8,
    allowed_shapes=[ClusterShape.CUSTOM],
    spatial_correlation=0.7,
    cluster_separation=2
)

# 分析故障分布并嵌入哈密尔顿路径
analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
clusters = analyzer.analyze_fault_distribution(fault_edges)
embedding = RegionBasedHamiltonianEmbedding(Q, rbf_params)
path = embedding.embed_hamiltonian_path_rbf(fault_edges, source, target)
```

## 算法复杂度

- **时间复杂度**: O(n² × k^n + n × |C|² × s_max²)
- **空间复杂度**: O(k^n + n × |C| × s_max)

其中 |C| 是故障簇数量，s_max 是最大簇大小。

## 与PEF模型对比

| 特性 | PEF模型 | RBF模型 |
|------|---------|---------|
| 故障模式 | 独立边故障 | 聚集簇故障 |
| 容错上界 | Θ_PEF | Θ_RBF > Θ_PEF |
| 实际适用性 | 理论模型 | 更符合实际 |
| 提升比例 | - | 54.5%-95.8% |

## 应用场景

- **数据中心网络**: 机架级故障、交换机故障、电源故障
- **片上网络 (NoC)**: 制造缺陷、热点故障、老化故障
- **无线传感器网络**: 环境干扰、物理损坏、能量耗尽

## 运行测试

```bash
# 运行理论验证测试
python comprehensive_theory_test.py
```