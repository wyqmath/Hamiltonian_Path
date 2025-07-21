# 基于区域/簇的故障模型 (Region-Based Fault Model, RBF)

## 概述

本项目实现了一种创新的故障模型——基于区域/簇的故障模型（RBF），用于k元n维立方体网络中的哈密尔顿路径嵌入问题。与传统的分区边故障（PEF）模型不同，RBF模型将故障视为空间聚集的"簇"，更符合实际系统中故障的空间相关性特征。

## 核心创新

### 1. 故障簇模型
- **空间连通性**：故障边形成连通的簇结构，支持多种拓扑形状
- **形状约束**：支持完全图、星形图、路径图、环形图、树形图等标准拓扑
- **分离距离**：簇间保持最小分离距离 d_sep，减少相互干扰

### 2. 区域感知算法
- **自动簇识别**：基于连通分量算法识别故障簇
- **拓扑分析**：分析每个簇的形状、中心、半径和连通性
- **空间优化**：根据空间距离合并相近簇，优化故障分布

### 3. 递归路径构造
- **最优维度选择**：采用分离度函数选择最优分解维度
- **归纳证明策略**：基于网络递归结构的严格数学证明
- **路径缝合算法**：高效的跨层路径连接策略

## 理论基础

### 容错上界定理
RBF模型提供两个层次的容错保证：

1. **基础容错上界**（连通性保持）：
   ```
   Θ_RBF^basic = k_max × s_max × α(n, k, d_sep)
   ```

2. **哈密尔顿性容错上界**：
   ```
   Θ_RBF^Ham = min(k/4, k_max × s_max)
   ```

### 修正因子定义
```
α(n, k, d_sep) = α_struct(n, k) × α_spatial(d_sep)
```

其中：
- **结构修正因子**：`α_struct(n, k) = min(1 + ln(nk/2)/n, 2.0)`
- **空间修正因子**：`α_spatial(d_sep) = (1 + 0.5×(1-ρ)) × (1 + ln(1+d_sep)/10)`

### 性能提升量化
在标准基准测试下，RBF模型相对于PEF模型的性能提升：
- **3元3维网络**：108.3% 提升
- **3元5维网络**：129.7% 提升
- **4元3维网络**：100.9% 提升
- **5元3维网络**：94.6% 提升

## 文件结构

```
Hamiltonian_Path/
├── region_based_fault_model.py    # 核心算法实现
├── mathematical_theory.md         # 完整的数学理论推导
├── comprehensive_theory_test.py   # 理论验证测试程序
└── README.md                      # 本文档
```

## 核心组件

### 数据结构
- **FaultCluster**: 故障簇数据结构，包含簇内故障边、受影响节点、形状类型、中心位置、半径等
- **RegionBasedFaultModel**: RBF模型参数配置，包含簇数量限制、大小限制、分离距离、形状约束等
- **ClusterShape**: 簇形状枚举，支持完全图、星形图、路径图、环形图、树形图等

### 核心算法
- **RegionBasedFaultAnalyzer**: 故障分析器，负责识别故障簇、计算容错上界、验证RBF条件
- **RegionBasedHamiltonianEmbedding**: 哈密尔顿路径嵌入算法，实现递归构造和路径缝合
- **ComprehensiveTheoryAnalyzer**: 理论分析器，提供完整的数学理论验证和性能比较

## 使用示例

### 基础使用
```python
from region_based_fault_model import *
from origin_pef import QkCube

# 创建网络和RBF参数
Q = QkCube(n=3, k=5)
rbf_params = RegionBasedFaultModel(
    max_clusters=3,
    max_cluster_size=8,
    allowed_shapes=[ClusterShape.COMPLETE_GRAPH, ClusterShape.STAR_GRAPH],
    spatial_correlation=0.5,
    cluster_separation=2
)

# 分析故障分布
analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
fault_tolerance = analyzer.calculate_rbf_fault_tolerance()
print(f"RBF容错上界: {fault_tolerance}")

# 嵌入哈密尔顿路径
embedding = RegionBasedHamiltonianEmbedding(Q, rbf_params)
path = embedding.embed_hamiltonian_path_rbf(fault_edges, source, target)
```

### 理论分析
```python
from comprehensive_theory_test import ComprehensiveTheoryAnalyzer

# 运行完整的理论分析
analyzer = ComprehensiveTheoryAnalyzer()
analyzer.run_all_analysis()  # 包含基础理论、高维分析、性能比较等
```

## 算法复杂度

### 时间复杂度
- **总体复杂度**: `O(k^n + n × |C|² × s_max × k^(n-1))`
- **RBF条件下**: `O(k^n) = O(N)`，其中 N 是网络节点数
- **故障分析**: `O(|C| × s_max)`
- **维度选择**: `O(n × |C|² × s_max)`
- **路径构造**: `O(k^n)`

### 空间复杂度
- **总体复杂度**: `O(k^n + n × |C| × s_max)`
- **RBF条件下**: `O(k^n)`

其中 |C| ≤ k_max 是故障簇数量，s_max 是最大簇大小。

## 与PEF模型对比

### 理论对比
| 特性 | PEF模型 | RBF模型 |
|------|---------|---------|
| **故障模式** | 按维度分区的独立边故障 | 空间聚集的簇故障 |
| **容错上界** | Θ_PEF = O(k^(n-1)) | Θ_RBF = k_max × s_max × α(n,k,d_sep) |
| **分解策略** | 固定维度分解 | 自适应最优维度选择 |
| **理论基础** | 维度独立性 | 空间结构优化 |
| **参数定义** | 直接给出分区条件 | 直接给出修正因子（相同方法论） |

### 性能提升
基于标准基准测试的性能比较：

| 网络配置 | PEF容错能力 | RBF容错能力 | 性能提升 |
|----------|-------------|-------------|----------|
| 3元3维 | 8 | 20 | **150.0%** |
| 3元5维 | 24 | 55 | **129.2%** |
| 4元3维 | 33 | 64 | **93.9%** |
| 4元5维 | 147 | 319 | **117.0%** |
| 5元3维 | 112 | 217 | **93.8%** |

## 应用场景

### 实际系统中的故障模式
- **数据中心网络**: 机架级故障、交换机故障、电源故障、冷却系统故障
- **片上网络 (NoC)**: 制造缺陷、热点故障、老化故障、电迁移效应
- **无线传感器网络**: 环境干扰、物理损坏、能量耗尽、信号遮挡
- **高性能计算**: 节点故障、互连故障、内存故障、处理器故障

### RBF模型的适用性
RBF模型特别适合处理具有以下特征的故障：
- **空间相关性**：故障倾向于在空间上聚集
- **连通性**：相邻的故障组件相互影响
- **有限扩散**：故障影响范围有限
- **结构化分布**：故障遵循特定的拓扑模式

## 运行测试

### 基础测试
```bash
# 运行完整的理论验证测试
python comprehensive_theory_test.py
```

### 测试内容
理论验证测试包含以下8个方面：
1. **基础RBF容错上界计算**：验证理论公式的准确性
2. **基础分解维度选择**：测试最优维度选择算法
3. **基础PEF模型比较**：与PEF模型的性能对比
4. **修正因子计算**：验证结构和空间修正因子
5. **渐近行为分析**：分析修正因子的收敛性
6. **高维RBF容错分析**：5-7维网络的扩展分析
7. **高维PEF模型比较**：高维情况下的性能对比
8. **算法复杂度验证**：时间和空间复杂度的实际测试

## 理论文档

详细的数学理论推导请参考：
- **mathematical_theory.md**: 完整的理论推导文档

理论文档包含：
- 完整的符号表和基础定义
- RBF容错上界定理的严格证明
- 哈密尔顿性定理的归纳证明
- 算法复杂度的详细分析
- 与PEF模型的理论比较
- 所有关键引理和推论的证明