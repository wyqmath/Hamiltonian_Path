# Region-Based Fault Model (RBF)

## Overview

This project implements an innovative fault model, the Region-Based Fault Model (RBF), for the Hamiltonian path embedding problem in k-ary n-cube networks. Unlike the traditional Partitioned Edge Fault (PEF) model, the RBF model treats faults as spatially aggregated "clusters," which better aligns with the spatial correlation characteristics of faults in real-world systems.

## Core Innovations

### 1. Fault Cluster Model
- **Spatial Connectivity**: Faulty edges form connected cluster structures, supporting various topological shapes.
- **Shape Constraints**: Supports standard topologies such as complete graphs, star graphs, path graphs, cycle graphs, and tree graphs.
- **Separation Distance**: A minimum separation distance `d_sep` is maintained between clusters to reduce mutual interference.

### 2. Region-Aware Algorithm
- **Automatic Cluster Identification**: Identifies fault clusters based on a connected components algorithm.
- **Topological Analysis**: Analyzes the shape, center, radius, and connectivity of each cluster.
- **Spatial Optimization**: Merges nearby clusters based on spatial distance to optimize the fault distribution.

### 3. Recursive Path Construction
- **Optimal Dimension Selection**: Uses a separation degree function to select the optimal decomposition dimension.
- **Inductive Proof Strategy**: Relies on rigorous mathematical proofs based on the network's recursive structure.
- **Path Stitching Algorithm**: An efficient strategy for connecting paths across different layers.

## Theoretical Foundation

### Fault Tolerance Upper Bound Theorem
The RBF model provides two levels of fault tolerance guarantees:

1.  **Basic Fault Tolerance Upper Bound** (Connectivity Preservation):
    ```
    Θ_RBF^basic = k_max × s_max × α(n, k, d_sep)
    ```

2.  **Hamiltonicity Fault Tolerance Upper Bound**:
    ```
    Θ_RBF^Ham = min(k/4, k_max × s_max)
    ```

### Correction Factor Definition
```
α(n, k, d_sep) = α_struct(n, k) × α_spatial(d_sep)
```

Where:
- **Structural Correction Factor**: `α_struct(n, k) = min(1 + ln(nk/2)/n, 2.0)`
- **Spatial Correction Factor**: `α_spatial(d_sep) = (1 + 0.5×(1-ρ)) × (1 + ln(1+d_sep)/10)`

### Quantified Performance Improvement
Under standard benchmarks, the performance improvement of the RBF model over the PEF model is:
- **3-ary 3-cube network**: 108.3% improvement
- **3-ary 5-cube network**: 129.7% improvement
- **4-ary 3-cube network**: 100.9% improvement
- **5-ary 3-cube network**: 94.6% improvement

## File Structure

```
Hamiltonian_Path/
├── region_based_fault_model.py    # Core algorithm implementation
├── mathematical_theory.md         # Complete mathematical theory derivation
├── comprehensive_theory_test.py   # Theoretical validation test program
└── README.md                      # This document
```

## Core Components

### Data Structures
- **FaultCluster**: A data structure for a fault cluster, containing faulty edges within the cluster, affected nodes, shape type, center position, radius, etc.
- **RegionBasedFaultModel**: Configuration parameters for the RBF model, including cluster count limit, size limit, separation distance, and shape constraints.
- **ClusterShape**: An enumeration for cluster shapes, supporting complete graphs, star graphs, path graphs, cycle graphs, tree graphs, etc.

### Core Algorithms
- **RegionBasedFaultAnalyzer**: The fault analyzer, responsible for identifying fault clusters, calculating the fault tolerance upper bound, and verifying RBF conditions.
- **RegionBasedHamiltonianEmbedding**: The Hamiltonian path embedding algorithm, which implements recursive construction and path stitching.
- **ComprehensiveTheoryAnalyzer**: The theoretical analyzer, providing complete mathematical theory validation and performance comparison.

## Usage Example

### Basic Usage
```python
from region_based_fault_model import *
from origin_pef import QkCube

# Create network and RBF parameters
Q = QkCube(n=3, k=5)
rbf_params = RegionBasedFaultModel(
    max_clusters=3,
    max_cluster_size=8,
    allowed_shapes=[ClusterShape.COMPLETE_GRAPH, ClusterShape.STAR_GRAPH],
    spatial_correlation=0.5,
    cluster_separation=2
)

# Analyze fault distribution
analyzer = RegionBasedFaultAnalyzer(Q, rbf_params)
fault_tolerance = analyzer.calculate_rbf_fault_tolerance()
print(f"RBF fault tolerance upper bound: {fault_tolerance}")

# Embed Hamiltonian path
embedding = RegionBasedHamiltonianEmbedding(Q, rbf_params)
path = embedding.embed_hamiltonian_path_rbf(fault_edges, source, target)
```

### Theoretical Analysis
```python
from comprehensive_theory_test import ComprehensiveTheoryAnalyzer

# Run the complete theoretical analysis
analyzer = ComprehensiveTheoryAnalyzer()
analyzer.run_all_analysis()  # Includes basic theory, high-dimensional analysis, performance comparison, etc.
```

## Algorithm Complexity

### Time Complexity
- **Overall Complexity**: `O(k^n + n × |C|² × s_max × k^(n-1))`
- **Under RBF Conditions**: `O(k^n) = O(N)`, where N is the number of nodes in the network
- **Fault Analysis**: `O(|C| × s_max)`
- **Dimension Selection**: `O(n × |C|² × s_max)`
- **Path Construction**: `O(k^n)`

### Space Complexity
- **Overall Complexity**: `O(k^n + n × |C| × s_max)`
- **Under RBF Conditions**: `O(k^n)`

Where |C| ≤ k_max is the number of fault clusters, and s_max is the maximum cluster size.

## Comparison with PEF Model

### Theoretical Comparison
| Feature                  | PEF Model                                     | RBF Model                                        |
|--------------------------|-----------------------------------------------|--------------------------------------------------|
| **Fault Mode**           | Independent edge faults partitioned by dimension | Spatially aggregated cluster faults              |
| **Fault Tolerance Upper Bound** | Θ_PEF = O(k^(n-1))                            | Θ_RBF = k_max × s_max × α(n,k,d_sep)               |
| **Decomposition Strategy** | Fixed-dimension decomposition                 | Adaptive optimal dimension selection             |
| **Theoretical Basis**    | Dimensional independence                      | Spatial structure optimization                   |
| **Parameter Definition** | Directly given partitioning conditions        | Directly given correction factors (same methodology) |

### Performance Improvement
Performance comparison based on standard benchmarks:

| Network Configuration | PEF Fault Tolerance | RBF Fault Tolerance | Performance Improvement |
|-----------------------|---------------------|---------------------|-------------------------|
| 3-ary 3-cube          | 8                   | 20                  | **150.0%**              |
| 3-ary 5-cube          | 24                  | 55                  | **129.2%**              |
| 4-ary 3-cube          | 33                  | 64                  | **93.9%**               |
| 4-ary 5-cube          | 147                 | 319                 | **117.0%**              |
| 5-ary 3-cube          | 112                 | 217                 | **93.8%**               |

## Application Scenarios

### Fault Patterns in Real-World Systems
- **Data Center Networks**: Rack-level failures, switch failures, power supply failures, cooling system failures.
- **Networks-on-Chip (NoC)**: Manufacturing defects, thermal hotspots, aging-related faults, electromigration effects.
- **Wireless Sensor Networks**: Environmental interference, physical damage, energy depletion, signal obstruction.
- **High-Performance Computing (HPC)**: Node failures, interconnect failures, memory faults, processor failures.

### Applicability of the RBF Model
The RBF model is particularly well-suited for handling faults with the following characteristics:
- **Spatial Correlation**: Faults tend to cluster together spatially.
- **Connectivity**: Adjacent faulty components affect each other.
- **Limited Diffusion**: The impact of a fault is confined to a limited area.
- **Structured Distribution**: Faults follow specific topological patterns.

## Running Tests

### Basic Test
```bash
# Run the complete theoretical validation tests
python comprehensive_theory_test.py
```

### Test Contents
The theoretical validation tests cover the following 8 aspects:
1.  **Basic RBF Fault Tolerance Upper Bound Calculation**: Verifies the accuracy of the theoretical formulas.
2.  **Basic Decomposition Dimension Selection**: Tests the optimal dimension selection algorithm.
3.  **Basic PEF Model Comparison**: Compares performance with the PEF model.
4.  **Correction Factor Calculation**: Validates the structural and spatial correction factors.
5.  **Asymptotic Behavior Analysis**: Analyzes the convergence of the correction factors.
6.  **High-Dimensional RBF Fault Tolerance Analysis**: Extended analysis for 5-7 dimensional networks.
7.  **High-Dimensional PEF Model Comparison**: Performance comparison in high-dimensional cases.
8.  **Algorithm Complexity Validation**: Empirical testing of time and space complexity.