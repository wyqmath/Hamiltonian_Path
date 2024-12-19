from tqdm import tqdm

# 定义 rt(p, q) 的子立方体类
class RT:
    def __init__(self, p, q, k, n):
        assert 0 <= p < n, f"p 的值必须在 0 到 {n-1} 之间，当前 p={p}"
        assert 0 <= q < n, f"q 的值必须在 0 到 {n-1} 之间，当前 q={q}"
        self.p = p
        self.q = q
        self.k = k
        self.n = n  # 添加 n 属性

    def subcube(self, new_p, new_q):
        return RT(new_p, new_q, self.k, self.n)  # 传递 n

# 模数运算辅助函数
def modulo_k(x, k):
    return x % k

# 生成 N⁺ 路径
def generate_N_plus(u_start, u_end, k, p, q, reconstruct_node):
    path = []
    current = u_start
    total_steps = (u_end[q] - u_start[q]) % k if u_end[q] >= u_start[q] else (u_end[q] - u_start[q] + k)
    print(f"生成 N⁺ 路径，从 {u_start} 到 {u_end}")
    with tqdm(total=total_steps, desc="生成 N⁺ 路径") as pbar:
        while current[q] != u_end[q]:
            next_j = (current[q] + 1) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
    return path

# 生成 N⁻ 路径
def generate_N_minus(u_start, u_end, k, p, q, reconstruct_node):
    path = []
    current = u_start
    total_steps = (u_start[q] - u_end[q]) % k if u_start[q] >= u_end[q] else (u_start[q] - u_end[q] + k)
    print(f"生成 N⁻ 路径，从 {u_start} 到 {u_end}")
    with tqdm(total=total_steps, desc="生成 N⁻ 路径") as pbar:
        while current[q] != u_end[q]:
            next_j = (current[q] - 1 + k) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
    return path

# 生成 C⁺ 环路径
def generate_C_plus(m, u_start, u_end, k, other_row_func, p, q, reconstruct_node):
    path = []
    current = u_start
    print(f"生成 C⁺ 环路径，从 {u_start} 到 {u_end}")
    with tqdm(total=m + k, desc="生成 C⁺ 环路径") as pbar:
        for _ in range(m):
            next_j = (current[q] + 1) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
        # 转换到另一行
        current_p = other_row_func(current[p])
        next_node = reconstruct_node(current_p, current[q])
        path.append((current, next_node))
        current = next_node
        pbar.update(1)
        # 继续在另一行前进
        while current != u_end:
            next_j = (current[q] + 1) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
    return path

# 生成 C⁻ 环路径
def generate_C_minus(m, u_start, u_end, k, other_row_func, p, q, reconstruct_node):
    path = []
    current = u_start
    print(f"生成 C⁻ 环路径，从 {u_start} 到 {u_end}")
    with tqdm(total=m + k, desc="生成 C⁻ 环路径") as pbar:
        for _ in range(m):
            next_j = (current[q] - 1 + k) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
        # 转换到另一行
        current_p = other_row_func(current[p])
        next_node = reconstruct_node(current_p, current[q])
        path.append((current, next_node))
        current = next_node
        pbar.update(1)
        # 继续在另一行后退
        while current != u_end:
            next_j = (current[q] - 1 + k) % k
            next_node = reconstruct_node(current[p], next_j)
            path.append((current, next_node))
            current = next_node
            pbar.update(1)
    return path

def select_edge_on_row(P, row):
    """
    从路径 P 中选择位于指定行的边
    """
    for edge in P:
        if edge[0][0] == row and edge[1][0] == row:
            return edge
    return None

# 定义 HP_rtFree 函数
def HP_rtFree(rt_pq, s, t, F=None, depth=0):
    """
    在 rt(p,q) 中寻找从 s 到 t 的 Hamiltonian 路径，考虑故障边集 F。

    参数：
        rt_pq: RT 类的实例，表示子立方体 rt(p,q)。
        s: 起始节点，n 维元组。
        t: 结束节点，n 维元组。
        F: 故障边集合，形式为 {(u1, v1), (u2, v2), ...}。
        depth: 当前递归深度。

    返回：
        P: Hamiltonian 路径的边列表。
    """
    # 添加递归深度限制（例如最大深度为1000）
    if depth > 1000:
        raise RecursionError("达到最大递归深度限制。")

    # 添加终止条件：如果起点和终点相同，返回空路径
    if s == t:
        return []

    # 仅在特定深度打印，例如 depth < 2
    if depth < 2:
        print(f"{'    '*depth}递归调用 HP_rtFree(rt({rt_pq.p}, {rt_pq.q}), s={s}, t={t})")

    P = []  # 初始化路径为空
    p = rt_pq.p
    q = rt_pq.q
    k = rt_pq.k
    n = rt_pq.n  # 获取维度

    # 检查 p 和 q 是否相同，避免索引错误
    if p == q:
        raise ValueError(f"RT 类的 p 和 q 不能相同: p={p}, q={q}")

    # 获取节点在 p 和 q 维度上的坐标
    try:
        a = s[p]
        b = s[q]
        c = t[p]
        d = t[q]
    except IndexError as e:
        raise IndexError(f"节点元组的维度不足，无法访问索引 p={p} 或 q={q}。节点 s={s}, 节点 t={t}") from e

    # 固定其他维度的坐标
    other_dims = [i for i in range(n) if i != p and i != q]
    fixed_coords = [s[i] for i in other_dims]

    def reconstruct_node(i_value, j_value):
        # 根据 i 和 j 的值，以及固定的维度，重构完整的节点坐标
        coords = list(s)  # 使用 list 保留其他维度的值
        coords[p] = i_value
        coords[q] = j_value
        return tuple(coords)

    # 检查 t 的 p, q 维度是否超出范围
    if len(t) <= max(p, q):
        raise IndexError(f"结束节点 t 的维度不足，无法访问索引 p={p} 或 q={q}。节点 t={t}")

    # 示例递归调用，确保 p 和 q 不相同
    if (a, c) == (0, 1):
        # 构造新的 RT 实例，确保 p 和 q 不相同
        rt_mid_q = RT(new_p=2, new_q=1, k=k, n=n)  # 已修复：传递 n
        # 需要定义 u，根据上下文，这里假设 u 是某个中间节点
        # 需要从路径中选择一个中间节点，这里假设 u 已定义
        # 请根据您的具体算法逻辑调整 u 的选取
        u = reconstruct_node(1, 1)  # 示例，需根据实际情况调整
        P2 = HP_rtFree(rt_mid_q, u, t, F, depth + 1)
        P.extend(P2)
    elif (a, c) in [(1, 0), (2, 1), (1, 2)]:
        # 处理其他特定情况，确保 p 和 q 不相同
        rt_mid_q = RT(new_p=0, new_q=1, k=k, n=n)  # 已修复：传递 n
        u = reconstruct_node(0, 1)  # 示例，需根据实际情况调整
        P2 = HP_rtFree(rt_mid_q, u, t, F, depth + 1)
        P.extend(P2)
    else:
        # 处理一般情况
        # 根据您的具体算法逻辑实现，例如生成 N⁺ 路径或其他操作
        path_n_plus = generate_N_plus(s, t, k, p, q, reconstruct_node)
        P.extend(path_n_plus)

    return P

# 定义嵌入 H 路径的主函数
def embed_H_path(Q, F, s, t):
    """
    主函数：在给定的 k 元 n 维立方体 Q 中，嵌入从节点 s 到节点 t 的 Hamiltonian 路径，考虑故障边集 F。

    参数：
        Q: QkCube 类的实例，表示 k 元 n 维立方体
        F: 故障边集合，形式为 {(u1, v1), (u2, v2), ...}
        s: 源节点，以元组形式表示，例如 (a_0, a_1, ..., a_{n-1})
        t: 目标节点，以元组形式表示

    返回：
        P: Hamiltonian 路径的边列表，形式为 [(u1, v1), (u2, v2), ...]
    """
    print(f"开始嵌入 H 路径，从 {s} 到 {t}")
    P = []

    n = Q.n
    k = Q.k

    # 检查故障边集是否满足 PEF 模型的条件
    if not Q.is_PEF(F):
        print("故障边集不满足 PEF 模型的条件，无法嵌入 H 路径。")
        return P

    if n == 2:
        print("n = 2，调用基础算法")
        P = HP_rtFree(RT(0, 1, k, n), s, t)
    else:
        # 找到拥有最多故障边的维度 i'
        e_list = [Q.edge_fault_count(F, i) for i in range(n)]
        i_prime = e_list.index(max(e_list))

        # 沿着第 i' 维度将立方体划分为 k 个子图
        subcubes = Q.divide_subcubes(i_prime)

        # 找到第 i' 维度上拥有最多故障边的两层之间的层号 l'
        faulty_layers = [Q.edge_fault_count_between_layers(F, i_prime, l, (l + 1) % k) for l in range(k)]
        l_prime = faulty_layers.index(max(faulty_layers))

        # 确定源节点和目标节点
        l_s = s[i_prime]
        l_t = t[i_prime]
        if (l_s - l_prime) % k <= (l_t - l_prime) % k:
            a = s
            b = t
        else:
            a = t
            b = s

        # 根据节点位置调用相应的处理函数
        if (l_s == (l_prime + 1) % k):
            if l_s == l_t:
                P = HP_Round(Q, F, l_prime, 1, a, b)
            else:
                P = HP_Direct(Q, F, l_prime, 1, a, b)
        elif l_s == l_prime:
            P = HP_Direct(Q, F, (l_prime + 1) % k, -1, a, b)
        else:
            P = HP_Direct(Q, F, l_prime, 1, a, b)
            # 处理特殊情况，可能需要重新选择边进行路径拼接
            # 这里需要实现更多的逻辑，根据算法步骤

    print("H 路径嵌入完成")
    return P

# 定义立方体类
class QkCube:
    def __init__(self, n, k):
        """
        初始化 k 元 n 维立方体

        参数：
            n: 维度
            k: 每个维度的节点数量，k 为奇数
        """
        self.n = n
        self.k = k
        # 可以初始化其他必要的属，例如节点列表、边列表等

    def is_PEF(self, F):
        """
        检查故障边集 F 是否满足 PEF 模型的条件

        参数：
            F: 故障边集合

        返回：
            布尔值，True 表示满足条件，False 表示不满足
        """
        n = self.n
        k = self.k
        total_edges = self.total_edges()
        max_faults_allowed = (k ** n - k ** 2) // (k - 1) - 2 * n + 5

        if len(F) > max_faults_allowed:
            return False

        # 计算每个维度的故障边数
        e = [self.edge_fault_count(F, i) for i in range(n)]

        # 检查条件 (3)
        if e[0] != 0 or e[1] > 1:
            return False

        # 检查条件 (2)
        for i in range(2, n):
            if e[i] > k ** i - 2:
                return False

        return True

    def total_edges(self):
        """
        计算立方体的总边数

        返回：
            总边数
        """
        n = self.n
        k = self.k
        return n * k ** n

    def edge_fault_count(self, F, dimension):
        """
        计算指定维度上的故障边数量

        参数：
            F: 故障边集合
            dimension: 维度索引

        返回：
            故障边数量
        """
        count = 0
        for edge in F:
            u, v = edge
            # 检查边是否在指定维度上
            diff = sum(1 for i in range(self.n) if u[i] != v[i])
            if diff == 1 and u[dimension] != v[dimension]:
                count += 1
        return count

    def divide_subcubes(self, dimension):
        """
        将立方体沿指定维度划分为 k 个子立方体

        参数：
            dimension: 维度索引

        返回：
            子立方体列表，每个子立方体是节点的合
        """
        subcubes = []
        for l in range(self.k):
            nodes = []
            for node in self.generate_all_nodes():
                if node[dimension] == l:
                    nodes.append(node)
            subcubes.append(nodes)
        return subcubes

    def generate_all_nodes(self):
        """
        生成立方体内的所有节点

        返回：
            节点的生成器
        """
        from itertools import product
        ranges = [range(self.k) for _ in range(self.n)]
        return product(*ranges)

    def edge_fault_count_between_layers(self, F, dimension, l1, l2):
        """
        计算在指定维度上，层 l1 和 l2 之间的故障边数量

        参数：
            F: 故障边集合
            dimension: 维度索引
            l1, l2: 层号

        返回：
            故障边数量
        """
        count = 0
        for edge in F:
            u, v = edge
            # 检查边是否连接层 l1 和 l2
            if (u[dimension] == l1 and v[dimension] == l2) or (u[dimension] == l2 and v[dimension] == l1):
                count += 1
        return count

    # 其他必要的方法

def HP_Round(Q, F, l_prime, d, s, t):
    """
    Procedure: HP_Round(Q, F, l', d, s, t)
    在符号化的子立方体中造从 s 到 t 的 Hamiltonian 路径，
    其中 l' 为层号，d 为方向（+1 或 -1）。

    参数：
        Q: QkCube 实例
        F: 故障边集合
        l_prime: 分割维度的层号
        d: 方向，+1 表示正向，-1 表示反向
        s: 起始节点
        t: 结束节点

    返回：
        P: Hamiltonian 路径的边列表
    """
    P = []
    n = Q.n
    k = Q.k
    dimension = n - 1  # 选择最高维度进行划分
    layers = [(l_prime + i * d) % k for i in range(k)]  # 确定遍历的层列表

    # 初始化当前节点
    current_node = s

    for idx, layer in enumerate(layers):
        # 获取当前层的所有节点
        subcube_nodes = [node for node in Q.generate_all_nodes() if node[dimension] == layer]
        # 如果当前层只有一个节点，直接连接
        if len(subcube_nodes) == 1:
            next_node = subcube_nodes[0]
            if current_node != next_node:
                P.append((current_node, next_node))
                current_node = next_node
            continue

        # 确定子立方体的起始和结束节点
        if idx == 0:
            u_start = current_node
        else:
            u_start = (current_node[0:dimension] + (layer,))
        if idx == k - 1:
            u_end = t
        else:
            u_end = subcube_nodes[-1]

        # 构造子立方体的 Hamiltonian 路径
        rt_pq = RT(0, n - 1, k, n)
        P_sub = HP_rtFree(rt_pq, u_start, u_end)
        P.extend(P_sub)
        current_node = u_end

        # 连接到下一层节点
        if idx < k - 1:
            next_layer = layers[idx + 1]
            next_node = (current_node[0:dimension] + (next_layer,))
            P.append((current_node, next_node))
            current_node = next_node

    return P

def HP_Direct(Q, F, l_prime, d, s, t):
    """
    Procedure: HP_Direct(Q, F, l', d, s, t)
    在符号化的子立方体中直接构造从 s 到 t 的 Hamiltonian 路径，
    其中 l' 为层号，d 为方向（+1 或 -1）。

    参数：
        Q: QkCube 实例
        F: 故障边集合
        l_prime: 分割维度的层号
        d: 方向，+1 表示正向，-1 表示反向
        s: 起始节点
        t: 结束节点

    返回：
        P: Hamiltonian 路径的边列表
    """
    P = []
    n = Q.n
    k = Q.k
    dimension = n - 1  # 选择最高维度进行划分

    # 获取 s 和 t 所在的层号
    l_s = s[dimension]
    l_t = t[dimension]

    # 根据方向 d，确定遍历的层列表
    if d == 1:
        if l_s <= l_t:
            layers = list(range(l_s, l_t + 1))
        else:
            layers = list(range(l_s, k)) + list(range(0, l_t + 1))
    else:
        if l_s >= l_t:
            layers = list(range(l_s, l_t - 1, -1))
        else:
            layers = list(range(l_s, -1, -1)) + list(range(k - 1, l_t - 1, -1))

    current_node = s

    for idx, layer in enumerate(layers):
        # 获取当前层的所有节点
        subcube_nodes = [node for node in Q.generate_all_nodes() if node[dimension] == layer]

        # 确定子立方体的起始和结束节点
        if idx == 0:
            u_start = current_node
        else:
            u_start = list(current_node)
            u_start[dimension] = layer
            u_start = tuple(u_start)
            P.append((current_node, u_start))  # 连接上一层的末节点到本层的起始节点
            current_node = u_start

        if idx == len(layers) - 1:
            u_end = t
        else:
            # 选择一个末尾节点
            u_end = subcube_nodes[-1] if d == 1 else subcube_nodes[0]

        # 构造子立方体的 Hamiltonian 路径
        rt_pq = RT(0, n - 1, k, n)
        P_sub = HP_rtFree(rt_pq, u_start, u_end)
        P.extend(P_sub)
        current_node = u_end

    return P

# 示例调用
if __name__ == "__main__":
    print("开始嵌入 Hamiltonian 路径...")
    print("\n示例 7：十维立方体，包含少量故障边")
    n = 10  # 维度
    k = 5   # k 值，奇数
    Q = QkCube(n, k)
    F = {
        tuple([0]*10), tuple([0]*9 + [1]),  # 示例故障边
        tuple([1]*10), tuple([1]*9 + [2]),
    }
    s = tuple([0] * n)  # 起始节点
    t = tuple([4] * n)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")

    '''
    if __name__ == "__main__":
    print("示例 1：二维立方体，无故障边")
    n = 2  # 维度
    k = 3  # k 值，奇数
    Q = QkCube(n, k)
    F = set()  # 无故障边
    s = (0, 0)  # 起始节点
    t = (2, 2)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''

    '''
    if __name__ == "__main__":
    print("示例 2：三维立方体，包含少量故障边")
    n = 3  # 维度
    k = 5  # k 值，奇数
    Q = QkCube(n, k)
    F = {
        ((0, 0, 0), (0, 0, 1)),
        ((1, 2, 3), (1, 2, 4)),
    }  # 故障边集
    s = (0, 0, 0)  # 起始节点
    t = (4, 4, 4)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''

    '''
    if __name__ == "__main__":
    print("示例 3：四维立方体，较大 k 值，满足 PEF 条件")
    n = 4  # 维度
    k = 5  # k 值，奇数
    Q = QkCube(n, k)
    F = {
        ((0, 0, 0, 0), (0, 0, 0, 1)),
        ((1, 1, 1, 1), (1, 1, 1, 2)),
        ((2, 2, 2, 2), (2, 2, 2, 3)),
        ((3, 3, 3, 3), (3, 3, 3, 4)),
    }  # 故障边集
    s = (0, 0, 0, 0)  # 起始节点
    t = (4, 4, 4, 4)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''

    '''
    if __name__ == "__main__":
    print("示例 4：三维立方体，较多故障边但满足 PEF 条件")
    n = 3  # 维度
    k = 7  # k 值，奇数
    Q = QkCube(n, k)
    F = {
        ((0, 0, 0), (0, 0, 1)),
        ((0, 1, 1), (0, 1, 2)),
        ((1, 2, 3), (1, 2, 4)),
        ((2, 3, 4), (2, 3, 5)),
        ((3, 4, 5), (3, 4, 6)),
        ((4, 5, 6), (4, 5, 0)),
    }  # 故障边集
    s = (0, 0, 0)  # 起始节点
    t = (6, 6, 6)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''
    
    '''
    if __name__ == "__main__":
    print("示例 5：三维立方体，无法满足 PEF 条件的故障边集")
    n = 3  # 维度
    k = 3  # k 值，奇数
    Q = QkCube(n, k)
    F = {
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 0, 2)),
        ((0, 1, 2), (0, 1, 0)),
    }  # 故障边集，超过最大允许故障边数
    s = (0, 0, 0)  # 起始节点
    t = (2, 2, 2)  # 结束节点

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''

    '''
    # 示例 6：十维立方体，无故障边
    print("\n示例 6：十维立方体，无故障边")
    n = 10  # 维度
    k = 5   # k 值，奇数
    Q = QkCube(n, k)
    F = set()  # 无故障边
    s = tuple([0] * n)  # 起始节点，如 (0, 0, ..., 0)
    t = tuple([4] * n)  # 结束节点，如 (4, 4, ..., 4)

    P = embed_H_path(Q, F, s, t)
    if P:
        print("Hamiltonian 路径 P 生成完毕：")
        for edge in P:
            print(edge)
    else:
        print("无法生成 Hamiltonian 路径。")
    '''