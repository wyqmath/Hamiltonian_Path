# 定义 rt(p, q) 的子立方体类
class RT:
    def __init__(self, p, q, k):
        self.p = p
        self.q = q
        self.k = k

    def subcube(self, new_p, new_q):
        return RT(new_p, new_q, self.k)

# 模数运算辅助函数
def modulo_k(x, k):
    return x % k

# 生成 N⁺ 路径
def generate_N_plus(u_start, u_end, k):
    path = []
    current = u_start
    while current != u_end:
        next_node = (current[0], (current[1] + 1) % k)
        path.append((current, next_node))
        current = next_node
    return path

# 生成 N⁻ 路径
def generate_N_minus(u_start, u_end, k):
    path = []
    current = u_start
    while current != u_end:
        next_node = (current[0], (current[1] - 1 + k) % k)
        path.append((current, next_node))
        current = next_node
    return path

# 生成 C⁺ 环路径
def generate_C_plus(m, u_start, u_end, k, other_row_func):
    path = []
    current = u_start
    for _ in range(m):
        next_node = (current[0], (current[1] + 1) % k)
        path.append((current, next_node))
        current = next_node
    # 转换到另一行
    current = (other_row_func(current[0]), current[1])
    path.append(( (other_row_func(current[0]), (current[1] - 1 + k) % k), current ))
    current = (other_row_func(current[0]), (current[1] - 1 + k) % k)
    while current != u_end:
        next_node = (current[0], (current[1] - 1 + k) % k)
        path.append((current, next_node))
        current = next_node
    return path

# 生成 C⁻ 环路径
def generate_C_minus(m, u_start, u_end, k, other_row_func):
    path = []
    current = u_start
    for _ in range(m):
        next_node = (current[0], (current[1] - 1 + k) % k)
        path.append((current, next_node))
        current = next_node
    # 转换到另一行
    current = (other_row_func(current[0]), current[1])
    path.append(( (other_row_func(current[0]), (current[1] + 1) % k), current ))
    current = (other_row_func(current[0]), (current[1] + 1) % k)
    while current != u_end:
        next_node = (current[0], (current[1] + 1) % k)
        path.append((current, next_node))
        current = next_node
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
def HP_rtFree(rt_pq, s, t):
    """
    Procedure: HP_rtFree(rt(p,q), s, t)
    在故障边不存在的情况下，在 rt(p,q) 中寻找从 s 到 t 的 Hamiltonian 路径（H-路径）。

    输入：
        rt_pq: rt(p, q)，一个子立方体，包含属性 p, q 和 k
        s: 起始节点，格式为 (a, b)
        t: 结束节点，格式为 (c, d)

    输出：
        从 s 到 t 的 H 路径 P
    """
    P = []  # 初始化路径为空
    p = rt_pq.p
    q = rt_pq.q
    k = rt_pq.k
    a, b = s
    c, d = t

    def modulo_k(x):
        return x % k

    def other_row(i):
        return p + q - i

    if q - p == 1:
        # 当只有两行时
        if a == c:
            if modulo_k(d - b) % 2 == 1:
                # 情况1：a = c 且 (d - b) mod k 为奇数
                u_ad_minus_1 = (a, modulo_k(d - 1))
                u_a_bar_d_minus_1 = (other_row(a), modulo_k(d - 1))
                u_a_bar_d = (other_row(a), d)
                path_N_plus = generate_N_plus(s, u_ad_minus_1, k)
                path_C_plus = generate_C_plus(modulo_k(b - 1 + k) % k, u_a_bar_d, t, k, other_row)
                P.extend(path_N_plus)
                P.append((u_ad_minus_1, u_a_bar_d_minus_1))
                P.append((u_a_bar_d_minus_1, u_a_bar_d))
                P.extend(path_C_plus)
            else:
                # 情况2：a = c 且 (d - b) mod k 为偶数
                u_a_bar_b = (other_row(a), b)
                u_a_bar_b_minus_1 = (other_row(a), modulo_k(b - 1))
                u_a_bar_d = (other_row(a), d)
                path_C_plus = generate_C_plus(modulo_k(d - 1 + k) % k, s, u_a_bar_b, k, other_row)
                path_N_minus = generate_N_minus(u_a_bar_b_minus_1, u_a_bar_d, k)
                P.extend(path_C_plus)
                P.append((u_a_bar_b, u_a_bar_b_minus_1))
                P.extend(path_N_minus)
                P.append((u_a_bar_d, t))
        else:
            if modulo_k(d - b) % 2 == 1:
                # 情况3：a ≠ c 且 (d - b) mod k 为奇数
                u_ad = (a, d)
                path_N_minus = generate_N_minus(s, u_ad, k)
                path_C_minus = generate_C_minus(modulo_k(b + 1 + k) % k, u_ad, t, k, other_row)
                P.extend(path_N_minus)
                P.extend(path_C_minus)
            else:
                # 情况4：a ≠ c 且 (d - b) mod k 为偶数
                u_a_bar_b = (other_row(a), b)
                u_a_bar_b_plus_1 = (other_row(a), modulo_k(b + 1))
                u_a_d_minus_1 = (a, modulo_k(d - 1))
                u_a_d = (a, d)
                u_a_bar_d_minus_1 = (other_row(a), modulo_k(d - 1))
                path_C_minus = generate_C_minus(modulo_k(d + 1 + k) % k, s, u_a_bar_b, k, other_row)
                path_N_plus = generate_N_plus(u_a_bar_b_plus_1, u_a_bar_d_minus_1, k)
                P.extend(path_C_minus)
                P.append((u_a_bar_b, u_a_bar_b_plus_1))
                P.extend(path_N_plus)
                P.append((u_a_bar_d_minus_1, u_a_d_minus_1))
                P.append((u_a_d_minus_1, u_a_d))
                P.append((u_a_d, t))
    else:
        if a == p and c == q:
            # 情况5：a = p 且 c = q
            s_new = s
            t_new = (c - 1, modulo_k(d - 1))
            # 递归调用
            P = HP_rtFree(rt_pq.subcube(p, q - 1), s_new, t_new)
            # 扩展路径，从 t_new 延伸到 t
            path_extension = []
            current = t_new
            # 添加 (u_{c - 1, d - 1}, u_{c, d - 1})
            next_node = (c, modulo_k(d - 1))
            path_extension.append((current, next_node))
            current = next_node

            # 添加 (u_{c, d - 1}, u_{c, d - 2}), ..., (u_{c, d + 1}, t)
            for _ in range(k - 1):
                next_node = (current[0], modulo_k(current[1] - 1))
                path_extension.append((current, next_node))
                current = next_node
                if current == t:
                    break
            else:
                # 如果循环正常结束，确保最后连接到 t
                path_extension.append((current, t))
            # 将扩展路径添加到 P
            P.extend(path_extension)
        else:
            # 无损一般性，假设 s, t ∈ V(rt(p, q -1))
            # 递归调用 HP_rtFree(rt(p, q -1), s, t)
            P = HP_rtFree(rt_pq.subcube(p, q - 1), s, t)
            # 选择 P 中位于行 q - 1 的一条边 (u_{q -1, h}, u_{q -1, h +1})
            edge = select_edge_on_row(P, q - 1)
            if edge is None:
                print(f"未能在第 {q -1} 行找到边以进行扩展。")
                return P
            h = edge[0][1]
            # 从 P 中移除该边
            P.remove(edge)
            # 构造新的路径片段
            path_extension = []
            # 添加 (u_{q -1, h}, u_{q, h})
            path_extension.append(((q - 1, h), (q, h)))
            # 添加 (u_{q, h}, u_{q, h -1})
            path_extension.append(((q, h), (q, modulo_k(h - 1))))
            # 添加 (u_{q, h -1}, ..., u_{q, h +2})
            i = modulo_k(h - 1)
            while True:
                next_node = (q, modulo_k(i - 1))
                path_extension.append(((q, i), next_node))
                i = modulo_k(i - 1)
                if i == modulo_k(h + 2):
                    break
            # 添加 (u_{q, h +1}, u_{q -1, h +1})
            path_extension.append(((q, modulo_k(h + 1)), (q - 1, modulo_k(h + 1))))
            # 将扩展路径添加到 P
            P.extend(path_extension)
    return P

# 定义嵌入 H 路径的主函数
def embed_H_path(Q, F, s, t):
    """
    输入：
    Q: k 元 2 立方体对象，包含必要的方法和属性
    F: 故障边集合，满足 |F| ≤ 1
    s: 起始节点，格式为 (a, b)
    t: 结束节点，格式为 (c, d)

    输出：
    从 s 到 t 的 H 路径 P
    """
    P = []

    # 检查故障边的数量
    if len(F) == 0:
        # 无故障边，直接调用 HP_rtFree
        rt_pq = RT(0, Q.k - 1, Q.k)
        P = HP_rtFree(rt_pq, s, t)
    elif len(F) == 1:
        # 有一条故障边，需按照算法处理
        faulty_edge = list(F)[0]  # 获取故障边
        # 根据故障边的位置和 s, t 的位置，选择相应的处理方法
        # 这里需要根据具体的算法进一步实现
        print("故障边处理尚未实现。")
    else:
        print("故障过多，无法嵌入 H 路径。")

    return P

# 定义立方体类
class QkCube:
    def __init__(self, k):
        self.k = k
        # 根据需要初始化立方体结构

    # 其他必要的方法

# 示例调用
if __name__ == "__main__":
    k = 5  # 假设 k = 5，且 k 为奇数
    Q = QkCube(k)
    F = set()  # 假设无故障边
    s = (0, 1)  # 起始节点
    t = (4, 3)  # 结束节点，假设维度 n = 5，因此行号范围为 0 到 4
    P = embed_H_path(Q, F, s, t)
    print("Hamiltonian Path P:")
    for edge in P:
        print(edge)
