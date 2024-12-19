def embed_H_path(Q, F, s, t):
    """
    输入：
    Q: k元2立方体对象，包含必要的方法和属性
    F: 故障边集合，满足 |F| ≤ 1
    s: 起始节点，格式为 (a, b)
    t: 结束节点，格式为 (c, d)

    输出：
    从 s 到 t 的 H 路径 P
    """
    P = []  # 初始化路径为空
    k = Q.k  # 立方体的维度
    a, b = s
    c, d = t

    if k == 3:
        # 当 k = 3 时，执行穷举搜索
        P = exhaustive_search(Q, F, s, t)
    else:
        if len(F) == 0:
            # 如果没有故障边，调用 HP_rtFree
            P = HP_rtFree(Q.rt(0, k - 1), s, t)
        else:
            faulty_edge = list(F)[0] if F else None
            if {a, c} == {0, 1}:
                # 当 {a, c} = {0, 1} 时
                # 构造两条不相交的路径避开故障边
                P_prime = find_disjoint_path(s, Q.rt(0, 1), faulty_edge)
                P_double_prime = find_disjoint_path(t, Q.rt(0, 1), faulty_edge)
                # 获取 s' 和 t'
                s_prime = P_prime[-1]
                t_prime = P_double_prime[-1]
                # 获取 s'' 和 t''��它们是 s' 和 t' 在 Q - rt(0,1) 中的邻居
                s_double_prime = Q.get_neighbor(s_prime, exclude_rt=Q.rt(0, 1))
                t_double_prime = Q.get_neighbor(t_prime, exclude_rt=Q.rt(0, 1))
                # 在 rt(2, k - 1) 中调用 HP_rtFree
                P_rt = HP_rtFree(Q.rt(2, k - 1), s_double_prime, t_double_prime)
                # 合并路径
                P = P_prime + [(s_prime, s_double_prime)] + P_rt + [(t_double_prime, t_prime)] + P_double_prime[::-1]
            elif a >= 2 and c >= 2:
                # 当 {a, c} ⊆ {2, 3, ..., k - 1} 时
                # 构造环 C
                C = construct_cycle(Q, [(0, 1), (1, 1)], F)
                # 调用 HP_rtFree
                P = HP_rtFree(Q.rt(2, k - 1), s, t)
                # 将环 C 合并到路径 P
                P = merge_cycle_into_path(P, C, avoid_edge=(2, (0,1)))
            else:
                # 其他情况
                # 构造避开故障边的 H 路径，从 s 到 s'，其中 s' 不与 t 相邻
                P_prime = find_H_path(s, Q.rt(0, 1), faulty_edge, avoid_adjacent=t)
                s_prime = P_prime[-1]
                # 获取 t'，它是 s' 在 Q - rt(0,1) 中的邻居
                t_prime = Q.get_neighbor(s_prime, exclude_rt=Q.rt(0, 1))
                # 在 rt(2, k - 1) 中调用 HP_rtFree
                P_rt = HP_rtFree(Q.rt(2, k - 1), t_prime, t)
                # 合并路径
                P = P_prime + [(s_prime, t_prime)] + P_rt
    return P

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
                # P = N⁺(s, u_{a,d-1}) ∪ {(u_{a,d-1}, u_{ā,d-1}, u_{ā,d})} ∪ C⁺_{b-1}(u_{ā,d}, t)
                u_ad_minus_1 = (a, modulo_k(d - 1))
                u_a_bar_d_minus_1 = (other_row(a), modulo_k(d - 1))
                u_a_bar_d = (other_row(a), d)
                path_N_plus = generate_N_plus(s, u_ad_minus_1, k)
                path_C_plus = generate_C_plus(modulo_k(b - 1), u_a_bar_d, t, k, other_row)
                P = path_N_plus
                P.append((u_ad_minus_1, u_a_bar_d_minus_1))
                P.append((u_a_bar_d_minus_1, u_a_bar_d))
                P.extend(path_C_plus)
            else:
                # 情况2：a = c 且 (d - b) mod k 为偶数
                # P = C⁺_{d-1}(s, u_{ā,b}) ∪ {(u_{ā,b}, u_{ā,b-1}), (u_{ā,d}, t)} ∪ N⁻(u_{ā,b-1}, u_{ā,d})
                u_a_bar_b = (other_row(a), b)
                u_a_bar_b_minus_1 = (other_row(a), modulo_k(b - 1))
                u_a_bar_d = (other_row(a), d)
                path_C_plus = generate_C_plus(modulo_k(d - 1), s, u_a_bar_b, k, other_row)
                path_N_minus = generate_N_minus(u_a_bar_b_minus_1, u_a_bar_d, k)
                P = path_C_plus
                P.append((u_a_bar_b, u_a_bar_b_minus_1))
                P.extend(path_N_minus)
                P.append((u_a_bar_d, t))
        else:
            if modulo_k(d - b) % 2 == 1:
                # 情况3：a ≠ c 且 (d - b) mod k 为奇数
                # P = N⁻(s, u_{a,d}) ∪ C⁻_{b+1}(u_{a,d}, t)
                u_ad = (a, d)
                path_N_minus = generate_N_minus(s, u_ad, k)
                path_C_minus = generate_C_minus(modulo_k(b + 1), u_ad, t, k)
                P = path_N_minus + path_C_minus
            else:
                # 情况4：a ≠ c 且 (d - b) mod k 为偶数
                # P = C⁻_{d+1}(s, u_{ā,b}) ∪ {(u_{ā,b}, u_{ā,b+1}), (u_{a,d-1}, u_{a,d}, t)} ∪ N⁺(u_{ā,b+1}, u_{ā,d-1})
                u_a_bar_b = (other_row(a), b)
                u_a_bar_b_plus_1 = (other_row(a), modulo_k(b + 1))
                u_a_d_minus_1 = (a, modulo_k(d - 1))
                u_a_d = (a, d)
                u_a_bar_d_minus_1 = (other_row(a), modulo_k(d - 1))
                path_C_minus = generate_C_minus(modulo_k(d + 1), s, u_a_bar_b, k, other_row)
                path_N_plus = generate_N_plus(u_a_bar_b_plus_1, u_a_bar_d_minus_1, k)
                P = path_C_minus
                P.append((u_a_bar_b, u_a_bar_b_plus_1))
                P.extend(path_N_plus)
                P.append((u_a_bar_d_minus_1, u_a_d_minus_1))
                P.append((u_a_d_minus_1, u_a_d))
                P.append((u_a_d, t))
    else:
        if a == p and c == q:
            # 情况5：a = p 且 c = q
            # 递归调用 HP_rtFree(rt(p, q - 1), s, u_{c - 1, d - 1})
            s_new = s
            t_new = (c - 1, modulo_k(d - 1))
            P = HP_rtFree(rt_pq.subcube(p, q - 1), s_new, t_new)
            # 扩展路径
            path_extension = []
            current = t_new
            for i in range(k):
                next_node = (c, (current[1] - i) % k)
                path_extension.append((current, next_node))
                current = next_node
                if current == t:
                    break
            P.extend(path_extension)
        else:
            # 无损一般性，假设 s, t ∈ V(rt(p, q -1))
            P = HP_rtFree(rt_pq.subcube(p, q - 1), s, t)
            # 选择 P 中位于行 q -1 的一条边
            edge = select_edge_on_row(P, q - 1)
            if edge is None:
                print("No edge found on row", q - 1)
                return P
            h = edge[0][1]
            # 构造新的路径片段
            path_extension = [
                ((q - 1, h), (q, h)),
                ((q, h), (q, modulo_k(h - 1))),
            ]
            # 添加循环部分
            i = modulo_k(h - 1)
            while i != modulo_k(h + 2):
                path_extension.append(((q, i), (q, modulo_k(i - 1))))
                i = modulo_k(i - 1)
            path_extension.append(((q, modulo_k(h + 1)), (q - 1, modulo_k(h + 1))))
            # 更新路径
            P.remove(edge)
            P.extend(path_extension)
    return P

def exhaustive_search(Q, F, s, t):
    """
    穷举搜索所有可能的路径，寻找从 s 到 t 的 H 路径
    """
    # 需要具体实现，这里提供占位符
    pass

def find_disjoint_path(s, rt, faulty_edge):
    """
    在 rt 中找到从 s 开始的路径，避开故障边，且路径不相交
    """
    # 需要具体实现，这里提供占位符
    pass

def construct_cycle(Q, nodes, F):
    """
    构造一个包含指定节点的环，避开故障边
    """
    # 需要具体实现，这里提供占位符
    pass

def merge_cycle_into_path(P, C, avoid_edge):
    """
    将环 C 合并到路径 P 中，避开指定的边
    """
    # 需要具体实现，这里提供占位符
    pass

def find_H_path(s, rt, faulty_edge, avoid_adjacent):
    """
    在 rt 中找到从 s 开始的 H 路径，避开故障边，且路径终点不与 avoid_adjacent 相邻
    """
    # 需要具体实现，这里提供占位符
    pass

def generate_N_plus(u_start, u_end, k):
    """
    生成从 u_start 到 u_end 的 N⁺路径
    """
    path = []
    current = u_start
    while current != u_end:
        next_node = (current[0], (current[1] + 1) % k)
        path.append((current, next_node))
        current = next_node
    return path

def generate_N_minus(u_start, u_end, k):
    """
    生成从 u_start 到 u_end 的 N⁻路径
    """
    path = []
    current = u_start
    while current != u_end:
        next_node = (current[0], (current[1] - 1) % k)
        path.append((current, next_node))
        current = next_node
    return path

def generate_C_plus(m, u_start, u_end, k, other_row_func):
    """
    生成从 u_start 到 u_end 的 C⁺环，长度为 m
    """
    path = []
    current = u_start
    for _ in range(m):
        next_node = (current[0], (current[1] + 1) % k)
        path.append((current, next_node))
        current = next_node
    # 返回到另一行
    current = (other_row_func(current[0]), current[1])
    path.append(((other_row_func(current[0]), (current[1] - 1) % k), current))
    while current != u_end:
        next_node = (current[0], (current[1] - 1) % k)
        path.append((current, next_node))
        current = next_node
    return path

def generate_C_minus(m, u_start, u_end, k, other_row_func):
    """
    生成从 u_start 到 u_end 的 C⁻环，长度为 m
    """
    path = []
    current = u_start
    for _ in range(m):
        next_node = (current[0], (current[1] - 1) % k)
        path.append((current, next_node))
        current = next_node
    # 返回到另一行
    current = (other_row_func(current[0]), current[1])
    path.append(((other_row_func(current[0]), (current[1] + 1) % k), current))
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

# 定义 rt(p, q) 的子立方体类
class RT:
    def __init__(self, p, q, k):
        self.p = p
        self.q = q
        self.k = k

    def subcube(self, new_p, new_q):
        return RT(new_p, new_q, self.k)

# 示例调用
if __name__ == "__main__":
    k = 5  # 假设 k = 5
    rt_pq = RT(0, k - 1, k)
    s = (0, 1)
    t = (1, 3)
    P = HP_rtFree(rt_pq, s, t)
    print("Hamiltonian Path P:")
    for edge in P:
        print(edge)
