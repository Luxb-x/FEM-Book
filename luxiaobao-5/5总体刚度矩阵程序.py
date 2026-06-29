import numpy as np
import json
import os



# ===================== 1. 前处理模块 =====================
def load_model(json_path):
    """读取JSON模型文件，处理自由度编号转换（输入1起始 → 内部0起始）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    # 基础参数
    nsd = model['nsd']  # 空间维度 1/2
    ndof = model['ndof']  # 单节点自由度数
    nnp = model['nnp']  # 节点总数
    nel = model['nel']  # 单元总数
    nen = model['nen']  # 单单元节点数

    # 节点坐标
    x = np.array(model['x'], dtype=np.float64)
    y = np.array(model['y'], dtype=np.float64) if nsd >= 2 else None

    # 材料与截面
    E = np.array(model['E'], dtype=np.float64)
    CArea = np.array(model['CArea'], dtype=np.float64)

    # 单元连接数组（输入1起始 → 转为0起始）
    IEN = np.array(model['IEN'], dtype=int) - 1

    # 边界条件（输入1起始 → 转为0起始）
    fixed_dof = np.array(model['fixed_dof'], dtype=int) - 1
    fixed_value = np.array(model['fixed_value'], dtype=np.float64)

    # 节点载荷（输入1起始 → 转为0起始）
    force_dof = np.array(model['force_dof'], dtype=int) - 1
    force_value = np.array(model['force_value'], dtype=np.float64)

    # 总自由度数
    n_total_dof = nnp * ndof

    # 组装全局载荷向量
    F = np.zeros(n_total_dof, dtype=np.float64)
    for i, dof in enumerate(force_dof):
        F[dof] = force_value[i]

    return {
        'title': model['Title'],
        'nsd': nsd, 'ndof': ndof, 'nnp': nnp, 'nel': nel, 'nen': nen,
        'x': x, 'y': y, 'E': E, 'CArea': CArea,
        'IEN': IEN, 'fixed_dof': fixed_dof, 'fixed_value': fixed_value,
        'F': F, 'n_total_dof': n_total_dof
    }


# ===================== 2. 对号矩阵LM生成 =====================
def build_LM(IEN, ndof, nen):
    """根据单元连接数组IEN（0起始）生成对号矩阵LM
    LM[e, :] 存储第e个单元所有局部自由度对应的全局自由度编号
    """
    nel = IEN.shape[0]
    LM = np.zeros((nel, nen * ndof), dtype=int)
    for e in range(nel):
        for i in range(nen):
            node_id = IEN[e, i]
            # 节点node_id对应的全局自由度起始编号
            start = node_id * ndof
            LM[e, i * ndof: (i + 1) * ndof] = np.arange(start, start + ndof)
    return LM


# ===================== 3. 单元刚度矩阵计算 =====================
def compute_element_stiffness(e, model):
    """计算第e个单元的全局坐标系下的刚度矩阵"""
    x = model['x']
    y = model['y']
    E = model['E'][e]
    A = model['CArea'][e]
    IEN = model['IEN']
    ndof = model['ndof']
    nsd = model['nsd']

    node_i = IEN[e, 0]
    node_j = IEN[e, 1]

    # 计算单元长度与方向余弦
    dx = x[node_j] - x[node_i]
    if nsd == 1:
        L = abs(dx)
        c = dx / L
        s = 0
    else:
        dy = y[node_j] - y[node_i]
        L = np.sqrt(dx ** 2 + dy ** 2)
        c = dx / L
        s = dy / L

    # 单元轴向刚度
    k0 = E * A / L

    # 全局坐标系下单元刚度矩阵
    if nsd == 1:
        Ke = k0 * np.array([
            [1, -1],
            [-1, 1]
        ])
    else:
        Ke = k0 * np.array([
            [c ** 2, c * s, -c ** 2, -c * s],
            [c * s, s ** 2, -c * s, -s ** 2],
            [-c ** 2, -c * s, c ** 2, c * s],
            [-c * s, -s ** 2, c * s, s ** 2]
        ])

    return Ke, L, c, s


# ===================== 4. 总体刚度矩阵直接组装 =====================
def assemble_global_stiffness(LM, Ke_list, n_total_dof):
    """根据对号矩阵LM，将单元刚度矩阵累加组装为总体刚度矩阵"""
    K = np.zeros((n_total_dof, n_total_dof), dtype=np.float64)
    nel = LM.shape[0]

    for e in range(nel):
        Ke = Ke_list[e]
        lm = LM[e]
        # 对号入座累加
        for a in range(len(lm)):
            for b in range(len(lm)):
                K[lm[a], lm[b]] += Ke[a, b]
    return K


# ===================== 5. 边界条件处理与求解（缩减法） =====================
def solve_reduction(K, F, fixed_dof, fixed_value):
    """缩减法处理位移边界条件，求解节点位移与约束反力"""
    n_total = len(F)
    all_dof = np.arange(n_total)

    # 自由自由度（未知位移）
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    # 分块矩阵
    K_FF = K[np.ix_(free_dof, free_dof)]
    K_FE = K[np.ix_(free_dof, fixed_dof)]
    K_EF = K[np.ix_(fixed_dof, free_dof)]
    K_EE = K[np.ix_(fixed_dof, fixed_dof)]

    F_F = F[free_dof]
    d_E = fixed_value

    # 求解未知位移
    d_F = np.linalg.solve(K_FF, F_F - K_FE @ d_E)

    # 重构完整位移向量
    d = np.zeros(n_total, dtype=np.float64)
    d[fixed_dof] = d_E
    d[free_dof] = d_F

    # 计算约束反力
    R_E = K_EE @ d_E + K_EF @ d_F - F[fixed_dof]

    return d, R_E, free_dof


# ===================== 6. 后处理：单元应力与轴力 =====================
def compute_element_results(model, LM, d):
    """计算所有单元的长度、方向余弦、应力、轴力"""
    nel = model['nel']
    E = model['E']
    A = model['CArea']
    nsd = model['nsd']

    results = []
    for e in range(nel):
        Ke, L, c, s = compute_element_stiffness(e, model)
        lm = LM[e]
        de = d[lm]  # 提取单元节点位移

        # 应力计算
        if nsd == 1:
            B = np.array([-1, 1]) / L
        else:
            B = np.array([-c, -s, c, s]) / L
        sigma = E[e] * B @ de
        N = sigma * A[e]  # 轴力

        results.append({
            'element_id': e + 1,  # 输出转回1起始
            'L': L,
            'c': c,
            's': s,
            'sigma': sigma,
            'N': N
        })
    return results


# ===================== 7. 结果输出与校验 =====================
def print_results(model, K, d, R_E, elem_results, LM):
    print("=" * 60)
    print(f"算例名称：{model['title']}")
    print("=" * 60)

    # 1. 总体刚度矩阵
    print("\n1. 总体刚度矩阵 K（施加边界条件前）：")
    np.set_printoptions(precision=6, suppress=True)
    print(K)

    # 2. 对称性检查
    is_symmetric = np.allclose(K, K.T, atol=1e-12)
    print(f"\n2. 对称性检查：{'通过' if is_symmetric else '不通过'}")

    # 3. 奇异性检查
    det_K = np.linalg.det(K)
    print(f"3. 奇异性检查（行列式值）：{det_K:.6e}")
    print(f"   矩阵{'奇异' if abs(det_K) < 1e-9 else '非奇异'}")

    # 4. 对号矩阵LM
    print("\n4. 对号矩阵 LM（0起始编号）：")
    print(LM)

    # 5. 节点位移
    print("\n5. 节点位移结果：")
    ndof = model['ndof']
    nnp = model['nnp']
    for i in range(nnp):
        if ndof == 1:
            print(f"   节点{i + 1}：d = {d[i]:.6f}")
        else:
            print(f"   节点{i + 1}：u = {d[i * ndof]:.6f}, v = {d[i * ndof + 1]:.6f}")

    # 6. 约束反力
    print("\n6. 约束反力：")
    fixed_dof = model['fixed_dof']
    for i, dof in enumerate(fixed_dof):
        print(f"   自由度{dof + 1}：R = {R_E[i]:.6f}")

    # 7. 单元结果
    print("\n7. 单元计算结果：")
    print(f"{'单元':<4}{'长度':<10}{'c':<10}{'s':<10}{'应力':<12}{'轴力':<10}")
    for res in elem_results:
        print(
            f"{res['element_id']:<4}{res['L']:<10.6f}{res['c']:<10.6f}{res['s']:<10.6f}{res['sigma']:<12.6f}{res['N']:<10.6f}")
    print("=" * 60)


# ===================== 主程序入口 =====================
def main(json_path):
    # 1. 前处理
    model = load_model(json_path)

    # 2. 生成对号矩阵
    LM = build_LM(model['IEN'], model['ndof'], model['nen'])

    # 3. 单元分析：计算所有单元刚度矩阵
    Ke_list = []
    for e in range(model['nel']):
        Ke, _, _, _ = compute_element_stiffness(e, model)
        Ke_list.append(Ke)

    # 4. 组装总体刚度矩阵
    K = assemble_global_stiffness(LM, Ke_list, model['n_total_dof'])

    # 5. 边界条件处理与求解
    d, R_E, _ = solve_reduction(K, model['F'], model['fixed_dof'], model['fixed_value'])

    # 6. 后处理
    elem_results = compute_element_results(model, LM, d)

    # 7. 输出结果
    print_results(model, K, d, R_E, elem_results, LM)


if __name__ == "__main__":
    # 默认运行二维桁架算例，可替换为一维杆算例路径
    case_file = "case2_2d_truss.json"
    if os.path.exists(case_file):
        main(case_file)
    else:
        print(f"未找到输入文件 {case_file}，请确保JSON文件与脚本在同一目录下")