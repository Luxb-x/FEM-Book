import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=8, suppress=True)


import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常


# ===================== 一、自研稠密LDLT求解模块 =====================
def ldlt_factor(K):
    """
    对称矩阵LDL^T分解
    :param K: n×n对称二维数组/ndarray
    :return: L(单位下三角), D(对角向量)
    检测D[j]<=0则抛出异常
    """
    n = K.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    for j in range(n):
        # 计算Djj
        s = 0.0
        for k in range(j):
            s += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - s
        if D[j] <= 1e-12:
            raise ValueError(f"矩阵非正定或存在零主元，第{j}个主元D={D[j]:.4e}")
        # 计算L[i,j], i>j
        for i in range(j+1, n):
            s2 = 0.0
            for k in range(j):
                s2 += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - s2) / D[j]
    return L, D

def ldlt_solve(L, D, R):
    """
    LDLT求解 L D L^T a = R
    前代 -> 对角求解 -> 回代
    """
    n = len(D)
    # 前代 Ly = R
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = R[i] - s
    # 对角 D z = y
    z = y / D
    # 回代 L^T a = z
    a = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0.0
        for k in range(i+1, n):
            s += L[k, i] * a[k]
        a[i] = z[i] - s
    return a

def residual_norm(K, a, R):
    """计算残差向量、残差2范数、相对残差"""
    r = R - K @ a
    norm_r = np.linalg.norm(r, ord=2)
    norm_R = np.linalg.norm(R, ord=2)
    rel_r = norm_r / (norm_R + 1e-16)
    return r, norm_r, rel_r

def calc_cond(K):
    """计算2范数条件数"""
    eig = np.linalg.eigvalsh(K)
    cond = np.max(np.abs(eig)) / np.min(np.abs(eig))
    return cond

# ===================== 二、统一求解接口 =====================
def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    t0 = time.time()
    if method == "ldlt":
        L, D = ldlt_factor(K_FF)
        d_F = ldlt_solve(L, D, rhs)
    elif method == "pardiso":
        K_csr = csr_matrix(K_FF)
        d_F = spsolve(K_csr, rhs)
    else:
        raise NotImplementedError("仅支持 ldlt / pardiso")
    t1 = time.time()
    r, nr, rr = residual_norm(K_FF, d_F, rhs)
    return d_F, nr, rr, t1-t0

# ===================== 三、2.3桁架复用模块（简化版） =====================
class TrussFEA:
    def __init__(self):
        self.nodes = None
        self.elems = None
        self.E = None
        self.A = None
        self.LM = None
        self.K_full = None
        self.f_full = None
        self.fixed_dof = []
        self.fixed_disp = {}

    def build_1d_2bar(self):
        """一维两杆算例"""
        self.nodes = np.array([[0], [1], [2]])
        self.elems = [[0,1], [1,2]]
        self.E = 100
        self.A = 1
        n_node = 3
        self.LM = np.zeros((2,2), dtype=int)
        self.LM[0] = [0,1]
        self.LM[1] = [1,2]
        n_dof = n_node
        self.K_full = np.zeros((n_dof, n_dof))
        # 单元刚度组装
        for e in range(2):
            ni, nj = self.elems[e]
            le = self.nodes[nj,0] - self.nodes[ni,0]
            ke = self.E*self.A/le * np.array([[1,-1],[-1,1]])
            dof_i, dof_j = self.LM[e]
            self.K_full[dof_i, dof_i] += ke[0,0]
            self.K_full[dof_i, dof_j] += ke[0,1]
            self.K_full[dof_j, dof_i] += ke[1,0]
            self.K_full[dof_j, dof_j] += ke[1,1]
        self.f_full = np.zeros(n_dof)
        self.f_full[2] = 10.0
        self.fixed_dof = [0]
        self.fixed_disp[0] = 0.0

    def split_FF_EF(self):
        """分块K_FF, K_EF, rhs = f_F - K_EF^T d_E"""
        all_dof = set(range(self.K_full.shape[0]))
        free_dof = sorted(list(all_dof - set(self.fixed_dof)))
        fixed_dof = self.fixed_dof
        nF = len(free_dof)
        nE = len(fixed_dof)
        mapF = {d:i for i,d in enumerate(free_dof)}
        mapE = {d:i for i,d in enumerate(fixed_dof)}
        K_FF = np.zeros((nF, nF))
        K_EF = np.zeros((nE, nF))
        f_F = np.zeros(nF)
        d_E = np.zeros(nE)
        for i, df in enumerate(free_dof):
            f_F[i] = self.f_full[df]
            for j, dj in enumerate(free_dof):
                K_FF[i,j] = self.K_full[df, dj]
        for i, de in enumerate(fixed_dof):
            d_E[i] = self.fixed_disp[de]
            for j, df in enumerate(free_dof):
                K_EF[i,j] = self.K_full[de, df]
        rhs = f_F - K_EF.T @ d_E
        return K_FF, rhs, free_dof, fixed_dof, mapF, mapE

    def recover_full_disp(self, d_F, free_dof, fixed_dof, mapF, mapE):
        n = self.K_full.shape[0]
        d_full = np.zeros(n)
        for d in fixed_dof:
            d_full[d] = self.fixed_disp[d]
        for i, d in enumerate(free_dof):
            d_full[d] = d_F[i]
        return d_full

    def calc_reaction_force(self, d_full):
        R = self.K_full @ d_full - self.f_full
        return R

# ===================== 四、病态矩阵误差分析算例 =====================
def ill_condition_test():
    print("========== 病态矩阵误差分析 ==========")
    K_ill = np.array([[1.0, 1.0], [1.0, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R_ill = K_ill @ a_exact
    cond = calc_cond(K_ill)
    print(f"病态矩阵条件数 cond = {cond:.4e}")
    # 双精度
    L, D = ldlt_factor(K_ill)
    a64 = ldlt_solve(L, D, R_ill)
    r64, nr64, rr64 = residual_norm(K_ill, a64, R_ill)
    err64 = np.linalg.norm(a64 - a_exact)/np.linalg.norm(a_exact)
    print(f"【float64】解{a64}, 相对残差{rr64:.4e}, 相对误差{err64:.4e}")
    # 4位有效数字舍入
    K_4dig = np.round(K_ill, 4)
    R_4dig = np.round(R_ill, 4)
    try:
        L4, D4 = ldlt_factor(K_4dig)
        a4 = ldlt_solve(L4, D4, R_4dig)
        r4, nr4, rr4 = residual_norm(K_4dig, a4, R_4dig)
        err4 = np.linalg.norm(a4 - a_exact)/np.linalg.norm(a_exact)
        print(f"【4位舍入】解{a4}, 相对残差{rr4:.4e}, 相对误差{err4:.4e}")
    except Exception as e:
        print("4位舍入矩阵分解异常:", e)
    print("结论：病态矩阵残差极小但解误差巨大\n")

# ===================== 五、Poisson方程Q4有限元稀疏求解 =====================
class PoissonQ4FEA:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.hx = 1.0 / nx
        self.hy = 1.0 / ny
        self.nodes = []
        self.elems = []
        self.build_mesh()
        self.n_node = len(self.nodes)
        self.coo_data = []
        self.coo_row = []
        self.coo_col = []
        self.R = np.zeros(self.n_node)
        self.assemble_sparse()
        self.boundary_treatment()

    def build_mesh(self):
        # 生成Q4网格节点与单元
        node_id = 0
        node_map = {}
        for j in range(self.ny+1):
            y = j * self.hy
            for i in range(self.nx+1):
                x = i * self.hx
                node_map[(i,j)] = node_id
                self.nodes.append([x, y])
                node_id += 1
        # 单元
        for j in range(self.ny):
            for i in range(self.nx):
                n0 = node_map[(i, j)]
                n1 = node_map[(i+1, j)]
                n2 = node_map[(i+1, j+1)]
                n3 = node_map[(i, j+1)]
                self.elems.append([n0,n1,n2,n3])

    def quad_gauss(self):
        # 2×2高斯积分
        xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        eta = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w = np.array([1,1])
        return xi, eta, w

    def shape_deriv(self, xi, eta):
        # Q4形函数对xi,eta导数
        dNdxi = np.array([
            -(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4
        ])
        dNdeta = np.array([
            -(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4
        ])
        return dNdxi, dNdeta

    def assemble_sparse(self):
        t0 = time.time()
        xi_list, eta_list, w_list = self.quad_gauss()
        for elem in self.elems:
            coords = np.array([self.nodes[n] for n in elem])
            Ke = np.zeros((4,4))
            Re = np.zeros(4)
            for i, xi in enumerate(xi_list):
                for j, eta in enumerate(eta_list):
                    w = w_list[i] * w_list[j]
                    dNdxi, dNdeta = self.shape_deriv(xi, eta)
                    # 雅可比矩阵
                    J = np.zeros((2,2))
                    for a in range(4):
                        J[0,0] += dNdxi[a] * coords[a,0]
                        J[0,1] += dNdxi[a] * coords[a,1]
                        J[1,0] += dNdeta[a] * coords[a,0]
                        J[1,1] += dNdeta[a] * coords[a,1]
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    gradN = np.zeros((2,4))
                    for a in range(4):
                        gradN[:,a] = invJ @ np.array([dNdxi[a], dNdeta[a]])
                    # 单元刚度积分
                    for a in range(4):
                        for b in range(4):
                            Ke[a,b] += (gradN[0,a]*gradN[0,b] + gradN[1,a]*gradN[1,b]) * detJ * w
                    # 源项f=2π² sinπx sinπy
                    x = (1+xi)/2 * self.hx + coords[0,0]
                    y = (1+eta)/2 * self.hy + coords[0,1]
                    f = 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
                    N = np.array([
                        (1-xi)*(1-eta)/4,
                        (1+xi)*(1-eta)/4,
                        (1+xi)*(1+eta)/4,
                        (1-xi)*(1+eta)/4
                    ])
                    for a in range(4):
                        Re[a] += N[a] * f * detJ * w
            # COO组装
            for a in range(4):
                ia = elem[a]
                self.R[ia] += Re[a]
                for b in range(4):
                    ib = elem[b]
                    self.coo_data.append(Ke[a,b])
                    self.coo_row.append(ia)
                    self.coo_col.append(ib)
        self.assemble_time = time.time() - t0

    def boundary_treatment(self):
        # 齐次Dirichlet边界u=0
        self.fixed_dof = []
        self.fixed_val = {}
        for idx, (x,y) in enumerate(self.nodes):
            if x<1e-6 or x>1-1e-6 or y<1e-6 or y>1-1e-6:
                self.fixed_dof.append(idx)
                self.fixed_val[idx] = 0.0
        # 生成缩减K_FF, rhs
        all_dof = set(range(self.n_node))
        free_dof = sorted(list(all_dof - set(self.fixed_dof)))
        self.free_dof = free_dof
        nF = len(free_dof)
        mapF = {d:i for i,d in enumerate(free_dof)}
        # 构建COO总体矩阵
        K_coo = coo_matrix((self.coo_data, (self.coo_row, self.coo_col)), shape=(self.n_node, self.n_node))
        K_full = K_coo.tocsr()
        # 分块提取K_FF
        rows_F = np.array(free_dof)
        cols_F = np.array(free_dof)
        K_FF = K_full[rows_F,:][:,cols_F]
        # rhs = R_F - K_EF^T d_E (d_E=0, rhs=R_F)
        rhs = self.R[rows_F]
        self.K_FF = K_FF
        self.rhs = rhs
        self.mapF = mapF

    def calc_error(self, u_num):
        # 计算L2相对误差、最大节点误差
        u_ex = []
        for x,y in self.nodes:
            ue = np.sin(np.pi*x)*np.sin(np.pi*y)
            u_ex.append(ue)
        u_ex = np.array(u_ex)
        u_full = np.zeros(self.n_node)
        for i, d in enumerate(self.free_dof):
            u_full[d] = u_num[i]
        err_node = np.abs(u_full - u_ex)
        max_err = np.max(err_node)
        L2_num = np.linalg.norm(u_full - u_ex)
        L2_den = np.linalg.norm(u_ex)
        L2_rel = L2_num / L2_den
        return max_err, L2_rel, u_full, u_ex

    def plot_contour(self, u_full, title="数值解云图"):
        # 绘图云图
        x = np.array([p[0] for p in self.nodes])
        y = np.array([p[1] for p in self.nodes])
        nx = self.nx + 1
        ny = self.ny + 1
        X = x.reshape(ny, nx)
        Y = y.reshape(ny, nx)
        U = u_full.reshape(ny, nx)
        plt.figure(figsize=(6,5))
        cnt = plt.contourf(X,Y,U,50,cmap="jet")
        plt.colorbar(cnt)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

# ===================== 六、主程序入口：批量执行所有算例 =====================
def main():
    print("==================== 2-4有限元平衡方程组求解作业主程序 ====================\n")
    # 算例1：一维两杆桁架 2.3对接验证
    print("---------- 算例0：一维两杆桁架 ----------")
    truss = TrussFEA()
    truss.build_1d_2bar()
    K_FF, rhs, free_dof, fixed_dof, mapF, mapE = truss.split_FF_EF()
    print("缩减刚度矩阵K_FF:\n", K_FF)
    print("右端载荷rhs:", rhs)
    # LDLT求解
    d_F, nr, rr, t_solve = solve_equilibrium(K_FF, rhs, method="ldlt")
    print("求解自由位移d_F:", d_F)
    print(f"残差范数={nr:.4e}, 相对残差={rr:.4e}, 求解时间={t_solve:.4f}s")
    d_full = truss.recover_full_disp(d_F, free_dof, fixed_dof, mapF, mapE)
    print("完整节点位移d_full:", d_full)
    R_react = truss.calc_reaction_force(d_full)
    print("节点约束反力R:", R_react)
    print()

    # 算例2：非正定矩阵检测
    print("---------- 非正定矩阵检测算例 ----------")
    K_neg = np.array([[1,2],[2,1]])
    try:
        L, D = ldlt_factor(K_neg)
    except ValueError as e:
        print("捕获预期异常：", e)
    print()

    # 算例3：病态矩阵误差分析
    ill_condition_test()

    # 算例4：Poisson Q4有限元稀疏PARDISO求解
    print("---------- Poisson方程Q4有限元算例 nx=50, ny=50 ----------")
    poiss = PoissonQ4FEA(nx=50, ny=50)
    print(f"网格{poiss.nx}×{poiss.ny}, 总节点数={poiss.n_node}, 自由度数={len(poiss.free_dof)}")
    print(f"矩阵装配时间={poiss.assemble_time:.4f}s")
    u_num, nr_p, rr_p, t_p = solve_equilibrium(poiss.K_FF, poiss.rhs, method="pardiso")
    print(f"PARDISO求解时间={t_p:.4f}s, 相对残差={rr_p:.4e}")
    max_e, L2_e, u_full, u_ex = poiss.calc_error(u_num)
    print(f"最大节点误差={max_e:.4e}, 离散L2相对误差={L2_e:.4e}")
    poiss.plot_contour(u_full, "Poisson数值解云图")
    poiss.plot_contour(np.abs(u_full-u_ex), "Poisson误差云图")

if __name__ == "__main__":
    main()