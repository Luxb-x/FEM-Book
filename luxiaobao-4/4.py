import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if L < 1e-12:
        raise ValueError("两个节点重合，无效单元")
    cx, cy, cz = dx/L, dy/L, dz/L
    Ke = np.zeros((6,6))
    c = np.array([cx, cy, cz])
    k = E*A/L
    Ke[:3,:3] = k * np.outer(c, c)
    Ke[:3,3:] = -k * np.outer(c, c)
    Ke[3:,:3] = -k * np.outer(c, c)
    Ke[3:,3:] = k * np.outer(c, c)
    return L, (cx, cy, cz), Ke

def truss3d_element_stress(x1, x2, E, A, de):
    L, (cx, cy, cz), _ = truss3d_element_stiffness(x1, x2, E, A)
    B = np.array([-cx, -cy, -cz, cx, cy, cz])/L
    epsilon = B @ de
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N

if __name__ == "__main__":
    print("===== 算例1 =====")
    x1 = [0,0,0]
    x2 = [2,0,0]
    E = 200e9
    A = 1.0e-4
    de = np.array([0,0,0, 1.0e-3,0,0])
    L, dircos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
    print("长度 L =", L)
    print("方向余弦 cx,cy,cz =", dircos)
    print("刚度矩阵 Ke:\n", Ke)
    print("应变 ε =", eps)
    print("应力 σ (MPa) =", sig/1e6)
    print("轴力 N (N) =", N)

    print("\n===== 算例2 =====")
    x1 = [0,0,0]
    x2 = [1,2,2]
    E = 210e9
    A = 2.0e-4
    de = np.array([0,0,0, 1.0e-3,2.0e-3,2.0e-3])
    L, dircos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
    print("长度 L =", L)
    print("方向余弦 cx,cy,cz =", dircos)
    print("刚度矩阵 Ke:\n", Ke)
    print("应变 ε =", eps)
    print("应力 σ (MPa) =", sig/1e6)
    print("轴力 N (N) =", N)

    print("\nKe 是否对称：", np.allclose(Ke, Ke.T))
    de_rigid = np.array([1,2,3,1,2,3])
    eps_r, sig_r, N_r = truss3d_element_stress(x1, x2, E, A, de_rigid)
    print("刚体位移下应变、应力、轴力：", eps_r, sig_r, N_r)