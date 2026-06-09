import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def wynn_epsilon_extrapolated(s):
    """
    Wynn-epsilon 外推算法
    """
    s = np.asarray(s).flatten()
    m = len(s)

    # 初始化 E 表
    E = np.full((m + 2, m + 2), np.nan)
    E[:, 0] = 0  # 第1列（索引0）为0
    E[1:m + 1, 1] = s  # 第2列（索引1）为 s

    # 计算外推值
    s_hat = []
    idx_hat = []

    for col in range(2, m + 2):  # col 从2到 m+1
        for row in range(1, m - col + 3):
            delta = E[row + 1, col - 1] - E[row, col - 1]
            if abs(delta) < np.finfo(float).eps:
                E[row, col] = np.nan
            else:
                E[row, col] = E[row + 1, col - 2] + 1.0 / delta

        if col % 2 == 1:  # 奇数列（索引从0开始，col=2对应第3列）
            s_hat.append(E[1, col])
            idx_hat.append(2 * len(s_hat))

    return np.array(s_hat), np.array(idx_hat) - 1  # 转换为0索引


def main():
    # 创建输出目录（当前脚本所在目录）
    out_dir = Path.cwd()

    # 参数设置
    n = 2 ** np.arange(0, 9)  # n = 1, 2, 4, 8, 16, 32, 64, 128, 256
    h = 1.0 / n
    pi_exact = np.pi

    # 多边形近似：π_n = n * sin(π/n)
    pi_approx = n * np.sin(np.pi / n)
    err_approx = np.abs(pi_exact - pi_approx)

    # 计算局部收敛阶
    order_approx = np.log(err_approx[1:] / err_approx[:-1]) / np.log(h[1:] / h[:-1])

    # Wynn-epsilon 外推
    pi_wynn_vals, idx_wynn = wynn_epsilon_extrapolated(pi_approx)

    # 构建完整的外推结果数组（在对应位置插入外推值）
    pi_wynn = np.full_like(pi_approx, np.nan)
    pi_wynn[idx_wynn] = pi_wynn_vals
    err_wynn = np.abs(pi_exact - pi_wynn)

    # 计算外推的局部收敛阶
    order_wynn = np.log(err_wynn[idx_wynn[1:]] / err_wynn[idx_wynn[:-1]]) / \
                 np.log(h[idx_wynn[1:]] / h[idx_wynn[:-1]])

    # 创建结果表格
    df = pd.DataFrame({
        'n': n,
        'h': h,
        'pi_n': pi_approx,
        'error_pi_n': err_approx,
        'pi_wynn': pi_wynn,
        'error_pi_wynn': err_wynn
    })

    # 保存结果
    df.to_csv(out_dir / 'pi_results.csv', index=False)

    # 保存为 .npz 格式
    np.savez(out_dir / 'pi_results.npz',
             n=n, h=h, pi_approx=pi_approx, err_approx=err_approx,
             order_approx=order_approx, pi_wynn=pi_wynn, err_wynn=err_wynn,
             order_wynn=order_wynn)

    # 输出文本结果
    with open(out_dir / 'pi_results.txt', 'w') as fid:
        fid.write('Polygon approximation of pi for a unit circle\n')
        fid.write('Formula: pi_n = n * sin(pi / n)\n\n')
        fid.write(f"{'n':>6} {'h':>12} {'pi_n':>22} {'error_pi_n':>16} "
                  f"{'pi_wynn':>22} {'error_pi_wynn':>16}\n")

        for k in range(len(n)):
            if not np.isnan(pi_wynn[k]):
                fid.write(f"{n[k]:6d} {h[k]:12.6g} {pi_approx[k]:22.15f} "
                          f"{err_approx[k]:16.8e} {pi_wynn[k]:22.15f} {err_wynn[k]:16.8e}\n")
            else:
                fid.write(f"{n[k]:6d} {h[k]:12.6g} {pi_approx[k]:22.15f} "
                          f"{err_approx[k]:16.8e} {'NaN':>22} {'NaN':>16}\n")

        fid.write('\nLocal order for pi_n      : ')
        fid.write(' '.join(f'{x:.2f}' for x in order_approx))
        fid.write('\nLocal order for Wynn-eps  : ')
        fid.write(' '.join(f'{x:.2f}' for x in order_wynn))
        fid.write('\n')

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor='white')

    # 绘制直接近似的误差
    ax.loglog(h, err_approx, '-v',
              color=[0.20, 0.45, 0.75], markerfacecolor='none',
              linewidth=1.4, markersize=7, label='Direct approximation')

    # 绘制 Wynn-epsilon 外推的误差
    ax.loglog(h[idx_wynn], err_wynn[idx_wynn], '-^',
              color=[0.85, 0.40, 0.10], markerfacecolor='none',
              linewidth=1.4, markersize=7, label='Wynn-ε extrapolation')

    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_xlim([1e-3, 1e0])
    ax.set_ylim([1e-15, 1e5])
    ax.set_xlabel('h = 1 / n', fontsize=12)
    ax.set_ylabel('e_n', fontsize=12)
    ax.set_title('Convergence of Polygon Approximation for π', fontsize=13)

    # 添加标签
    ax.text(1.1e-1, 2.5e-4, r'$e_n = |\pi - \pi_n|$', fontsize=12,
            color=[0.10, 0.10, 0.10])

    # 在图上标注收敛阶
    for k in range(len(order_approx)):
        ax.text(h[k + 1] * 0.95, 1.2e4, f'{order_approx[k]:.2f}',
                color=[0.10, 0.10, 0.10], fontsize=11)

    for k in range(len(order_wynn)):
        ax.text(h[idx_wynn[k + 1]] * 1.03, 3.0e-15, f'{order_wynn[k]:.2f}',
                color=[0.10, 0.10, 0.10], fontsize=11)

    # 标注平均斜率
    ax.text(1.6e-2, 2.0e1, f'slope: {np.mean(order_approx[-4:]):.2f}',
            color=[0.20, 0.10, 0.10], fontsize=12)
    ax.text(1.0e-2, 8.0e-13, f'slope: {order_wynn[-1]:.2f}',
            color=[0.20, 0.10, 0.10], fontsize=12)

    ax.legend()

    # 保存图片
    plt.savefig(out_dir / 'pi_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'pi_convergence.pdf', bbox_inches='tight')
    plt.show()

    # 打印结果到控制台
    print(df.to_string())
    print(f'\nLocal order for direct approximation: ', end='')
    print(' '.join(f'{x:.2f}' for x in order_approx))
    print(f'\nLocal order for Wynn-epsilon: ', end='')
    print(' '.join(f'{x:.2f}' for x in order_wynn))
    print(f'\nBest direct pi approximation       : {pi_approx[-1]:.15f} (n = {n[-1]})')
    print(f'Best Wynn-epsilon approximation    : {pi_wynn_vals[-1]:.15f} (n = {n[idx_wynn[-1]]})')


if __name__ == '__main__':
    main()