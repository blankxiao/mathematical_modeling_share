"""
@Author: blankxiao
@file: test.py
@Created: 2024-08-31 18:17
@Desc: 测试
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class PolarCoordinate:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta
        self.x, self.y = self.to_cartesian()
    
    def to_cartesian(self):
        x = self.r * sp.cos(self.theta)
        y = self.r * sp.sin(self.theta)
        return x, y

def get_angle(A, B, C):
    # 获取点的笛卡尔坐标
    Ax, Ay = A.x, A.y
    Bx, By = B.x, B.y
    Cx, Cy = C.x, C.y
    
    # 计算向量 CA 和 CB
    CA = (Cx - Ax, Cy - Ay)
    CB = (Cx - Bx, Cy - By)
    
    # 计算向量 CA 和 CB 的模
    magnitude_CA = sp.sqrt(CA[0]**2 + CA[1]**2)
    magnitude_CB = sp.sqrt(CB[0]**2 + CB[1]**2)
    
    # 计算向量 CA 和 CB 的点积
    dot_product = CA[0]*CB[0] + CA[1]*CB[1]
    
    # 计算角度 ACB 的余弦值
    cos_angle = dot_product / (magnitude_CA * magnitude_CB)
    
    # 计算角度 ACB
    simplified_angle = sp.acos(cos_angle)
    
    return simplified_angle


if __name__ == "__main__":
    r, theta = sp.symbols('r theta')

    D = PolarCoordinate(r, theta)
    C = PolarCoordinate(100, np.pi*2/9)
    B = PolarCoordinate(100, 0)
    A = PolarCoordinate(0, 0)

    # 计算角度 ADB 和角度 BDC
    alpha = get_angle(A=A, B=B, C=D)
    beta = get_angle(A=B, B=C, C=D)

    # 构建目标表达式
    expr = (alpha - np.pi/6)**2 + (beta - np.pi/9)**2

    # values = {r: 100, theta: sp.pi*2/3 }
    # for var, deriv in zip([theta, r], [expr]):
    #     result = deriv.subs(values).evalf()
    #     print(f"{expr} at {values} = {result}")

    # 将符号表达式转换为可计算的函数
    f = sp.lambdify((r, theta), expr, "numpy")

    # 生成数据点
    r_vals = np.linspace(95, 105, 10)
    theta_vals = np.linspace(max(A.theta, B.theta, C.theta), np.pi * 2, 100)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    # 计算函数值
    Z = f(R, Theta)

    # 找到值为0的点
    zero_points = np.argwhere(np.isclose(Z, 0, atol=1e-4))
    zero_r = R[zero_points[:, 0], zero_points[:, 1]]
    zero_theta = Theta[zero_points[:, 0], zero_points[:, 1]]
    zero_z = Z[zero_points[:, 0], zero_points[:, 1]]

    # 绘制图像
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面图
    surf = ax.plot_surface(R, Theta, Z, cmap='viridis', alpha=0.8)

    # 绘制值为0的点
    ax.scatter(zero_r, zero_theta, zero_z, color='red', s=50, label='Zero Points')

    # 设置标签和标题
    ax.set_xlabel('r')
    ax.set_ylabel('theta')
    ax.set_zlabel('(alpha - sp.pi/6)**2 + (beta - sp.pi/9)**2')
    ax.set_title('Expression (alpha - sp.pi/6)**2 + (beta - sp.pi/9)**2 with Zero Points Marked')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # 添加图例
    ax.legend()

    plt.show()





def get_err_exp(A, B, C, D):
    """
    传入的都是PolarCoordinate类
    @param A B C: 点A B C D 按照编号大小
    @return: 误差表达式
    """

    r, theta = sp.symbols('r theta')
    E = PolarCoordinate(r, theta)
    # 计算角度 ADB 和角度 BDC
    angles = sorted([E.get_angle(A=A, B=B), E.get_angle(A=A, B=C), E.get_angle(A=A, B=D)])

    # 构建目标表达式
    expr = (angles[0] - np.pi/18)**2 + (angles[1] - np.pi/3)**2 + (angles[2] - np.pi/3)**2

    d_angle_d_alpha = sp.diff(expr, theta)
    print(f"Derivative of err with respect to theta: {d_angle_d_alpha}")
    
    d_angle_d_r = sp.diff(expr, r)
    print(f"Derivative of err with respect to r: {d_angle_d_r}")




