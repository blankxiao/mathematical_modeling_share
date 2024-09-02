"""
@Author: blankxiao
@file: point.py
@Created: 2024-08-31 18:16
@Desc: 极坐标类和可视化方法
"""
import numpy as np
import matplotlib.pyplot as plt

class PolarCoordinate:
    def __init__(self, r: float, theta: float, index: int):
        self.r = r
        self.pi_theta = theta if theta > 0 else np.pi * 2 + theta
        self.theta = self.pi_theta * 180 / np.pi  # 将 theta 转换为以 π 为单位
        self.index = index
        self.x, self.y = self.to_cartesian()
    # 将极坐标转换为笛卡尔坐标
    def to_cartesian(self):
        x = self.r * np.cos(self.pi_theta)
        y = self.r * np.sin(self.pi_theta)
        return x, y
    def __str__(self) -> str:
        return f"极坐标点: (r={self.r}, θ={self.theta})"
    
    def from_cartesian(self, x: int, y: int):
        self.r = np.sqrt(x**2 + y**2)
        self.pi_theta = np.arctan2(y, x)
        self.theta = self.pi_theta * 180 / np.pi  # 将 theta 转换为以 π 为单位


    def get_angle(self, A_point, B_point):
        """
        点AB发送信号 C接受信号 计算角ACB
        :param A_point: 极坐标A点
        :param B_point: 极坐标B点
        :return: 角度值
        """
        # 获取A点和B点的笛卡尔坐标
        Ax, Ay = A_point.x, A_point.y
        Bx, By = B_point.x, B_point.y
        Cx, Cy = self.x, self.y
        # 计算向量CA和CB
        CA = np.array([Cx - Ax, Cy - Ay])
        CB = np.array([Cx - Bx, Cy - By])
        # 计算向量CA和CB的叉积
        cross_product = np.cross(CA, CB)
        # 计算向量CA和CB的点积
        dot_product = np.dot(CA, CB)
        # 计算向量CA和CB的模
        magnitude_CA = np.linalg.norm(CA)
        magnitude_CB = np.linalg.norm(CB)
        # 余弦值
        cos_angle = dot_product / (magnitude_CA * magnitude_CB)
        # 正弦值
        sin_angle = cross_product / (magnitude_CA * magnitude_CB)
        # 弧度值
        angle = np.arctan2(sin_angle, cos_angle)
        return angle


def generate_polar_coordinates(num_offset_points, num_points=9, radius=100, r_offset=10, theta_offset=5):
        """
        生成一组极坐标点，其中一个原点和若干个环绕原点分布的点, 其中有num_offset_points个随机偏移的点。
        num_offset_points: int, 随机偏移的点的数量。
        num_points: int, 生成的环绕原点的点的数量，默认为9。
        radius: float, 环绕原点的点到原点的距离，默认为100。
        r_offset: float, 半径的偏移范围，默认为10。
        theta_offset: float, 角度的偏移范围，默认为5度。
        :return
        coordinates: 包含所有生成的极坐标点的列表。
        correct_indices: 未偏移点的索引列表。
        """
        coordinates = []
        
        # 原点
        origin = PolarCoordinate(r=0, theta=0, index=0)
        coordinates.append(origin)
        
        angles = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]  # 等分角度
        
        # 随机选择 num_offset_points 个点进行偏移
        offset_indices = np.random.choice(num_points, num_offset_points, replace=False)
        correct_indices = [i for i in range(num_points + 1) if i not in offset_indices + 1]
        

        for i, angle in enumerate(angles, start=1):
            
            if i - 1 in offset_indices:
                # 生成随机偏移
                angle_offset = np.random.uniform(-theta_offset, theta_offset) * np.pi / 180  # 角度偏移转换为弧度
                radius_offset = np.random.uniform(-r_offset, r_offset)
                
                # 应用偏移
                actual_angle = angle + angle_offset
                actual_radius = radius + radius_offset
            else:
                # 不偏移
                actual_angle = angle
                actual_radius = radius
            
            # 创建 PolarCoordinate 实例并添加到列表中
            coord = PolarCoordinate(r=actual_radius, theta=actual_angle, index=i)
            coordinates.append(coord)
        
        return coordinates, correct_indices



def draw_polar_coordinates(coordinates: list[PolarCoordinate]):
    # 提取 x 和 y 坐标
    x_coords = [coord.x for coord in coordinates]
    y_coords = [coord.y for coord in coordinates]

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='blue', label='Points')

    # 绘制以原点为圆心、radius 为半径的圆
    circle = plt.Circle((0, 0), 100, color='red', fill=False, label='Radius Circle')
    plt.gca().add_patch(circle)

    # 设置图的标题和标签
    plt.title('Polar Coordinates Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')  # 确保 x 和 y 轴比例一致
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()

    

# 示例使用
if __name__ == "__main__":
    coordinates, correct_indices = generate_polar_coordinates(num_points=9, radius=100, num_offset_points=3)
    for i, coord in enumerate(coordinates):
        print(f"Index: {i}, x: {coord.x:.2f}, y: {coord.y:.2f}")
    fig, axes = plt.subplots()
    axes.scatter([coord.x for coord in coordinates], [coord.y for coord in coordinates])
    plt.show()

    print(f"Correct indices: {correct_indices}")