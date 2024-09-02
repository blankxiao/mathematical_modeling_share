"""
@Author: blankxiao
@file: third_question.py
@Created: 2024-09-02 13:47
@Desc: 在第一个问题的推导基础上(已知原点和两个点，求出目标点真实坐标) 模拟第三题的情景
假设作为发射点的三个无人机都是偏差不大的(与圆心组成的圆方差较小) 
对这三个无人机进行两两枚举 套用第一个模型得到三个拟真坐标 再将这三个坐标取一个平均
获得一个误差相对较小的拟真坐标 再通过这个拟真坐标与理论坐标取得一个向量
最后将真实坐标加上这个向量 得到调整后的坐标
"""

import numpy as np
from point import PolarCoordinate, generate_polar_coordinates, draw_polar_coordinates
from dream_position import get_intersection_point

def adjust_point(real_point: PolarCoordinate, origin_point: PolarCoordinate, know_points: list[PolarCoordinate], total_points_num: int, dream_r: int):
    """
    @param real_point: 实际点
    @param origin_point: 圆心点
    @param konw_points: 已知点
    @return: PolarCoordinate 调整后的点坐标
    """
    # 拟真点list
    fit_points = []
    for i in range(len(know_points)):
        for j in range(i+1, len(know_points)):
            aim_point = get_intersection_point(circle_center=origin_point, know_point_i=know_points[i], know_point_j=know_points[j], real_point=real_point, total_points_num=total_points_num)
            fit_points.append(aim_point)

    average_r = sum([point.r for point in fit_points]) / len(fit_points)
    average_theta = sum([point.theta for point in fit_points]) / len(fit_points)

    average_fit_point = PolarCoordinate(average_r, average_theta, index=real_point.r)
    dream_angle = (2 * np.pi/total_points_num) * (real_point.r - 1)
    dream_point = PolarCoordinate(dream_r, dream_angle, index=real_point.r)

    vector = [dream_point.x - average_fit_point.x, dream_point.y - average_fit_point.y]
    update_point = PolarCoordinate(0, 0, index=real_point.index)
    update_point.from_cartesian(real_point.x + vector[0], real_point.y + vector[1])

    return update_point


def choice_sent_point(point_list: list, num_to_choice: int):
    """
    @param total_points: 总点数
    @param num_to_choice: 选择num_to_choice个点
    @return: list[int] 被选中的点
    """
    # 题目要求最大为3
    assert num_to_choice < 4
    # 从 remaining_points 中随机选择 num_to_choice 个点
    return np.random.choice(point_list[1:], num_to_choice, replace=False)


def simulate_one():
    total_points_num = 9
    dream_r = 100
    coordinates, correct_indices = generate_polar_coordinates(num_offset_points=9, num_points=total_points_num, radius=dream_r)
    sent_points = choice_sent_point(coordinates, 3)
    for point in sent_points:
        print(point)

    # draw_polar_coordinates(coordinates)
    for i in range(1, len(coordinates)):
        if coordinates[i] in sent_points:
            continue
        print(coordinates[i])
        coordinates[i] = adjust_point(origin_point=coordinates[0], real_point=coordinates[i], know_points=sent_points, total_points_num=total_points_num, dream_r=dream_r)
        print(coordinates[i])
        break




if __name__ == '__main__':
    simulate_one()


