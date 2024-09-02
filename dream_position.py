"""
@Author: blankxiao
@file: dream_position.py
@Created: 2024-09-02 11:39
@Desc: 三个已知编号的点（包括原点） 计算所求的理想坐标
"""


import numpy as np

from point import PolarCoordinate



def distance(O_i: PolarCoordinate, O_j: PolarCoordinate):
    """
    获取两个点的距离
    @param: O_i O_j: 两个点的坐标
    @return: float
    """
    dx = O_i.x - O_j.x
    dy = O_i.y - O_j.y
    return np.sqrt(dx**2 + dy**2)

def get_real_position(O_i: PolarCoordinate, O_j: PolarCoordinate, aim_point: PolarCoordinate):
    """
    获取aim_point的理论坐标
    @param: O_i O_j: 两个已知的圆心
    @param: aim_point: 待求的圆心坐标
    """
    S_ij = distance(O_i, O_j)
    S_j = O_j.r
    # 传入的是pi为单位的float
    theta_i = O_i.pi_theta
    theta_j = O_j.pi_theta
    theta = theta_i + np.acos((S_j / S_ij) * np.sin(theta_j - theta_i))
    r = (2 * S_j / S_ij) * np.sin(theta_j - theta_i)
    return PolarCoordinate(r=r, theta=theta, index=aim_point.index)


def get_circle_center(circle_center: PolarCoordinate, know_point: PolarCoordinate, aim_point: PolarCoordinate, total_points_num: int):
    """
    获取三个点的圆心坐标
    @param: circle_center为零点 know_point为已知点 aim_point为待求点的编号
    """
    # 获取当前情况的角度

    alpha = abs(aim_point.get_angle(circle_center, know_point))
    # 根据夹角alpha获取半径
    r = (know_point.r / 2) * (1 / np.sin(alpha))
    # 根据圆心角计算圆心坐标
    theta = (2 * np.pi / total_points_num) * (know_point.index - 1)
    if know_point.index > aim_point.index + 5 or aim_point.index - 5 < know_point.index < aim_point.index:
        theta += (np.pi / 2) - alpha
    else:
        theta += -(np.pi / 2) + alpha
    return PolarCoordinate(r=r, theta=theta, index=-1)


def get_intersection_point(circle_center: PolarCoordinate, know_point_i: PolarCoordinate, know_point_j: PolarCoordinate, real_point: PolarCoordinate, total_points_num: int):
    """
    计算两个圆的交点作为坐标
    @param: circle_center konw_point Aim_Point: 四个点 circle_center 为圆心 konw_point为已知点
    @return: PolarCoordinate
    """
    O_i = get_circle_center(circle_center, know_point_i, real_point, total_points_num)
    O_j = get_circle_center(circle_center, know_point_j, real_point, total_points_num)
    dream_position = get_real_position(O_i, O_j, real_point)
    return dream_position



