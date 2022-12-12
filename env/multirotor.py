import airsim
import numpy as np
import math

from airsim import MultirotorClient
from math import *
from airsim import YawMode


class Multirotor:
    def __init__(self, client: MultirotorClient, is_full: bool):
        self.client = client

        client.reset()
        client.enableApiControl(True)  # 获取控制权
        client.armDisarm(True)  # 解锁py
        client.takeoffAsync().join()  # 起飞
        client.moveToZAsync(-1, 1).join()

        kinematic_state = self.client.simGetGroundTruthKinematics()

        # 无人机坐标
        self.ux = float(kinematic_state.position.x_val)
        self.uy = float(kinematic_state.position.y_val)
        self.uz = float(kinematic_state.position.z_val)

        # 无人机速度
        self.vx = float(kinematic_state.linear_velocity.x_val)
        self.vy = float(kinematic_state.linear_velocity.y_val)
        self.vz = float(kinematic_state.linear_velocity.z_val)

        self.bound_x = [-600, 600]
        self.bound_y = [-600, 600]
        self.bound_z = [-250, 0]
        if is_full:
            self.target_x = [-550, 550]
            self.target_y = [-550, 550]
            self.target_z = [-200, -50]
        else:
            self.target_x = [-400, -150]
            self.target_y = [250, 500]
            self.target_z = [-100, -50]
        self.d_safe = 15

        # 目标点坐标
        self.tx, self.ty, self.tz = self.generate_target()
        self.max_distance = pow((self.bound_x[1] - self.bound_x[0]) ** 2 + (self.bound_y[1] - self.bound_y[0]) ** 2 + (self.bound_z[1] - self.bound_z[0]) ** 2, 0.5)

    def land(self):
        self.client.landAsync(timeout_sec=0).join()  # 第五阶段：降落
        self.client.armDisarm(False)  # 上锁
        self.client.enableApiControl(False)  # 释放控制权

    def generate_target(self):
        """
        生成目标点的位置
        seed为随机种子
        """
        tx = np.random.rand() * (self.target_x[1] - self.target_x[0]) + self.target_x[0]
        ty = np.random.rand() * (self.target_y[1] - self.target_y[0]) + self.target_y[0]
        tz = np.random.rand() * (self.target_z[1] - self.target_z[0]) + self.target_z[0]
        return tx, ty, tz

    '''
    获取无人机与目标点的连线与无人机第一视角方向（飞行方向）的夹角
    tx,ty,tz为目标点坐标
    '''

    def get_deflection_angle(self):
        # 连线向量
        ax = self.tx - self.ux
        ay = self.ty - self.uy
        az = self.tz - self.uz

        # 速度方向向量
        bx = self.vx
        by = self.vy
        bz = self.vz

        if bx == 0 and by == 0 and bz == 0:  # 若无人机停止飞行，则判定完全偏航，给予一个惩罚
            return 180

        model_a = pow(ax ** 2 + ay ** 2 + az ** 2, 0.5)
        model_b = pow(bx ** 2 + by ** 2 + bz ** 2, 0.5)

        cos_ab = (ax * bx + ay * by + az * bz) / (model_a * model_b)
        radius = acos(cos_ab)  # 计算结果为弧度制，范围（0， PI），越小越好
        angle = np.rad2deg(radius)

        return angle

    def get_distance(self):
        xa = self.tx - self.ux
        ya = self.ty - self.uy
        za = self.tz - self.uz
        return pow(xa ** 2 + ya ** 2 + za ** 2, 0.5)

    '''
    距离传感器返回的距离数据，水平，竖直各半个圆周，每30度一采样，
    共13个数据，顺序为S、Y(1-6)、P(1-6)
    '''

    def get_distance_sensors_data(self):
        yaw_axis = ['Y', 'P']
        pitch_axis = ['1', '2', '3', '4', '5', '6']
        data = []
        prefix = "Distance"
        data.append(self.client.getDistanceSensorData(distance_sensor_name=prefix + 'S').distance)
        for i in yaw_axis:
            for j in pitch_axis:
                dsn = prefix + i + j
                data.append(self.client.getDistanceSensorData(distance_sensor_name=dsn).distance)

        return data

    '''
    返回无人机状态(numpy.array)
    '''
    def get_state(self):
        # 进行归一化
        position = np.array([self.tx - self.ux, self.ty - self.uy, self.tz - self.uz])
        target = np.array([self.get_distance() / self.max_distance])
        velocity = np.array([self.vx, self.vy, self.vz])
        angle = np.array([self.get_deflection_angle() / 180])
        sensor_data = np.array(self.get_distance_sensors_data()) / 20

        state = np.append(position, target)
        state = np.append(state, velocity)
        state = np.append(state, angle)
        state = np.append(state, sensor_data)

        return state

    '''
    计算当前状态下的奖励、是否完成
    加速度的坐标为NED，z轴加速度为负则向上飞行
    三个加速度有统一的范围
    碰撞惩罚要略大于目标点惩罚
    '''
    def step(self, action):
        done = self.if_done()
        arrive_reward = self.arrive_reward()
        yaw_reward = self.yaw_reward()
        num_sensor_reward = 0  # self.num_sensor_reward()
        collision_reward = 0
        step_reward = self.step_reward()
        min_sensor_reward = self.min_sensor_reward()
        cross_border_reward = self.cross_border_reward()

        sensor_data = self.get_distance_sensors_data()

        # 碰撞完成与碰撞奖励一起做
        if self.client.simGetCollisionInfo().has_collided:
            # 发生碰撞
            done = True
            '''
            碰撞惩罚-25
            '''
            collision_reward = -25

        ax = action[0]
        ay = action[1]
        az = action[2]
        my_yaw_mode = YawMode()
        my_yaw_mode.is_rate = False
        my_yaw_mode.yaw_or_rate = 0
        self.client.moveByVelocityAsync(vx=self.vx + ax,
                                        vy=self.vy + ay,
                                        vz=self.vz + az,
                                        duration=0.5,
                                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=my_yaw_mode).join()

        kinematic_state = self.client.simGetGroundTruthKinematics()
        p = kinematic_state.position
        distance_reward = self.distance_reward(p.x_val, p.y_val, p.z_val)
        reward = arrive_reward + yaw_reward + min_sensor_reward + num_sensor_reward + collision_reward + step_reward + distance_reward + cross_border_reward

        self.ux = float(kinematic_state.position.x_val)
        self.uy = float(kinematic_state.position.y_val)
        self.uz = float(kinematic_state.position.z_val)
        self.vx = float(kinematic_state.linear_velocity.x_val)
        self.vy = float(kinematic_state.linear_velocity.y_val)
        self.vz = float(kinematic_state.linear_velocity.z_val)
        next_state = self.get_state()

        return next_state, reward, done

    def if_done(self):
        # 与目标点距离小于25米
        model_a = self.get_distance()
        if model_a <= 25.0:
            return True
        # 触及边界
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return True

        return False

    '''
    越界惩罚-25
    '''

    def cross_border_reward(self):
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return -25
        return 0

    '''
    (-.5,0.5)，另一种与目标点距离惩罚(-0.5,0)，两种方式需要权衡
    '''

    def distance_reward(self, next_ux, next_uy, next_uz):
        model_a = self.get_distance()
        xa = self.tx - next_ux
        ya = self.ty - next_uy
        za = self.tz - next_uz
        model_b = pow(xa ** 2 + ya ** 2 + za ** 2, 0.5)
        x = model_a - model_b

        return 1 / (1 + math.exp(-2 * x)) - 0.5

    '''
    抵达目标点奖励+25
    '''

    def arrive_reward(self):
        x = self.tx - self.ux
        y = self.ty - self.uy
        z = self.tz - self.uz
        model_a = pow(x ** 2 + y ** 2 + z ** 2, 0.5)
        if model_a <= 25.0:
            return 25
        else:
            return 0

    '''
    偏航惩罚(-0.25,0)，偏航30°以内不会惩罚
    '''

    def yaw_reward(self):
        yaw = self.get_deflection_angle()
        if yaw <= 30.:
            return 0
        return -0.25 * (yaw / 180)

    '''
    最短激光雷达长度惩罚(-.7,-.6) or (.3,.4)
    当前(-.7,-.3)
    '''

    def min_sensor_reward(self):
        data = self.get_distance_sensors_data()
        d_min = min(data)
        if d_min < self.d_safe:
            return math.exp((self.d_safe - d_min) / -29) - 1.3
        else:
            return 0

    '''
    小于安全阈值的激光雷达条数惩罚(-0.2,0)
    '''

    def num_sensor_reward(self):
        sensor_data = self.get_distance_sensors_data()
        num = sum(i < self.d_safe for i in sensor_data)
        return -0.2 * (num / len(sensor_data))

    '''
    漫游惩罚-0.1
    '''

    def step_reward(self):
        if not self.if_done():
            return -0.1
        else:
            return 0


if __name__ == '__main__':
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    mr = Multirotor(client, True)
    data = np.array(mr.get_distance_sensors_data()) / mr.d_safe
    print(data)
