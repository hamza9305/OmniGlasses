import numpy as np
import torch
import math


class Image_to_3D_Virtual_Space():
    def __init__(self, x, y, cx, cy, width, height, FOV,optical_axis):
        self.x = x
        self.y = y
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.optical_axis = optical_axis
        self.FOV = FOV

    def points_from_centre(self, x, y):
        x_star = x - self.cx
        y_star = y - self.cy
        return x_star, y_star

    def length(self, x, y):
        len_vec = torch.sqrt(torch.square(x) + torch.square(y))
        return len_vec

    def phi(self, x, y):
        rad = torch.atan2(y, x)
        rad = torch.where(rad < 0, rad + 2 * math.pi,rad)
        # rad[rad < 0] = 2 * math.pi + rad[rad < 0]
        return rad

    def focal_length(self, width):
        focal_length = width / self.FOV
        return focal_length

    def theta(self, len, focal_length):
        theta_rad = len / focal_length
        return theta_rad

    def light_ray(self, phi, theta):
        x = torch.cos(torch.ravel(phi)) * torch.sin(torch.ravel(theta))
        y = torch.sin(torch.ravel(phi)) * torch.sin(torch.ravel(theta))
        z = torch.cos(torch.ravel(theta))
        lightray = torch.stack((x,y,z),0)
        return lightray

    def vector_q(self,light_ray, device):

        light_q = torch.stack((torch.zeros(light_ray[0].size(), device = device),light_ray[1], light_ray[2]))
        assert light_q.shape == light_q.shape
        return light_q

    def alpha(self, q_ray):
        q_ray = torch.transpose(q_ray,0,1)
        addition = torch.matmul(q_ray,self.optical_axis)
        length_of_q = torch.linalg.norm(q_ray, dim = 1)
        length_of_optical = 1
        cos_value = torch.ravel(addition)/ length_of_q * length_of_optical
        rad = torch.arccos(cos_value) * torch.sign(q_ray[:, 1])
        rad = torch.reshape(rad,(self.height, self.width))
        return rad

    def beta_left(self, light_ray):
        magnitude_yz = torch.sqrt(torch.square(light_ray[1]) + torch.square(light_ray[2]))
        beta = torch.atan2(magnitude_yz, light_ray[0])
        beta = np.pi - beta
        print(f'The shape of beta = {beta.size()}')
        return beta


    def beta_right(self, beta_left, disp):
        beta_right = beta_left - disp
        return beta_right

    def light_ray_right(self, beta_right, alpha):
        alpha = torch.ravel(alpha)
        x = -torch.cos(beta_right)
        y = torch.sin(alpha) * torch.sin(beta_right)
        z = torch.cos(alpha) * torch.sin(beta_right)
        light_ray_right = torch.stack((x,y,z),0)
        return light_ray_right

    def light_ray_right_without_z(self, light_ray_right):
        light_ray_right_without_z = torch.stack((light_ray_right[0],light_ray_right[1]),0)
        return light_ray_right_without_z


    def right_ray_coords(self, x_coord, y_coord):
        x = self.cx + x_coord
        y = self.cy + y_coord
        return x, y


    def angle_theta_right(self, light_ray):
        light_ray = torch.transpose(light_ray,0,1)
        dot_product = torch.matmul(light_ray,self.optical_axis)
        length_of_light_ray = torch.linalg.norm(light_ray, dim = 1)
        length_of_optical = 1
        cos_value = torch.ravel(dot_product) / (length_of_light_ray * length_of_optical)
        rad = torch.arccos(cos_value)
        return rad


    def length_right_ray(self, angle_theta, focal_length):
        length = angle_theta * focal_length
        return length


    def coordinates(self, coords, length):
        norm = torch.linalg.norm(coords, dim=0)
        x, y = (coords / norm) * length
        x_coord = x + self.cx
        y_coord = y + self.cy
        return x_coord, y_coord