from __future__ import division
import numpy as np
from numpy import linalg as la
import copy as cp
from Quaternion import QuaternionBetween
from Solution import Solution

class Mer(object):

    def __init__(self, body, arm=None, leg=None, index=0):

        self.particle_index = 0
        self.body = body
        self.body.image = [[0,0,0] for _ in range(len(body.body_sites))]
        self.arms = []
        self.legs = []
        self.leg = leg
        self.arm = arm
        self.total_particles = len(self.body.body_sites) + len(self.body.binding_sites)
        self.center = [0,0,0]
        self.mr = None

        for site in body.arm_sites:
            copy = cp.deepcopy(arm)
            copy.align([0,0,-1])
            copy.shift(site)
            self.arms.append(copy)
            self.total_particles += len(copy.position)

        for site in body.leg_sites:
            copy = cp.deepcopy(leg)
            copy.align(site)
            copy.shift(site)
            self.legs.append(copy)
            self.total_particles += len(copy.position)

        self.rigid_count = 1
        if arm is not None:
            self.rigid_count += arm.rigid_count * len(self.body.arm_sites)
        if leg is not None:
            self.rigid_count += leg.rigid_count * len(self.body.leg_sites)
        self.body.binding_mass = 0
        if arm is not None:
            self.body.binding_mass = arm.mass[0]
        self.index = index

    def dump_xyz(self, dump_file=None):
        """

        :param dump_file: name of the file to dump the xyz to
        :return: nothing
        """
        filename = dump_file
        if dump_file is None:
            filename = 'dump.xyz'
        elif dump_file[-4:] != '.xyz':
            filename = dump_file + '.xyz'
        f_out = open(filename, 'w')
        f_out.write(str(self.total_particles + 1))
        f_out.write('\n\n')
        f_out.write('%s %f %f %f\n' % ('Center', self.body.center[0], self.body.center[1], self.body.center[2]))

        for ind, site in enumerate(self.body.body_sites):
            f_out.write('%s %f %f %f\n' % (self.body.types[ind], site[0], site[1], site[2]))

        #for site in self.body.binding_sites:
        #    f_out.write('%s %f %f %f\n' % (self.body.binding_type, site[0], site[1], site[2]))

        #for chain in self.arms:
        #    for i in range(chain.length):
        #        f_out.write('%s %f %f %f\n' % (chain.type[i], chain.position[i][0], chain.position[i][1],
          #                                     chain.position[i][2]))
        #for chain in self.legs:
        #    for i in range(chain.length):
        #        f_out.write('%s %f %f %f\n' % (chain.type[i], chain.position[i][0], chain.position[i][1],
         #                                      chain.position[i][2]))
        f_out.close()

    def rigid_center_of_mass(self, box_length=None):

        if box_length is not None:
            bod_sites = np.add((self.body.body_sites), np.multiply(self.body.image, box_length))
            position_array = np.array(bod_sites)
        else:
            position_array = np.array(self.body.body_sites + self.body.binding_sites)

        mass = [self.body.mass for _ in self.body.body_sites] + \
               [self.body.binding_mass for _ in self.body.binding_sites]

        mass_array = np.array(mass)
        return np.sum(self.pos_x_mass(position_array, mass_array), axis=0) / np.sum(mass_array)

    def pos_x_mass(self, position_array, mass_array):

        y = np.zeros_like(position_array)
        for ind in range(len(mass_array)):
            y[ind][0] = position_array[ind][0] * mass_array[ind]
            y[ind][1] = position_array[ind][1] * mass_array[ind]
            y[ind][2] = position_array[ind][2] * mass_array[ind]
        return y

    def moment_inertia(self):
        mass = [self.body.mass for _ in self.body.body_sites] + \
               [self.body.binding_mass for _ in self.body.binding_sites]

        mass_array = np.array(mass)
        position_array = np.array(self.body.body_sites + self.body.binding_sites)

        cen = self.center_of_mass_arrays(position_array, mass_array)
        position_array = np.subtract(position_array, cen)

        return self.calculate_inertia_tensor(position_array, mass_array)
        #return self.sum_over_xyz(self.pos_x_mass(self.pos_squared(position_array), mass_array))

    def pos_squared(self, position_array):

        y = np.zeros_like(position_array)

        for ind in range(len(position_array)):
            y[ind][0] = position_array[ind][0] * position_array[ind][0]
            y[ind][1] = position_array[ind][1] * position_array[ind][1]
            y[ind][2] = position_array[ind][2] * position_array[ind][2]
        return y

    def sum_over_xyz(self, array):

        final = np.array([0, 0, 0])

        for list in array:
            final[0] += list[0]
            final[1] += list[1]
            final[2] += list[2]
        return final

    def center_of_mass_arrays(self, position_array, mass_array):
        return np.sum(self.pos_x_mass(position_array, mass_array), axis=0) / np.sum(mass_array)

    def total_rigid_mass(self):
        return len(self.body.body_sites * self.mass) + len(self.binding_sites) * self.body.binding_mass

    def get_solution(self):
        return Solution([self], [])

    def shift(self, vector):

        self.body.shift(vector)
        for arm in self.arms:
            arm.shift(vector)
        for leg in self.legs:
            leg.shift(vector)

    def max_radius(self):

        if self.mr is not None:
            return self.mr
        cen = self.body.center
        max = 0
        for pos in self.body.body_sites:
            dist = la.norm(np.subtract(pos, cen))
            if dist > max:
                max = dist
        for arm in self.arms:
            for pos in arm.position:
                dist= la.norm(np.subtract(pos, cen))
                if dist> max:
                    max = dist
        for leg in self.legs:
            for pos in leg.position:
                dist= la.norm(np.subtract(pos, cen))
                if dist> max:
                    max = dist
        self.mr = max
        return max

    def align(self, vec):

        q = QuaternionBetween([0,0,1], vec)
        hoomd = self.body.align(q)
        del self.arms
        self.arms = []
        for ind, site in enumerate(self.body.arm_sites):
            copy = cp.deepcopy(self.arm)
            copy.align(site)
            copy.shift(site)
            self.arms.append(copy)
            #self.total_particles += len(copy.position)
        del self.legs
        self.legs = []
        for site in self.body.leg_sites:
            copy = cp.deepcopy(self.leg)
            copy.align(site)
            copy.shift(site)
            self.legs.append(copy)
            #self.total_particles += len(copy.position)
        return hoomd


    def calculate_inertia_tensor(self, position_array, mass_array):

        tensor = np.zeros((3, 3))
        for idx in range(len(mass_array)):

            tensor[0][0] += (np.square(position_array[idx][1]) + np.square(position_array[idx][2])) * mass_array[
        idx]
            tensor[1][1] += (np.square(position_array[idx][2]) + np.square(position_array[idx][0])) * mass_array[
        idx]

            tensor[2][2] += (np.square(position_array[idx][0]) + np.square(position_array[idx][1])) * mass_array[
        idx]
            tensor[0][1] -= position_array[idx][0] * position_array[idx][1] * mass_array[idx]
            tensor[0][2] -= position_array[idx][0] * position_array[idx][2] * mass_array[idx]
            tensor[1][2] -= position_array[idx][1] * position_array[idx][2] * mass_array[idx]
            tensor[1][0] = tensor[0][1]
            tensor[2][0] = tensor[0][2]
            tensor[2][1] = tensor[1][2]

            values, vectors = la.eig(tensor)
        return values

    def enforce_cubic_bc(self, box_length):

        self.body.enforce_cubic_bc(box_length)
        for arm in self.arms:
            arm.enforce_cubic_bc(box_length)
        for leg in self.legs:
            leg.enforce_cubic_bc(box_length)
