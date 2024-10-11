from __future__ import division
import numpy as np
from numpy import linalg as la
from Quaternion import QuaternionBetween



class PolyAbs(object):

    def __init__(self, length, index=-1):

        self.position = [[] for i in range(length)]
        self.type = ['P' for i in range(length)]
        self.bonds = []
        self.bond_names = []
        self.angles = []
        self.angle_names = []
        self.charge = [0 for i in range(length)]
        self.length = length
        self.rigid_count = 0
        self.mass = [1 for _ in range(length)]
        self.index = index
        self.mr = None


    def align(self, vec):
        q = QuaternionBetween(self.chain_vector(), vec)
        for x in range(len(self.position)):
            self.position[x] = q.orient(self.position[x])

    def align_to_q(self, q):
        for x in range(len(self.position)):
            self.position[x] = q.orient(self.position[x])

    def shift(self, vector):

        for ind, site in enumerate(self.position):
            self.position[ind] = np.add(site, vector)

    def chain_vector(self):

        return np.subtract(self.position[-1], self.position[0])

    def rigid_center_of_mass(self):

        mass = self.mass

        mass_array = np.array(mass)
        position_array = np.array(self.position)
        return np.sum(self.pos_x_mass(position_array, mass_array), axis=0) / np.sum(mass_array)

    def pos_x_mass(self, position_array, mass_array):

        y = np.zeros_like(position_array)
        for ind in range(len(mass_array)):
            y[ind][0] = position_array[ind][0] * mass_array[ind]
            y[ind][1] = position_array[ind][1] * mass_array[ind]
            y[ind][2] = position_array[ind][2] * mass_array[ind]
        return y

    def moment_inertia(self):
        mass = self.mass

        mass_array = np.array(mass)
        position_array = np.array(self.position)

        cen = self.center_of_mass_arrays(position_array, mass_array)
        position_array = np.subtract(position_array, cen)
        return self.sum_over_xyz(self.pos_x_mass(self.pos_squared(position_array), mass_array))

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

    def max_radius(self):

        if self.mr is not None:
            return self.mr
        cen = self.rigid_center_of_mass()
        max = 0
        for pos in self.position:
            dist = la.norm(np.subtract(pos, cen))
            if dist > max:
                max = dist
        self.mr = max
        return max

    def enforce_cubic_bc(self, box_length):

        self.image = [[0, 0, 0] for _ in range(len(self.position))]
        half = box_length / 2
        for ind1, position in enumerate(self.position):
            if not self.is_ion(ind1):
                for ind2 in range(0, 3):
                    if position[ind2] > box_length + half:
                        raise ValueError("Polymer bead is greater than a box length outside")
                    elif position[ind2] > half:
                        self.position[ind1][ind2] -= box_length
                        self.image[ind1][ind2] += 1
                    elif position[ind2] < -1 * (box_length + half):
                        raise ValueError("Polymer bead is greater than a box length outside")
                    elif position[ind2] < -1 * half:
                        self.position[ind1][ind2] += box_length
                        self.image[ind1][ind2] -= 1