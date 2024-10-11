from __future__ import division
import numpy as np


class Body(object):

    def __init__(self, index=0):

        self.body_sites = []
        self.binding_sites = []
        self.binding_mass = 0
        self.arm_sites = []
        self.leg_sites = []
        self.types = []
        self.center = [0,0,0]
        self.mass = 1
        self.binding_mass = 1
        self.binding_type = "bind"
        self.index = index
        self.orienation = [1,0,0,0]
        self.image = None

    def shift(self, vector):

        for ind, site in enumerate(self.body_sites):
            self.body_sites[ind] = np.add(site, vector)

        for ind, site in enumerate(self.arm_sites):
            self.arm_sites[ind] = np.add(site, vector)

        for ind, site in enumerate(self.leg_sites):
            self.leg_sites[ind] = np.add(site, vector)

        self.center = np.add(self.center, vector)

    def align(self, quat):

        for ind, site in enumerate(self.body_sites):
            self.body_sites[ind] = quat.orient(site)
        for ind, site in enumerate(self.arm_sites):
            self.arm_sites[ind] = quat.orient(site)
        for ind, site in enumerate(self.leg_sites):
            self.leg_sites[ind] = quat.orient(site)

        self.orientation = quat.q

        return [quat.q[3], quat.q[0], quat.q[1], quat.q[2]]

    def enforce_cubic_bc(self, box_length):

        self.image = [[0, 0, 0] for _ in range(len(self.body_sites))]
        half = box_length / 2
        for ind1, position in enumerate(self.body_sites):
            for ind2 in range(0, 3):
                if position[ind2] > box_length + half:
                    raise ValueError("Polymer bead is greater than a box length outside")
                elif position[ind2] > half:
                    self.body_sites[ind1][ind2] -= box_length
                    self.image[ind1][ind2] += 1
                elif position[ind2] < -1 * (box_length + half):
                    raise ValueError("Polymer bead is greater than a box length outside")
                elif position[ind2] < -1 * half:
                    self.body_sites[ind1][ind2] += box_length
                    self.image[ind1][ind2] -= 1







