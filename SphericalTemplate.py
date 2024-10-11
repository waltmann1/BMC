from __future__ import division
import numpy as np
from PolyAbs import PolyAbs


class SphericalTemplate(PolyAbs):

    def __init__(self, radius, index=0):

        sa = 4 * np.pi * radius * radius
        area = .25 * np.pi
        length = int(sa/area)
        super(SphericalTemplate, self).__init__(length, index=index)
        self.rigid_count = 1

        points = np.multiply(self.unit_sphere(length), radius)

        for ind, point in enumerate(points):
            self.position[ind] = point
            self.mass[ind] = 100
            self.type[ind] = 'qPm'

    def unit_sphere(self, n):

        points = []
        offset = 2. / n
        increment = np.pi * (3. - np.sqrt(5.));

        for i in range(n):
            y = ((i * offset) - 1) + (offset / 2);
            r = np.sqrt(1 - pow(y, 2))

            phi = i * increment

            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x, y, z])

        return points