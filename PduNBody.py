from __future__ import division
import numpy as np
from Body import Body


class PduNBody(Body):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduNBody, self).__init__(index=0)
        points = [[0, 0, 0]]
        types = ['N0']
        ir = 1
        ir = ir /2
        for i in range(5):
            points.append([2 * ir * np.sin(2 * np.pi / 5 * i), 2 * ir * np.cos(2 * np.pi / 5 * i), 0])
            types.append('N3')
        for i in range(5):
            points.append([ir * np.sin(2 * np.pi / 5 * i), ir * np.cos(2 * np.pi / 5 * i), 0])
            types.append('X')
        for i in range(5):
            temp1 = points[(i%5) + 1]
            temp2 = points[((i+1)% 5) + 1]
            print(points)
            new_point = self.x_y_linear_interp(temp1, temp2)

            a1 = self.x_y_linear_interp(new_point, temp1)
            dir = np.subtract(new_point, temp1)
            new_dir = np.divide([dir[1], -dir[0], 0], np.linalg.norm([dir[1], -dir[0], 0]))
            if np.dot(new_dir, a1) < 0:
                new_dir = -new_dir
            #a1 = np.add(a1, np.multiply(np.power(2, (1/6)) / 2, new_dir))

            a2 = self.x_y_linear_interp(new_point, temp2)
            dir = np.subtract(new_point, temp2)
            new_dir = np.divide([dir[1], -dir[0], 0], np.linalg.norm([dir[1], -dir[0], 0]))
            if np.dot(new_dir, a2) < 0:
                new_dir = -new_dir
            #a2 = np.add(a2, np.multiply(np.power(2, (1 / 6)) / 2, new_dir))

            points.append(new_point)
            types.append('NX')
            points.append(a1)
            types.append('N4')
            points.append(a2)
            types.append('N5')

        points.append([0, 0, -1/2])
        types.append("N1")
        points.append([0, 0, 1/2])
        types.append("N2")

        if with_blocker:
            points.append([0, 0, -3 / 2])
            types.append("blockX")
            points.append([0, 0, -5 / 2])
            types.append("blockX")

        for i in range(len(points)):
            self.body_sites.append(list(points[i]))
            self.types.append(types[i])


    def x_y_linear_interp(self, temp1, temp2):

        x = temp1[0] + .5 * (temp2[0] - temp1[0])
        y = temp1[1] + .5 * (temp2[1] - temp1[1])
        return [x,y,0]
