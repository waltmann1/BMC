from __future__ import division
import numpy as np
from Body import Body


class PduABody(Body):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduABody, self).__init__(index=0)
        points = [[0, 0, 0]]
        types = ['A0']
        ir = 2.35115/2
        ir = ir /2
        for i in range(6):
            points.append([2 * ir * np.sin(2 * np.pi / 6 * i), 2 * ir * np.cos(2 * np.pi / 6 * i), 0])
            types.append('A3')
        for i in range(6):
            points.append([ir * np.sin(2 * np.pi / 6 * i), ir * np.cos(2 * np.pi / 6 * i), 0])
            types.append('X')
        for i in range(6):
            temp1 = points[(i%6) + 1]
            temp2 = points[((i+1)% 6) + 1]
            #print(points)
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
            types.append('AX')
            points.append(a1)
            types.append('A4')
            points.append(a2)
            types.append('A5')

        points.append([0,0, -1/2])
        types.append("A1")
        points.append([0,0, 1/2])
        types.append("A2")

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
