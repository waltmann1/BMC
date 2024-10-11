from __future__ import division
import numpy as np
from numpy import linalg as la
import copy as cp



class MakeBend(object):

    def __init__(self, r1, r2, sep=.2):
        self.r1 = r1
        self.r2 = r2
        self.sep = sep
        self.deez = [self.d_01_theta_num(t) for t in range(-10, 91)]

    def d_01_theta_num(self, t):

        #hinge_dist = self.sep/np.sqrt(2)
        pos_00 = [-self.r1 - self.sep, 0]
        pos_01 = [-self.r1 - self.sep, -.5]

        length = self.r2
        pos_10 = [length * np.cos(np.deg2rad(t)), -length * np.sin(np.deg2rad(t))]
        pos_11 = np.add([- np.sin(np.deg2rad(t)) * np.sqrt(2)/2, -np.cos(np.deg2rad(t)) * np.sqrt(2)/2], pos_10)

        dist = np.linalg.norm(np.subtract(pos_00, pos_11))
        dist_also = np.linalg.norm(np.subtract(pos_01, pos_10))
        dist_11 = np.linalg.norm(np.subtract(pos_01, pos_11))

        #print(t, dist, dist_11)
        return dist

    def theta_d(self, d):

        theta = [t for t in range(-10, 91)]
        d_01 = [self.d_01_theta_num(t) for t in theta]
        #quit()
        for ind in list(range(1, len(d_01))):
            if d >=d_01[ind] and d<= d_01[ind-1]:
                return self.linear_interp(d_01[ind-1], theta[ind-1], d_01[ind], theta[ind], d)

        return 0


    def get_v_d(self, number, eps=1):

        v_theta_func = None
        if number == 1:
            v_theta_func = self.sharp_pent_hex_func
        elif number == 2:
            v_theta_func = self.hex_hex_thirty
        elif number == 3:
            v_theta_func = self.hex_hex_thirty_wide
        elif number == 4:
            v_theta_func = self.hex_hex_thirty_2wide
        elif number == 5:
            v_theta_func = self.hex_hex_double_30_0
        elif number == 6:
            v_theta_func = self.hex_hex_double_30_half_0
        elif number == 7:
            v_theta_func = self.flat
        elif number == 8:
            v_theta_func = self.hex_hex_double_30_5
        elif number == 9:
            v_theta_func = self.hex_hex_double_30_half_5
        elif number == 10:
            v_theta_func = self.flat_5
        elif number == 11:
            v_theta_func = self.longwell
        elif number == 12:
            v_theta_func = self.long_halfwell
        elif number == 13:
            v_theta_func = self.sixty
        else:
            raise IndexError("Function  " + str(number) + "does not exist")

        v_d = np.multiply(.5, [v_theta_func(self.theta_d(d), eps=eps) for d in self.deez])
        return v_d

    def get_params(self, number, eps=1):

        rmin = np.min(self.deez)
        rmax = np.max(self.deez)

        return rmin, rmax, self.get_v_d(number, eps=eps), self.deez

    def eval_v(self, r, v_given, d_given):

        ind1, ind2 = self.find_between(d_given, r)
        return self.linear_interp(d_given[ind1], v_given[ind1], d_given[ind2], v_given[ind2], r)

    @staticmethod
    def linear_interp(first_key, first_val,  second_key, second_val, middle):

        diff = (first_key - middle) / (second_key - first_key)
        a = first_val - ((first_val - second_val) * diff)
        return a



    def sharp_pent_hex_func(self, t, eps=10):
        if t > 50 or t < 32:
            return 0
        return - eps * (1 - np.abs((t-41)) / 9)

    def hex_hex_thirty(self, t, eps=1):
        if t > 40 or t < 20:
            return 0
        return - eps * (1 - np.abs((t-30)) / 10)

    def hex_hex_thirty_wide(self, t, eps=1):
        if t > 50 or t < 10:
            return 0
        return - eps * (1 - np.abs((t-30)) / 20)

    def hex_hex_thirty_2wide(self, t, eps=1):
        if t > 60 or t < 0:
            return 0
        return - eps * (1 - np.abs((t-30)) / 30)

    def hex_hex_double_30_0(self, t, eps=1):
        if t < 40 and t > 20:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t < 10 and t > -10:
            return - eps * (1 - np.abs((t)) / 10)
        return 0

    def hex_hex_double_30_half_0(self, t, eps=1):
        if t < 40 and t > 20:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t < 10 and t> -10:
            return - .5 * eps * (1 - np.abs((t)) / 10)
        return 0

    def hex_hex_double_30_half_5(self, t, eps=1):
        if t < 40 and t > 20:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t < 15 and t> -5:
            return - .5 * eps * (1 - np.abs((t-5)) / 10)
        return 0

    def hex_hex_double_30_5(self, t, eps=1):
        if t < 40 and t > 20:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t < 15 and t > -5:
            return - eps * (1 - np.abs((t)) / 10)
        return 0

    def longwell(self,t, eps=1):
        if t < 40 and t > 30:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t > -5 and t<=30:
            return - eps * (1 - np.abs((30 - t)) / 35)
        return 0

    def long_halfwell(self,t, eps=1):
        if t < 40 and t > 30:
            return - eps * (1 - np.abs((t-30)) / 10)
        elif t > 20 and t<=30:
            return - eps * (1 - np.abs((30 - t)) / 20)
        elif t > 5 and t<=20:
            return - eps * .5
        elif t <= 5 and t > -5:
            return - .5 * eps * (1 - np.abs((t - 5)) / 10)
        return 0

    def sixty(self,t, eps=1):
        if t > 70 or t < 50:
            return 0
        return - eps * (1 - np.abs((t-60)) / 10)


    def flat(self,t, eps=1):
        if t < 10 and t> -10:
            return -  eps * (1 - np.abs((t)) / 10)
        return 0

    def flat_5(self,t, eps=1):
        if t < 15 and t> -5:
            return -  eps * (1 - np.abs((t-5)) / 10)
        return 0

    @staticmethod
    def double_well(t, spot1, spot2, height=10):

        if t < spot1 - 10:
            t = spot1 -10
        if t > spot1 + 10:
            t = spot1 + 10
        points =np.array([[spot1 - 10, height],[spot1,0],[(spot1+ + spot2)/2, height],[spot2, 0],[spot2+10, height]])
        order = 4
        params = np.polyfit(points[:,0],points[:,1], order)

        return np.dot(params, [t ** (order-i) for i in range(order+1)])

    @staticmethod
    def trimer_flat(t):

        if t < 10:
            t = 10
        if t > 70:
            t = 70
        points =np.array([[10, 2],[20,0],[40, 0],[60, 0],[70, 2]])
        order = 2
        params = np.polyfit(points[:,0],points[:,1], order)
        return np.dot(params, [t ** (order-i) for i in range(order+1)])


    @staticmethod
    def polynomial_pair(r, rmin, rmax, params):

        V=0
        F=0

        if r > rmin and r < rmax:
            order = len(params) - 1
            r_v = [r ** (order-i) for i in range(order+1)]
            r_f = [(1+order - i) * r ** (order-i) for i in range(1, order + 1)]
            V = np.dot(params, r_v)
            F = - np.dot(params[:-1], r_f)

            return V, F


    @staticmethod
    def find_between(d_array, d):
        for ind in list(range(1, len(d_array))):
            if d > d_array[ind] and d < d_array[ind - 1]:
                return ind, ind - 1