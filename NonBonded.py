from __future__ import division
import numpy as np
import hoomd
from hoomd import md
from numpy import linalg as la
from Loggable import Loggable
from MakeBend import MakeBend


class LJRepulsive(Loggable):
    def __init__(self, log_list=None):

        super(LJRepulsive, self).__init__(log_list)
        self.log_values = ['pair_table_energy']

        self.epsilon = []
        self.sigma = []
        self.names = []

        self.lj_repulsive_pair = None

    def set_lj_repulsive(self, neighbor_list, system, hagan=False, eps_ss=1):

        self.lj_repulsive_pair = hoomd.md.pair.table(width=1000, nlist=neighbor_list)
        self.add_to_logger()
        list = []
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                if t1 == "X" or t2 == "X":
                    if t1 == "C" or t2 == "C" or is_number(2, t1) or is_number(2, t2):
                        list.append([t1, t2, 3/4, 1])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=3/4,
                                                              func=LJRepulsive_pair, coeff=dict(sigma=3/4, epsilon=1))
                    elif is_number(3, t1) or is_number(3, t2):
                        list.append([t1, t2, .6, 1])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.6,
                                                              func=LJRepulsive_pair, coeff=dict(sigma=.6, epsilon=1))
                    elif t1 == "blockX" or t2 == "blockX":
                        list.append([t1, t2, 1, 1])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=1,
                                                              func=LJRepulsive_pair, coeff=dict(sigma=1, epsilon=3))
                    else:
                        list.append([t1, t2, 0, 0])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.1, func=LJRepulsive_pair,
                                                          coeff=dict(sigma=10e-5, epsilon=0))
                elif t1 == "C" or t2 == "C":
                    if is_number("X", t2) or is_number("X", t1) or is_number(3, t1) or is_number(3, t2) \
                            or is_number(4, t1) or is_number(4, t2) or is_number(5, t1) or is_number(5, t2) \
                            or is_number(2, t1) or is_number(2, t2) or is_number(0, t1) or is_number(0, t2):
                        list.append([t1, t2, .5, 1])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.5,
                                                              func=LJRepulsive_pair, coeff=dict(sigma=.5, epsilon=1))
                    else:
                        list.append([t1, t2, 0, 0])
                        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.1,
                                                              func=LJRepulsive_pair,
                                                              coeff=dict(sigma=10e-5, epsilon=0))
                elif (is_number("X", t1) and  (is_number(4, t2) or is_number(5, t2) or is_number("X", t2))) or \
                        (is_number("X", t2) and  (is_number(4, t1) or is_number(5, t1) or is_number("X", t1))):
                    list.append([t1, t2, .4/2, 1])
                    self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.4/2, func=LJRepulsive_pair,
                                                          coeff=dict(sigma=.4/2, epsilon=1))
                elif (is_number("X", t1) and is_number(3, t2) or \
                        (is_number("X", t2) and  is_number(3, t1))):
                    list.append([t1, t2, .6/2, 1])
                    self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.6/2, func=LJRepulsive_pair,
                                                          coeff=dict(sigma=.6/2, epsilon=1))
                #elif (is_number(0, t1) and  (not is_number(0, t2) and not is_number(1, t2))) or \
                #        (is_number(0, t2) and (not is_number(0, t1) and not is_number(1, t1))):
                #    list.append([t1, t2, 1, 1])
                #    self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=2, func=LJRepulsive_pair,
                 #                                         coeff=dict(sigma=1, epsilon=1))
                #elif (is_number(3, t1) and (is_number(1, t2) or is_number(2, t2))) or \
                #        (is_number(3, t2) and (is_number(1, t1) or is_number(2, t1))):
                #    list.append([t1, t2, 1, 1])
                #    self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=2, func=LJRepulsive_pair,
                 #                                         coeff=dict(sigma=1, epsilon=1))
                elif is_number(2, t1) and is_number(2, t2):
                    list.append([t1, t2, -2, -2])
                    self.ABN_22(t1, t2, hagan=hagan, eps_ss=eps_ss)
                elif (is_number(2, t1) and is_number(1,t2)) or (is_number(2, t2) and is_number(1, t1)):
                    list.append([t1, t2, -2, -1])
                    self.ABN_21(t1, t2, eps_ss=eps_ss)
                elif is_number(1, t1) and is_number(1, t2) and hagan:
                    list.append([t1, t2, -1, -1])
                    self.ABN_11(t1, t2, eps_ss=eps_ss)
                else:
                    list.append([t1, t2, 0, 0])
                    self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=.1, func=LJRepulsive_pair,
                                                          coeff=dict(sigma=10e-5, epsilon=0))

        #for thing in list:
        #    print(thing)
        #quit()
        return self.lj_repulsive_pair

    def ABN_00(self, t1, t2):
        print("set 00 interaction")
        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=1/2,
                                              func=LJRepulsive_pair, coeff=dict(sigma=1/2, epsilon=1))
    def ABN_22(self,t1, t2, hagan=False, eps_ss=1):

        print("set 22 N geometry")
        # sigma is the distance between centers for 0 bending of units
        #sig_dist = 2.436 * 2
        sig_dist = 2.236 * 2
        #sig_dist = 2.036 * 2
        if is_letter("N", t1):
            sig_dist -= .418
        if is_letter("N", t2):
            sig_dist -= .418

        if hagan:
            sig_dist = 2.436 * 2
            if is_letter("N", t1) or is_letter("N", t2):
                sig_dist -= 2 * 0.167
            if is_letter("N", t2) and is_letter("N", t1):
                sig_dist -= 2 * 0.169

        #print("aaaa", t1, t2, sig_dist)
        key = ["A", "B", "N"]

        if not type(eps_ss) == type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        if eeps == 0:
            eeps = values[0][0]

        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=sig_dist/2, func=LJRepulsive_pair,
                                              coeff=dict(sigma=sig_dist/2, epsilon=eeps))
    def ABN_21(self,t1, t2, eps_ss=1):

        # hagan sigma tb if 1.8 r_b
        dist = 1.8
        #dist = 2.3

        key = ["A", "B", "N"]

        if not type(eps_ss) == type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        if eeps == 0:
            eeps = values[0][0]
        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=dist*2/2, func=LJRepulsive_pair,
                                              coeff=dict(sigma=dist * 2/2, epsilon=eeps))
    def ABN_11(self, t1, t2, eps_ss=1):

        # hagan sigma bb if 1.5 r_b
        #quit()
        key = ["A", "B", "N"]

        if not type(eps_ss) == type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        if eeps == 0:
            eeps = values[0][0]
        self.lj_repulsive_pair.pair_coeff.set(str(t1), str(t2), rmin=10e-5, rmax=3/2, func=LJRepulsive_pair,
                                              coeff=dict(sigma=1.5 * 2/2, epsilon=eeps))


class LJ(Loggable):
    def __init__(self, log_list=None):

        super(LJ, self).__init__(log_list)

        self.log_values = ['pair_lj_energy']

        self.lj_pair = None

    def set_lj(self, neighbor_list, system, eps_cc=1):

        cut = 6/2
        if eps_cc==0:
            cut = 1
        self.lj_pair = hoomd.md.pair.lj(r_cut=cut, nlist=neighbor_list)
        self.add_to_logger()
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                #if is_number(3, t1) and is_number(3, t2):
                    #self.ABN_33(t1, t2, eps_ss)
                #if (is_number(1, t1) and t2 == "C") or (is_number(1, t2) and t1 == "C"):
                 #   self.ABN_1C(t1, t2, eps_sc)
                if t1 == "C" and t2 == "C":
                    #print("set cargo sigma")
                    #print(eps_cc, t1, t2)
                    #cargo sigma is 2
                    self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=eps_cc, sigma=2/2)
                else:
                    self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=0, sigma=10e-5)

    def is_center(self, string):

        if len(string) >= 6 and string[:6] == 'center':
            return True
        return False

   # def ABN_1C(self, t1, t2, eps_cc):

    #    key = ["A1", "B1", "N1"]
    #    thing = t2
    #    if is_number(1, t1):
    #        thing = t1
    #    if not type(eps_cc) == type([1,2]):
    #        values = [eps_cc for _ in range(len(key))]
    #    else:
    #        values = eps_cc

     #   eeps = values[key.index(thing)]
     #   self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=eeps, sigma=2)


class Hinge(Loggable):

    def __init__(self, log_list=None, trimer_rules=False):

        super(Hinge, self).__init__(log_list)

        self.name1 = "cargo"
        self.name2 = "hinge"

        self.log_values = ['pair_morse_energy_' + self.name1, 'pair_morse_energy_' + self.name2]
        self.hinge = None

        self.the_list = []
        self.trimer_rules = trimer_rules

    def set_hinge(self, neighbor_list, system, eps_ss=1, eps_sc=1, allostery=False):

        self.hinge = hoomd.md.pair.morse(nlist=neighbor_list, r_cut=4/2, name=self.name2)
        self.log_values = ['pair_morse_energy_' + self.name2]
        self.add_to_logger()
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                if (is_number(4, t1) and is_number(5, t2))  or (is_number(5, t1) and is_number(4, t2)):
                    self.ABN_45(t1, t2, eps_ss=eps_ss)
                elif (is_number(6, t1) and is_number(7, t2))  or (is_number(6, t1) and is_number(7, t2)):
                    self.ABN_67(t1, t2, eps_ss=eps_ss)
                elif is_number(3, t1) and is_number(3, t2):
                    self.ABN_33(t1, t2, eps_ss)
                else:
                    self.hinge.pair_coeff.set(str(t1), str(t2), D0=0, r0=.4/2, alpha=20)

    def set_cargo(self, neighbor_list, system, eps_ss=1, eps_sc=1, allostery=False):

        self.cargo = hoomd.md.pair.morse(nlist=neighbor_list, r_cut=4/2, name=self.name1)
        self.log_values = ['pair_morse_energy_' + self.name1]
        self.add_to_logger()
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                if (is_number(1, t1) and t2 == "C") or (is_number(1, t2) and t1 == "C"):
                    self.ABN_1C(t1, t2, eps_sc, allostery=allostery)
                else:
                    self.cargo.pair_coeff.set(str(t1), str(t2), D0=0, r0=.4/2, alpha=20)


    def ABN_45(self, t1, t2, eps_ss=1):


        key = ["A", "B", "N"]

        if not type(eps_ss)==type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        self.the_list.append(["entered 45", t1, t2])
        if not self.trimer_rules:
            self.the_list.append(["45", t1, t2, eeps])
            self.hinge.pair_coeff.set(str(t1), str(t2), D0=eeps, r0=.4 / 2, alpha=20, r_cut=4 / 2)
        elif t1[0] != t2[0]:
            self.the_list.append(["45", t1, t2, eeps])
            self.hinge.pair_coeff.set(str(t1), str(t2), D0=eeps, r0=.4/2, alpha=20, r_cut=4/2)
        else:
            self.hinge.pair_coeff.set(str(t1), str(t2), D0=0, r0=.4/2, alpha=20)

    def ABN_67(self, t1, t2, eps_ss=1):

        key = ["A", "B", "N"]

        if not type(eps_ss)==type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        self.the_list.append(["67", t1, t2, eeps])
        self.hinge.pair_coeff.set(str(t1), str(t2), D0=eeps, r0=.4/2, alpha=20, r_cut=4/2)

    def ABN_33(self, t1, t2, eps_ss):

        key = ["A3", "B3", "N3"]

        if not type(eps_ss) == type([1,2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        eeps = values[key.index(t1)][key.index(t2)]
        self.the_list.append(["33", t1, t2, eeps/2])
        self.hinge.pair_coeff.set(str(t1), str(t2), D0=eeps/2, r0=.4/2, alpha=20, r_cut=4/2)

    def ABN_1C(self, t1, t2, eps_sc, allostery=False):

        if t1 == "C":
            temp = t2
            t2 = t1
            t1 = temp

        key = ["A1", "B1", "N1"]

        if not type(eps_sc) == type([1,2]):
            values = np.multiply(np.ones(3), eps_sc)
        else:
            values = eps_sc

        t1_short = t1[:2]
        eeps = values[key.index(t1_short)]
        add = 0
        if len(t1) > 2:
            add = int(t1[3]) * int(allostery)

        eeps += add

        self.the_list.append(["1C"  + " " + t1, t1, t2, eeps])
        cut = 6.0/2
        if eeps == 0:
            cut =1
        self.cargo.pair_coeff.set(str(t1), str(t2), D0=eeps, r0=1/2, alpha=5, r_cut=cut)


class Bend(Loggable):

    def __init__(self, log_list=None):

        super(Bend, self).__init__(log_list)

        #self.log_values = ["bend_energy"]

        self.bend = None

    def set_bend(self, neighbor_list, system, eps_ss=1, func_to_use=0):

        print("fu", func_to_use)
        print("bepsss", eps_ss)

        self.bend = hoomd.md.pair.table(width=1000, nlist=neighbor_list)
        self.add_to_logger()
        self.the_list = []

        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                if (is_number(0, t1) and is_number(1, t2)) or (is_number(0, t2) and is_number(1, t1)):
                    self.ABN_01(t1, t2, eps_ss=eps_ss, func_to_use=func_to_use)
                else:
                    self.bend.pair_coeff.set(str(t1), str(t2), rmin=0, rmax=1, func=Polynomial_pair,
                                                          coeff=dict(params=[0, 0, 0]))


    def ABN_01(self, t1, t2, func_to_use=0, eps_ss=1):

        key = ["A", "B", "N"]

        sizes = [2.036 / 2, 2.036 / 2, 2.036 / 2]

        if not type(eps_ss)==type([1, 2]):
            values = np.multiply(np.ones((3, 3)), eps_ss)
        else:
            values = eps_ss

        if not type(func_to_use)==type([1, 2]):
            funcs = np.multiply(np.ones((3, 3)), func_to_use)
        else:
            funcs = func_to_use

        eeps = values[key.index(t1[0])][key.index(t2[0])]
        funkin = funcs[key.index(t1[0])][key.index(t2[0])]

        print("set 01 interactino")

        if not funkin == 0:
            mb = MakeBend(sizes[key.index(t1[0])], sizes[key.index(t2[0])])
            rmin, rmax, v_given, d_given = mb.get_params(funkin, eps=eeps)
            self.bend.pair_coeff.set(str(t1), str(t2), rmin=rmin, rmax=rmax, func=Evaluate_v, coeff=dict(v_given=v_given, d_given=d_given))
        else:
            self.bend.pair_coeff.set(str(t1), str(t2), rmin=0, rmax=1, func=Evaluate_v,
                                                          coeff=dict(v_given=[0 for _ in range(100)], d_given=[.01 * i for i in range(100)]))


def LJRepulsive_pair(r,rmin, rmax, sigma, epsilon):

    if r < sigma:
        V = epsilon * ((sigma / r) ** 12 - 1)
        F = 12 * epsilon * (sigma/r) ** 13
    else:
        V = 0
        F = 0
    return (V,F)


def is_number(number, tipe):


    if len(tipe) < 2:
        return False
    if tipe[1] == str(number):
        return True
    else:
        return False

def is_letter(letter, tipe):

    if len(tipe) < 2:
        return False
    if tipe[0] == letter:
        return True
    else:
        return False


def d_01_A_A(r, rmin, rmax):


    params = [ 2.44748814e-01, -6.24277655e+00,  7.04510668e+01, -4.61537236e+02,
               1.93380758e+03, -5.37255458e+03,  9.89389534e+03, -1.16392616e+04,
                7.92456589e+03, -2.36816560e+03]

    order = len(params) - 1
    V = [np.dot(params, [r ** (order-i) for i in range(order+1)]) / 2]

    F = [-np.dot(params[:-1], [(order-i+1) * r ** (order-i) for i in range(order+1)][1:]) / 2]

    return V, F


def Polynomial_pair(r, rmin, rmax, params):

    V=0
    F=0
    if r > rmin and r < rmax:
        order = len(params) - 1
        r_v = [r ** (order-i) for i in range(order+1)]
        r_f = [(1+order - i) * r ** (order-i) for i in range(1, order + 1)]
        V = np.dot(params, r_v)
        F = - np.dot(params[:-1], r_f)

    return V, F


def Hinge_pair(r, rmin, rmax, mean, std, depth):

    V = -depth * gaussian(std, mean, r)
    F = -depth * gaussian_prime(std, mean, r)

    #print(r, V, F, gaussian(std, mean, r), mean, std, depth, rmin, rmax)
    #quit()
    return V,F

def gaussian(std, mean, x):

    return np.exp(-(x-mean)**2 / (2 * std**2))

def gaussian_prime(std, mean, x):

    return 2 * (x-mean)/(2 * std**2) * gaussian(std, mean, x)


def linear_interp(first_key, first_val,  second_key, second_val, middle):

    diff = (first_key - middle) / (first_key - second_key)
    a = first_val + ((second_val - first_val) * diff)
    #print(first_key, first_val, second_key, second_val, middle, diff, a)
    return a


def Evaluate_v(r, rmin, rmax, v_given, d_given):


    rmin = d_given[-1]
    rmax = d_given[0]
    #for i in range(len(d_given)):
    #    print(d_given[i], v_given[i])
    #quit()
    V = 0
    F = 0

    if r < rmin or r > rmax:
        return V, F

    if r == rmin:
        V = eval_v(r, v_given, d_given)
        F = - eval_deriv(rmin, v_given, d_given, plus=True)

    else:
        V = eval_v(r, v_given, d_given)
        F = - eval_deriv(r, v_given, d_given)
    #print("r, V F", r, V, F)

    #if r > 2.4:
        #quit()
    #print(r, V, F)

    return V, F

def eval_deriv(r, v_given, d_given, plus=False, step=.001):

    plus_minus = int(not plus) - int(plus)
    return (eval_v(r, v_given, d_given) - eval_v(r - step * plus_minus, v_given, d_given)) / step * plus_minus


def eval_v(r, v_given, d_given):

    ind1, ind2 = find_between(d_given, r)
    #print(ind1, ind2)
    #print((d_given[ind1], v_given[ind1]), (d_given[ind2], v_given[ind2]))
    return linear_interp(d_given[ind1], v_given[ind1], d_given[ind2], v_given[ind2], r)


def find_between(d_array, d):
    for ind in list(range(1, len(d_array))):
        if d >= d_array[ind] and d <= d_array[ind - 1]:
            return  ind - 1, ind

