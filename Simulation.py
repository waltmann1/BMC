from __future__ import division
#from Bonds import Bonds
from NonBonded import LJ
from NonBonded import LJRepulsive
from NonBonded import Bend
from NonBonded import Hinge
from RigidBodies import Rigid
#from Angles import Angles
import numpy as np
from numpy import linalg as la
import hoomd


class Simulation(object):

    def __init__(self, system, temperature=1, name="protein_sim", reinit=None, o_list=None, eps_sc=1, eps_cc=1, eps_ss=5,
                 hagan=False, bend_ss=0, bend_funcs=0, allostery=False, map_name=None):

        self.system = system
        self.nlist = hoomd.md.nlist.cell(check_period=1)
        self.nlist.reset_exclusions(exclusions=['bond', 'angle', 'dihedral', 'constraint', 'body'])
        #self.log_list = ['potential_energy', 'temperature', 'kinetic_energy']
        self.log_list = ['potential_energy', 'ndof', 'kinetic_energy']
        self.log_list.append('temperature')
        self.log_period = 1000
        self.dump_period = 5e5
        self.temperature = temperature
        self.name = name
        self.allostery = allostery
        #if o_list is not None:
        #    for i in range(len(o_list)):
        #        self.system.particles[i].orientation = o_list[i]

        self.dt = .004

        if allostery:
            self.mers = []
            self.mer_names = []
            self.mer_name_indices = []
            self.cargo = []
            self.read_map(map_name)

        #self.check()


        self.rigid = Rigid()
        self.rigid.set_rigid_bodies(self.system, reinit=reinit, o_list=o_list)

        #self.bonds = Bonds(self.log_list)
        #self.bonds.set_all_harmonic_bonds(system)

        #self.angles = Angles(self.log_list)
        #self.angles.set_all_harmonic_angles(system)

        self.ljs = LJ(self.log_list)
        self.ljs.set_lj(self.nlist, self.system, eps_cc=eps_cc)

        self.ljr = LJRepulsive(self.log_list)
        self.ljr.set_lj_repulsive(self.nlist, self.system, hagan=hagan, eps_ss=eps_ss)


        self.hinge = Hinge(self.log_list)
        self.hinge.set_hinge(self.nlist, self.system, eps_ss=eps_ss, eps_sc=eps_sc, allostery=allostery)
        self.hinge.set_cargo(self.nlist, self.system, eps_ss=eps_ss, eps_sc=eps_sc, allostery=allostery)
        #print(self.hinge.the_list)
        #quit()


        if bend_funcs and not hagan:
            #quit()
            self.bnd = Bend(self.log_list)
            self.bnd.set_bend(self.nlist, self.system, eps_ss=bend_ss, func_to_use=bend_funcs)
         #   print(bend)
         #   quit()


        self.all = hoomd.group.all()



        self.to_integrate = hoomd.group.union(name='dof', a=hoomd.group.rigid_center(), b=hoomd.group.nonrigid())

        hoomd.md.integrate.mode_standard(dt=self.dt)
        self.nve = hoomd.md.integrate.nve(group=self.to_integrate, limit=.001)
        self.nve.disable()
        self.langevin = hoomd.md.integrate.langevin(group=self.to_integrate, kT=self.temperature, seed=np.random.randint(0, high=1e6))

        log_name = self.name + ".log"
        #self.add_s1_energy()
        self.logger = hoomd.analyze.log(filename=log_name, quantities=self.log_list, period=self.log_period,
                          overwrite=True)

        dump_name = self.name + ".gsd"
        self.dumper = hoomd.dump.gsd(filename=dump_name, period=self.dump_period, group=self.all, overwrite=True)

    def run(self, time):

        #print(self.system.constraints)
        hoomd.run(time)

    def set_cc(self, eps_cc):

        self.ljs.set_lj(self.nlist, self.system, eps_cc=eps_cc)

    def set_sc(self, eps_sc):

        self.hinge.set_cargo(self.nlist, self.system, eps_ss=0, eps_sc=eps_sc, allostery=False)

    def add_s1_energy(self):


        c_tags = [part.tag for part in self.system.particles if part.type == "C" ]
        self.c_group = hoomd.group.tag_list("C", c_tags)
        hoomd.compute.thermo(self.c_group)

        s1c_tags = [part.tag for part in self.system.particles if part.type == "C" or self.is_number(1, part.type)]
        self.s1c_group = hoomd.group.tag_list("S1C", s1c_tags)
        hoomd.compute.thermo(self.s1c_group)

        self.log_list.extend(["potential_energy_S1C"])
        self.log_list.extend(["pair_morse_energy_S1C"])
        self.log_list.extend(["potential_energy_C"])
        self.log_list.extend(["num_particles_C"])
        self.log_list.extend(["num_particles_S1C"])


    def is_number(self, number, tipe):

        if len(tipe) < 2:
            return False
        if tipe[1] == str(number):
            return True
        else:
            return False

    def check(self, name_to_update="B"):

        print("check2")
        if not self.allostery:
            quit()

        print("check3")
        print(self.system.particles.types)


        connex = self.get_all_connections()

        to_update = self.mer_name_indices[self.mer_names.index(name_to_update)]

        for mer in to_update:
            valency = len(connex[mer]) - 1
            if valency < 1:
                self.system.particles[self.mers[mer][1][0]].type = name_to_update + "1"
            elif valency < 3:
                self.system.particles[self.mers[mer][1][0]].type = name_to_update + "1" + "_1"
            elif valency < 5:
                self.system.particles[self.mers[mer][1][0]].type = name_to_update + "1" + "_2"
            else:
                self.system.particles[self.mers[mer][1][0]].type = name_to_update + "1" + "_3"


        print(74, self.system.particles[74].type)
        for ind, newmer in enumerate(to_update):
            #self.system.particles[self.mers[newmer][2][0]].body = -1
            print(self.mers[newmer][1][0], self.system.particles[self.mers[newmer][1][0]].type,
                  self.system.particles[self.mers[newmer][1][0]].typeid)
            self.system.particles[self.mers[newmer][1][0]].type = "B1_1"
            print(self.mers[newmer][1][0], self.system.particles[self.mers[newmer][1][0]].type,
                  self.system.particles[self.mers[newmer][1][0]].typeid)
            self.system.particles[ind + 74].type = "center1_1"
        print(74, self.system.particles[74].type)

        print(self.system.particles[self.cargo[0]].type)
        print(self.system.particles[self.cargo[0]].typeid)
        self.system.particles[self.cargo[0]].type = "A0"
        print(self.system.particles[self.cargo[0]].type)
        print(self.system.particles[self.cargo[0]].typeid)
        self.rigid.set_rigid_bodies(self.system, reset=True)


    def look_for_connection(self, mer1, mer2, cut=.5):

        positions1 = np.array([self.get_position(index1) for index1 in self.mers[mer1][3]])

        positions2 = np.array([self.get_position(index1) for index1 in self.mers[mer2][3]])

        dist_array = np.linalg.norm(np.subtract(positions1, positions2))

        if np.any(dist_array < cut):
            return True
        return False

    def get_position(self, index):

        return self.system.particles[index].position

    def get_all_connections(self, cut=0.5):

        connects = [[] for _ in range(len(self.mers))]
        for i in range(len(self.mers)):
            for j in range(len(self.mers)):
                connected = self.look_for_connection(i, j, cut=cut)
                if connected:
                    connects[i].append(j)
        return connects


    def run_nanoseconds(self, time):

        real_time = int(time * 1e-9 / (self.time_unit * self.dt))
        self.run(real_time)

    def nve_relaxation(self, time):


         self.langevin.disable()
         self.nve.enable()

         hoomd.run(time)
         self.nve.set_params(limit=.01)
         #hoomd.run(time / 2)
         #self.nve.set_params(limit=.001)
         #self.nve.set_params(limit=.01)
         #hoomd.run(time)
         #self.nve.set_params(limit=.1)
         #hoomd.run(time)
         #self.nve.set_params(limit=1)
         self.nve.disable()
         self.langevin.enable()

    def set_dt(self, dt):
        hoomd.md.integrate.mode_standard(dt=dt)

    def run_fire(self, time):

        self.langevin.disable()
        self.nve.enable()
        fire = hoomd.md.integrate.mode_minimize_fire(dt=0.1, group=self.to_integrate, ftol=1e-2, Etol=1e-7)
        hoomd.run(time)
        del fire
        self.langevin.enable()
        self.nve.disable()
        hoomd.md.integrate.mode_standard()

    def temp_interp(self, temp1, temp2, time):

        t1 = temp1
        t2 = temp2
        self.langevin.set_params(kT=hoomd.variant.linear_interp(points=[(0, t1), (time, t2)]))
        hoomd.run(time)
        self.langevin.set_params(kT=self.temperature)

    def set_temperature(self, t):

        self.temperature = t
        self.langevin.set_params(kT=self.temperature)

    def box_interp(self, old_size, new_size, time):

        up = hoomd.update.box_resize(L=hoomd.variant.linear_interp(points=[(0, old_size), (time, new_size)]))
        hoomd.run(time)
        up.disable()

    def set_log_period(self, period):

        self.logger.disable()
        self.log_period = period
        log_name = self.name + ".log"
        self.logger = hoomd.analyze.log(filename=log_name, quantities=self.log_list, period=self.log_period,
                                        overwrite=True)

    def set_dump_period(self, period):

        self.dumper.disable()
        self.dump_period = period
        dump_name = self.name + ".gsd"
        self.dumper = hoomd.dump.gsd(filename=dump_name, period=self.dump_period, group=self.all, overwrite=True)


    def total_kinetic_energy(self):

        ke = 0
        for part in self.system.particles:
            kin = .5 * part.mass * np.linalg.norm(part.velocity) ** 2
            print(part.type, kin)
            ke += kin


        return ke

    def ndof(self):

        return self.total_kinetic_energy() * 2 / self.temperature

    def remove_rigid_template_from_integration(self):

        for part in self.system.particles:
            name = part.type
            if name[:6] == 'center' and len(name) > 6 and part.body > -1:
                group_gone = hoomd.group.type(name=name[:6], type=name)
                self.to_integrate = hoomd.group.difference(name="", a=self.to_integrate, b=group_gone)

    def check_mobile(self):
        center_number=0
        for part in self.system.particles:
            if part.type[:6] == 'center' and len(part.type)>6:
                center_number += 1
        if center_number == 1:
             return False

        return True

    def read_map(self, map_name):

        f = open(map_name, 'r')

        data = f.readlines()
        for ind, line in enumerate(data[:-1]):
            s = line.split()
            if s[0] not in self.mer_names:
                self.mer_names.append(s[0])
            if not len(self.mer_name_indices) == len(self.mer_names):
                self.mer_name_indices.append([ind])
            else:
                self.mer_name_indices[self.mer_names.index(s[0])].append(ind)
            mer = [[] for _ in range(7)]
            index = 1
            mer_index = -1
            while index < len(s):
                try:
                    tester = unicode(s[index], 'utf-8')
                except:
                    tester = s[index]
                if not tester.isnumeric():
                    mer_index += 1
                else:
                    mer[mer_index].append(int(s[index]))
                    #print(mer_index)
                index += 1
            self.mers.append(mer)
        line = data[-1]
        for thing in line[1:].split():
            self.cargo.append(int(thing))



class InitGSD(Simulation):

    def __init__(self, name, frame, gpu=True, eps_sc=1, eps_cc=1, eps_ss=5, hagan=False, reinit=None,
                 bend_ss=0, bend_funcs=0, allostery=False, nocargo=True):

        if not gpu:
            hoomd.context.initialize("--mode=cpu")
        else:
            hoomd.context.initialize("--mode=gpu")
        system = hoomd.init.read_gsd(name, frame=frame)

        i = 0

        while not name[i].isalpha():
            i = i + 1

        name_no_loc = name[i:]

        super(InitGSD, self).__init__(system, name=name_no_loc[:-4] + '_frame' + str(frame), eps_ss=eps_ss,
                                      eps_cc=eps_cc, bend_ss=bend_ss, bend_funcs=bend_funcs, eps_sc=eps_sc,
                                      hagan=hagan, reinit=reinit, allostery=allostery, map_name=name_no_loc[:-4]+".map")
