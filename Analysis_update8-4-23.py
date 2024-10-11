from __future__ import division
import gsd.hoomd
import gsd.fl
import numpy as np
import numpy.linalg as la
import os.path
import networkx as nx
import copy as cp
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 22})
import hoomd.data as hbox
from mpl_toolkits.mplot3d import Axes3D
import math as m
import MDAnalysis as mda
from MDAnalysis.analysis import distances


class Analysis(object):

    def __init__(self, gsd_name, map_name):
        f = gsd.fl.open(name=gsd_name, mode='rb', application='', schema='hoomd',
                        schema_version=[1, 0])
        self.trajectory = gsd.hoomd.HOOMDTrajectory(f)
        self.mers = []
        self.mer_names = []
        self.mer_name_indices = []
        self.cargo = []
        self.read_map(map_name)
        self.frames = []
        self.box = self.trajectory.read_frame(0).configuration.box[:6]
        self.gsd_name = gsd_name

        self.graphs = []
        self.graph_frames = []

        self.cargo_graph_frames = []
        self.cargo_graphs = []

    def get_box(self, frame):
        self.box = self.trajectory.read_frame(frame).configuration.box[:6]

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
                if not s[index].isnumeric():
                    mer_index += 1
                else:
                    mer[mer_index].append(int(s[index]))
                    #print(mer_index)
                index += 1
            self.mers.append(mer)
        line = data[-1]
        for thing in line[1:].split():
            self.cargo.append(int(thing))

    def look_for_connection(self, mer1, mer2, frame, cut=.5):

        positions1 = np.array([self.get_position(index1, frame) for index1 in self.mers[mer1][3]])
        #print(self.mers[47][4])
        #quit()

        positions2 = np.array([self.get_position(index1, frame) for index1 in self.mers[mer2][3]])

        dist_array = distances.distance_array(positions1, positions2)

        if np.any(dist_array < cut):
            return True, self.mer_mer_bending_angle(mer1, mer2, frame)
        return False, 0

    def look_for_adsorbtion(self, mer, frame, cut=1.15):

        positions1 = np.array([self.get_position(index1, frame) for index1 in self.mers[mer][1]])
        # print(self.mers[47][4])
        # quit()

        positions2 = np.array([self.get_position(cargo, frame) for cargo in self.cargo])

        dist_array = distances.distance_array(positions1, positions2)

        if np.any(dist_array < cut):
            return True
        return False


    def look_for_cluster_adsorbtion(self, cluster, frame, cut=1.15):

        if len(cluster) < 3:
            return False

        frame = self.trajectory.read_frame(frame)

        return np.sum([int(self.look_for_adsorbtion(mer, frame)) for mer in cluster]) > .2 * len(cluster)


    def get_cargo_shell_connections(self, frame, cut=1.15):

        positions_cargo = np.array([self.get_position(cargo, frame) for cargo in self.cargo])

        position_1_mers = np.array([self.get_position(mer[1][0], frame) for mer in self.mers])
        dist_array = distances.distance_array(position_1_mers, positions_cargo)
        return dist_array < cut


    def get_globules(self, frame, cut=1.15):

        cargo_positions = np.array([self.get_position(cargo, frame) for cargo in self.cargo])

        dist_array = distances.distance_array(cargo_positions, cargo_positions, box=self.box)

        dist_array = dist_array < cut

        return dist_array


    def mer_mer_bending_angle(self, mer1, mer2, frame):

        zero_1 = self.get_position(self.mers[mer1][0][0], frame)
        zero_2 = self.get_position(self.mers[mer2][0][0], frame)

        positions_2 = np.array([self.get_position(index, frame) for index in self.mers[mer2][6]])
        chosen_1 = positions_2[np.argmin(distances.distance_array(positions_2, zero_1))]

        positions_1 = np.array([self.get_position(index, frame) for index in self.mers[mer1][6]])
        chosen_2 = positions_1[np.argmin(distances.distance_array(positions_1, zero_2))]

        #print(mer1, self.mers[mer1])
        #print(mer2, self.mers[mer2])
        #print(chosen_2)
        #quit()
        return self.angle(zero_1, np.average([chosen_1, chosen_2], axis=0), zero_2)

    def angle(self, pos1, pos2, pos3):

        vec1 = np.subtract(pos2, pos1)
        vec2 = np.subtract(pos2, pos3)

        return np.arccos( np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))


    def get_all_connections(self, frame, cut=0.5):

        connects = [[] for _ in range(len(self.mers))]
        for i in range(len(self.mers)):
            for j in range(i+1, len(self.mers)):
                connected, angle = self.look_for_connection(i, j, frame, cut=cut)
                if connected:
                    connects[i].append((j, angle))
        return connects

    def make_graph(self, frame):

        if frame in self.graph_frames:
            g = self.graphs[self.graph_frames.index(frame)]
            return g
        else:
            self.graph_frames.append(frame)

        frame = self.trajectory.read_frame(frame)
        G = nx.Graph()
        color_map = []
        names = ["A", "B", "N"]
        colors = ['blue', 'green', 'yellow']
        total_number = len(self.mers)
        for i in range(total_number):
            name = self.mer_names[np.argmax([int(i in self.mer_name_indices[ind]) for ind in range(len(self.mer_names))])]
            G.add_node(i, name=name, color=colors[names.index(name)])
            color_map.append(colors[names.index(name)])

        clusters = self.get_all_connections(frame)

        for i in range(total_number):
            for thing in clusters[i]:
                    G.add_edge(i, thing[0])
                    G[i][thing[0]]['angle'] = thing[1]

        self.graphs.append(G)
        return G

    def make_cargo_graph(self, frame):

        if frame in self.cargo_graph_frames:
            #print(self.cargo_graph_frames, len(self.cargo_graphs))
            g = self.cargo_graphs[self.cargo_graph_frames.index(frame)]
            return g
        else:
            self.cargo_graph_frames.append(frame)

        frame = self.trajectory.read_frame(frame)
        G = nx.Graph()
        color_map = ["red"]
        total_number = len(self.cargo)
        for i in range(total_number):
            G.add_node(i, name="cargo", color="red")
            color_map.append("red")

        clusters = self.get_globules(frame)

        for ind in range(total_number):
            for ind2, thing in enumerate(clusters[ind]):
                if thing and ind2 > ind:
                    G.add_edge(ind, ind2)

        self.cargo_graphs.append(G)
        return G



    def make_total_graph(self, frame):

        shell_graph = self.make_graph(frame)
        cargo_graph = self.make_cargo_graph(frame)
        total_graph = nx.disjoint_union(shell_graph, cargo_graph)
        frame = self.trajectory.read_frame(frame)
        shell_cargo_connect = self.get_cargo_shell_connections(frame)

        for ind in range(len(self.mers)):
            for ind2 in range(len(self.cargo)):
                if shell_cargo_connect[ind][ind2]:
                    total_graph.add_edge(ind, ind2 + len(self.mers))
        return total_graph


    def graph_cargo_network(self, frame):

        G = self.make_cargo_graph(frame)
        self.network_grapher(G, name="cargo_" + str(frame))

    def graph_network(self, frame):

        G = self.make_graph(frame)
        self.network_grapher(G, name=frame)

    def graph_total_network(self, frame):

        G = self.make_total_graph(frame)
        self.network_grapher(G, name="total_net_" + str(frame))

    def network_grapher(self, G, name="g"):

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        ax1.set_title(name)
        #nx.draw_networkx(g1, with_labels=True)
        color_map = [G.nodes[i]['color'] for i in G.nodes]
        nx.draw_networkx(G, with_labels=True, node_color=color_map)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for pos in ['right','top','bottom','left']:
            plt.gca().spines[pos].set_visible(False)

        plt.savefig("network_" + str(name) + ".png")
        plt.show()

    def n_largest_globule(self, frame):

        graph = self.make_cargo_graph(frame)
        clusters = list(nx.connected_components(graph))
        n_clusters = [len(c) for c in clusters]
        save = np.max(n_clusters)
        n_clusters.remove(save)
        return np.max(n_clusters) + save


    def average_degree(self, frames=None, types=None, allowed_combos=None):

        if frames is None:
            frames = self.graph_frames
        if types is None:
            types = self.mer_names

        degrees = []

        for frame in frames:
            graph = self.make_graph(frame)
            nodes = graph.nodes
            degree = self.get_degrees_from_graph(graph, types, allowed_combos)
            degrees.extend(degree)
        return np.average(degrees)

    def average_degree_of_adsorbed(self, frames=None, types=None, allowed_combos=None):

        if frames is None:
            frames = self.graph_frames
        if types is None:
            types = self.mer_names

        all_degrees = []


        for frame in frames:
            real_frame = self.trajectory.read_frame(frame)
            graph = self.make_graph(frame)
            clusters = list(nx.connected_components(graph))
            a_clusters = [c for c in clusters if self.look_for_cluster_adsorbtion(c, frame)]
            nodes = graph.nodes
            g_degrees = graph.degree()
            for degree in g_degrees:
                if np.sum([int(degree[0] in a_c) for a_c in a_clusters]) > 0:
                    all_degrees.append(degree[1])

        return np.average(all_degrees)

    def average_number_adsorbed(self, frames, cut=1.15):

        if frames is None:
            frames = self.graph_frames

        numbers = []
        for frame in frames:
            real_frame = self.trajectory.read_frame(frame)
            c_graph = self.make_cargo_graph(frame)
            c_clusters = list(nx.connected_components(c_graph))
            index = 0
            max = 0
            for ind, c in enumerate(c_clusters):
                if len(c) > max:
                    max = len(c)
                    index = ind
            largest_cluster = c_clusters[index]
            #print(largest_cluster)
            adsorbed_mers = []
            print(len(self.mers))
            for mer in range(len(self.mers)):
                positions1 = np.array([self.get_position(index1, real_frame) for index1 in self.mers[mer][1]])
                positions2 = np.array([self.get_position(cargo + len(self.mers), real_frame) for cargo in largest_cluster])
                dist_array = distances.distance_array(positions1, positions2)
                #print(np.min(dist_array))
                if np.any(dist_array < cut):
                    adsorbed_mers.append(mer)

            #print(len(largest_cluster), len(adsorbed_mers))
            numbers.append(len(adsorbed_mers))

        return np.average(numbers), np.std(numbers)


    def number_free_mers(self, frames):

        if frames is None:
            frames = self.graph_frames

        numbers = []
        for frame in frames:
            real_frame = self.trajectory.read_frame(frame)
            graph = self.make_graph(frame)
            clusters = nx.connected_components(graph)
            numbers.append(len([c for c in clusters if len(c) == 1]))
        return np.average(numbers) / len(self.mers), np.std(numbers) / len(self.mers)



    def average_degree_of_clustered(self, frames=None, types=None, allowed_combos=None):

        if frames is None:
            frames = self.graph_frames
        if types is None:
            types = self.mer_names

        all_degrees = []

        for frame in frames:
            real_frame = self.trajectory.read_frame(frame)
            graph = self.make_graph(frame)
            clusters = list(nx.connected_components(graph))
            a_clusters = [c for c in clusters if len(c) > 1]
            nodes = graph.nodes
            g_degrees = graph.degree()
            for degree in g_degrees:
                if np.sum([int(degree[0] in a_c) for a_c in a_clusters]) > 0 and self.get_name_from_mer_index(degree[0]) in types:
                    all_degrees.append(degree[1])
        return np.average(all_degrees)

    def graph_average_degree_of_clustered_time_series(self, frames=None, types=None):

        data_by_type = []
        for tipe in types:
            data = [self.average_degree_of_clustered(frames=[frame], types=[tipe]) for frame in frames]
            data_by_type.append(data)

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Average Valency of Clustered Shell Proteins")
        ax1.set_ylabel('Valency of Shell Proteins')
        ax1.set_xlabel('Time')

        colors = ["cyan", "red", "blue", 'k']
        color_names = ["A", "B", "N", "total"]

        for type in types:
            ax1.plot(frames, data_by_type[types.index(type)], label=type, color=colors[color_names.index(type)])

        plt.legend()
        plt.savefig("cluster_degree_stoich.png")
        plt.show()





    def no_pent_shell_score(self, frame):

        all_degrees = []
        real_frame = self.trajectory.read_frame(frame)
        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        a_clusters = [c for c in clusters if self.look_for_cluster_adsorbtion(c, frame)]
        nodes = graph.nodes
        g_degrees = graph.degree()
        for degree in g_degrees:
            if np.sum([int(degree[0] in a_c) for a_c in a_clusters]) > 0:
                all_degrees.append(degree[1])

        n = len(all_degrees)
        #print(n, (6 * (n-10)/n))

        return np.average(all_degrees) / (6 * (n-10)/n), 6*n  - np.sum(all_degrees) - 60

    def average_angle(self, frames=None, types=None, allowed_combos=None):

        if frames is None:
            frames = self.graph_frames
        if types is None:
            types = self.mer_names
        if allowed_combos is not None:
            types = allowed_combos

        angles = []

        for frame in frames:
            graph = self.make_graph(frame)
            frame_angle = self.get_angles_from_graph(graph, types, allowed_combos)
            angles.extend(np.rad2deg(frame_angle))
        return np.average(angles)

    def angle_distribution(self, frames=None, types=None, allowed_combos=None, min_size=0):

        if frames is None:
            frames = self.graph_frames
        if types is None:
            types = self.mer_names
        if allowed_combos is not None:
            types = allowed_combos

        angles = []

        for frame in frames:
            graph = self.make_graph(frame)
            frame_angle = self.get_angles_from_graph(graph, types, allowed_combos, min_size=min_size)
            angles.extend(np.rad2deg(frame_angle))

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Angle Distribution")
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('P')


        hist, be = np.histogram(angles, bins=[90 + i for i in range(0,91,5)])
        bin_middles = [(be[i] + be[i + 1]) / 2 for i in range(len(be) - 1)]
        hist = [h * be[ind] for ind, h in enumerate(hist)]
        hist = np.divide(hist, np.sum(hist))
        ax1.plot(bin_middles, hist)
        plt.legend()
        plt.savefig("angle_dist" +str(allowed_combos) + ".png")
        plt.show()



    def get_angles_from_graph(self, graph, types, allowed_combos, min_size=0):

        nodes = graph.nodes
        edges = graph.edges()
        clusters = list(nx.connected_components(graph))
        cluster_size_in = [0 for _ in range(len(list(nodes)))]
        for cluster in clusters:
            for thing in cluster:
                cluster_size_in[thing] = len(cluster)

        frame_angle = [graph.edges[edge]['angle'] for edge in edges
                       if nodes[edge[0]]['name'] in types and nodes[edge[1]]['name'] in types and cluster_size_in[edge[0]] > min_size]
        if allowed_combos is not None:
            frame_angle = [graph.edges[edge]['angle'] for edge in edges
                           if ((nodes[edge[0]]['name'] + nodes[edge[1]]['name'] in allowed_combos) or
                           (nodes[edge[1]]['name'] + nodes[edge[0]]['name'] in allowed_combos)) and cluster_size_in[edge[0]] > min_size]
        return frame_angle

    def count_6fold_hexes(self, frame):

        graph = self.make_graph(frame)
        print(self.get_degrees_from_graph(graph, ["A"], ["AA"]))
        return self.get_degrees_from_graph(graph, ["A"], ["AA"]).count(6)

    def get_degrees_from_graph(self, graph, types, allowed_combos):

        nodes = graph.nodes
        edges = graph.edges()
        degree = graph.degree()
        if types == self.mer_names and allowed_combos is None:
            return degree

        fake_edges = [[] for i in range(len(nodes))]
        for edge in edges:
            if allowed_combos is None:
                if nodes[edge[0]]['name'] in types and nodes[edge[1]]['name'] in types:
                    fake_edges[edge[0]].append(edge[1])
                    fake_edges[edge[1]].append(edge[0])
            else:
                if nodes[edge[0]]['name'] + nodes[edge[1]]['name'] in allowed_combos:
                    fake_edges[edge[0]].append(edge[1])
                if nodes[edge[1]]['name'] + nodes[edge[0]]['name'] in allowed_combos:
                    fake_edges[edge[1]].append(edge[0])

        if allowed_combos is None:
            return [len(partners) for ind, partners in enumerate(fake_edges) if nodes[ind]['name'] in types]
        else:
            allowed_starters = [combo[0] for combo in allowed_combos]
            return [len(edge) for ind, edge in enumerate(fake_edges) if not len(edge) == 0 or nodes[ind]['name'] in allowed_starters]

    def cluster_sizes(self, frame):

        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        #print(clusters)
        #print(graph.nodes)
        for cluster in clusters:
            sub = nx.subgraph(graph, cluster)
            if self.mathematically_closed(sub):
                print("CLOSURE")
            #print(sub.nodes)
            #if len(cluster) == 12:
             #   self.network_grapher(sub)
             #   print(sub.degree)
        return sorted([len(list(c)) for c in clusters])

    def globule_sizes(self, frame):

        graph = self.make_cargo_graph(frame)
        clusters = list(nx.connected_components(graph))

        return sorted([len(list(c)) for c in clusters])

    def cluster_size_distribution(self, frames):

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Cluster Size Distribution")
        ax1.set_xlabel('# of Shell Proteins')
        ax1.set_ylabel('P')

        for frame in frames:
            sizes = self.cluster_sizes(frame)
            hist, be = np.histogram(sizes, bins=[0.5 + i for i in range(max(sizes) + 1)])
            bin_middles = [(be[i] + be[i + 1]) / 2 for i in range(len(be) - 1)]
            hist = [h * be[ind] for ind, h in enumerate(hist)]
            hist = np.divide(hist, np.sum(hist))
            ax1.plot(bin_middles, hist, label="Frame " + str(frame))
        plt.legend()
        plt.show()

    def cluster_adsorbed_distribution(self, frames):

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Cluster Size Distribution")
        ax1.set_xlabel('# of Shell Proteins')
        ax1.set_ylabel('P')

        colors = ["b", "g", "r", "c", "m", "y", "k"]

        for color_ind, frame in enumerate(frames):
            graph = self.make_graph(frame)
            clusters = list(nx.connected_components(graph))
            sizes_on = [len(cluster) for cluster in clusters if self.look_for_cluster_adsorbtion(cluster, frame)]
            if len(sizes_on) > 0:
                hist, be = np.histogram(sizes_on, bins=[0.5 + i for i in range(max(sizes_on) + 1)])
                bin_middles = [(be[i] + be[i + 1]) / 2 for i in range(len(be) - 1)]
                hist = [h * be[ind] for ind, h in enumerate(hist)]
                hist = np.divide(hist, len(self.mers))
                ax1.plot(bin_middles, hist, label="On_Frame " + str(frame), color=colors[color_ind % len(colors)])

            sizes_off = [len(cluster) for cluster in clusters if not self.look_for_cluster_adsorbtion(cluster, frame)]
            if len(sizes_off) > 0:
                hist, be = np.histogram(sizes_off, bins=[0.5 + i for i in range(max(sizes_off) + 1)])
                bin_middles = [(be[i] + be[i + 1]) / 2 for i in range(len(be) - 1)]
                hist = [h * be[ind] for ind, h in enumerate(hist)]
                hist = np.divide(hist, len(self.mers))
                ax1.plot(bin_middles, hist, label="Off_Frame " + str(frame), linestyle="dashed", color=colors[color_ind % len(colors)])

        plt.savefig("size_dist" + str(frame) + ".png",  bbox_inches='tight', pad_inches=.2)
        plt.legend()
        plt.show()


    def critical(self, frames):

        for f_ind, frame in enumerate(frames):
            graph = self.make_graph(frame)
            clusters = list(nx.connected_components(graph))
            for cluster in clusters:
                if self.look_for_cluster_adsorbtion(cluster, frame):
                    if f_ind < len(frames) - 1:
                        if self.look_for_cluster_adsorbtion(cluster, frames[f_ind + 1]):
                            if f_ind < len(frames) - 2:
                                if self.look_for_cluster_adsorbtion(cluster, frames[f_ind + 1]):
                                    return frame, len(cluster)
                            else:
                                return frame, len(cluster)
                    else:
                        return frame, len(cluster)
        return 0, 0


    def cluster_pentamer_distribution(self, frames):

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Cluster Size Distribution")
        ax1.set_xlabel('# of Pentamer Proteins')
        ax1.set_ylabel('P')

        for frame in frames:
            sizes = self.cluster_pentamers(frame)
            hist, be = np.histogram(sizes, bins=[0.5 + i for i in range(max(sizes) + 1)])
            bin_middles = [(be[i] + be[i + 1]) / 2 for i in range(len(be) - 1)]
            hist = [h * be[ind] for ind, h in enumerate(hist)]
            hist = np.divide(hist, np.sum(hist))
            ax1.plot(bin_middles, hist, label="Frame " + str(frame))
        plt.legend()
        plt.show()

    def cluster_pentamers(self, frame):

        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        pentamers = []
        for cluster in clusters:
            plist = []
            for thing in cluster:
                if graph.nodes[thing]['name'] == "N":
                    plist.append(thing)
            pentamers.append(plist)

        return sorted([len(list(c)) for c in pentamers])

    def cluster_stoichiometry(self, frame, monomers=True):

        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        results = []
        for clusters in clusters:
            stoich = {}
            for mer in clusters:
                name = self.get_name_from_mer_index(mer)
                if name in stoich.keys():
                    stoich[name] += 1
                else:
                    stoich[name] = 1
            stoich['total'] = np.sum(stoich[key] for key in stoich.keys())
            if stoich['total'] != 1 or monomers:
                results.append(stoich)
        return results

    def largest_cluster_stoichiometry(self, frame):

        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        sizes = [len(cluster) for cluster in clusters]
        biggest = sizes.index(max(sizes))
        return self.cluster_stoichiometry(frame)[biggest]

    def type_contact_percentage_on_shell(self, frame, typ):

        graph = self.make_graph(frame)
        nodes = graph.nodes
        edges = graph.edges()
        #print(edges)
        clusters = list(nx.connected_components(graph))
        allowed = []

        for cluster in clusters:
            if self.look_for_cluster_adsorbtion(cluster, frame):
                for thing in cluster:
                    allowed.append(thing)

        connected_types = []

        for edge in edges:
            if nodes[edge[0]]['name'] == typ and edge[0] in allowed:
                connected_types.append(nodes[edge[1]]['name'])
            elif nodes[edge[1]]['name'] == typ and edge[1] in allowed:
                connected_types.append(nodes[edge[0]]['name'])

        #print(self.mer_name_indices)
        #print("total", np.sum([deg[1] for deg in graph.degree
         #                      if deg[0] in allowed and typ == self.get_name_from_mer_index(deg[0])]))

        result = [connected_types.count(name) for name in self.mer_names]
        return result
        #return np.divide(result, np.sum(result) / 100)

    def get_largest_cluster_stoichiometry_time(self, frames):

        data = []
        names = []
        for frame in frames:
            stoich = self.largest_cluster_stoichiometry(frame)
            for key in stoich.keys():
                if key not in names:
                    names.append(key)
                    new_data = [0 for _ in range(frames.index(frame))]
                    new_data.append(stoich[key])
                    data.append(new_data)
                else:
                    data[names.index(key)].append(stoich[key])
            for name in names:
                if name not in stoich.keys():
                    data[names.index(name)].append(0)
        return names, data


    def largest_cluster_stoichiometry_time_series(self, frames):

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Largest Cluster Stoichiometry")
        ax1.set_ylabel('# of Shell Proteins')
        ax1.set_xlabel('Time, $\\tau$')

        data = []
        names = []
        colors = ["cyan", "red", "blue", 'k']
        color_names = ["A", "B", "N", "total"]
        for frame in frames:
            stoich = self.largest_cluster_stoichiometry(frame)
            for key in stoich.keys():
                if key not in names:
                    names.append(key)
                    new_data = [0 for _ in range(frames.index(frame))]
                    new_data.append(stoich[key])
                    data.append(new_data)
                else:
                    data[names.index(key)].append(stoich[key])
            for name in names:
                if name not in stoich.keys():
                    data[names.index(name)].append(0)


        for name in names:
            namex = name
            if name == "N":
                namex = "Z"
            ax1.plot(frames, data[names.index(name)], label=namex, color=colors[color_names.index(name)])

        plt.legend(bbox_to_anchor=(1.2,1.2))
        plt.savefig("cluster_addition_stoich.png", bbox_inches='tight', pad_inches=.2)
        plt.show()

    def get_name_from_mer_index(self, mer_index):

        for ind, set in enumerate(self.mer_name_indices):
            if mer_index in set:
                return self.mer_names[ind]
        return None




    def pentamer_shortest_paths(self, frame):

        graph = self.make_graph(frame)
        clusters = list(nx.connected_components(graph))
        pentamers = []
        for cluster in clusters:
            plist = []
            for thing in cluster:
                if graph.nodes[thing]['name'] == "N":
                    plist.append(thing)
            pentamers.append(plist)


        path_lengths = []
        for cluster in pentamers:
            for ind, pent in enumerate(cluster):
                for ind2 in range(ind+1, len(cluster)):
                    path = nx.shortest_path(graph, source=cluster[ind], target=cluster[ind2])
                    print(cluster[ind], cluster[ind2], path)
                    path_lengths.append(len(path) - 1)

        return path_lengths

    def mathematically_closed(self, g):
        degree = g.degree

        #print(degree)
        for ind, valency in g.degree:
            need = 6
            if g.nodes[ind]['name'] == "N":
                need = 5
            if valency != need:
                return False
        return True


    def get_position(self, index, frame):

        #print(frame.particles.image[index])
        #return np.add(frame.particles.position[index], np.multiply(frame.particles.image[index], self.box))
        return frame.particles.position[index]

    def get_type(self, index, frame):

        if not isinstance(index,int):
            #print("Warning index "  + str(index) + "is not an integer")
            return("!!!Faketype")
        return frame.particles.types[frame.particles.typeid[index]]

    #def reduced_graph(self, frame):

     #   g = self.make_graph(frame)
     #   self.network_grapher(g)
     #   names = [g.nodes[i]['name'] for i in g.nodes]
     #   print(names)
     #   deg = g.degree()
     #   print(deg)
     #   print(deg[0])
     #   for ind, name in enumerate(names):
     #       if name == 'N':
     #           g.remove_node(ind)
     #       elif deg[ind] == 6 or deg[ind] < 2:
     #           g.remove_node(ind)
     #   print(g.nodes)
     #   self.network_grapher(g)
     #   g2 = nx.DiGraph(g)

      #  print(list(nx.connected_components(g)))
      #  to_remove = []

