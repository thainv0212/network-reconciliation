from copy import deepcopy
import datetime
import math
import multiprocessing as mp
import numpy as np
from queue import Queue
from sklearn.utils import shuffle
from sys import argv
import tqdm
import time
from data import Graph

class Algorithm:
    def __init__(self, g1: Graph, g2: Graph, seed_list: list=[],
                C:float=1.7, desired_seed_size:int=0, process_num=1, threshold=2, full_graph_size=0):
                
        print('initialization started', threshold)
        self.g1 = g1
        self.g2 = g2
        self.seed_list = seed_list
        self.g1_to_g2_link = {}
        self.g2_to_g1_link = {}
        self.g1_to_g2_link_prob = {}
        self.g2_to_g1_link_prob = {}
        self.graph_size = len(self.g1.link_strengs)
        self.C = C
        self.desired_seed_size = desired_seed_size
        self.threshold = threshold
        if full_graph_size == 0:
            self.full_graph_size = self.g1.size
        else:
            self.full_graph_size = full_graph_size
        
        print('initialize mapped node list')
        ## init mapped nodes list
        for i in range(len(self.g1.global_rank)):
            self.g1_to_g2_link[i] = -1
        for i in range(len(self.g2.global_rank)):
            self.g2_to_g1_link[i] = -1
            
        for i in seed_list:
            self.g1.mapped[i[0]] = True
            self.g2.mapped[i[1]] = True
            self.g1_to_g2_link[i[0]] = i[1]
            self.g2_to_g1_link[i[1]] = i[0]

        print('initialize list of each graph')
        ## init neighbor list of each graph
        print('initialize list of g1')
        self.g1_neighbors = []
        for i in range(len(self.g1.global_rank)):
            neighbors = self.g1.links_data.loc[(self.g1.links_data.user_1 == i)|(self.g1.links_data.user_2 == i)].values.flatten()
            neighbors = neighbors[neighbors != i]
            self.g1_neighbors.append(set(neighbors) - set([i]))
        
        print('initialize list of g2')
        self.g2_neighbors = []
        for i in range(len(self.g2.global_rank)):
            neighbors = self.g2.links_data.loc[(self.g2.links_data.user_1 == i)|(self.g2.links_data.user_2 == i)].values.flatten()
            neighbors = neighbors[neighbors != i]
            self.g2_neighbors.append(set(neighbors) - set([i]))
        self.mapped_pairs = {}
        self.mapped_probs = {}
        self.process_num = process_num
        self.map = set()
        print('initialization ended')


    ## global similarity of 2 nodes 
    def sim_global(self, g1_node: int, g2_node: int):
        R1 = self.g1.global_rank[g1_node]
        R2 = self.g2.global_rank[g2_node]
        return min(R1, R2)/max(R1, R2)

    ## local similarity of 2 nodes
    def sim_local(self, g1_node: int, g2_node: int, debug=False):
        # mapped neighbors of g1_node
        # self.g1.links_data
        neighbors_1 = self.g1_neighbors[g1_node]
        # mapped_nodes_1 = set([n for n in neighbors_1 if n in self.g1_to_g2_link.keys()])
        mapped_nodes_1 = set([n for n in neighbors_1 if self.g1_to_g2_link[n] != -1])
        mapped_nodes_1 = set([self.g1_to_g2_link[n] for n in mapped_nodes_1])

        neighbors_2 = self.g2_neighbors[g2_node]
        # mapped_nodes_2 = set([n for n in neighbors_2 if n in self.g2_to_g1_link.keys()])
        # try:
        mapped_nodes_2 = set([n for n in neighbors_2 if self.g2_to_g1_link[n] != -1])
        # except Exception as ex:
        #     print(ex)

        L1 = self.g1.link_strengs[g1_node]
        L2 = self.g2.link_strengs[g2_node]
        union = len(mapped_nodes_1.union(mapped_nodes_2))
        intersect = len(mapped_nodes_1.intersection(mapped_nodes_2))
        if debug:
            print('sim local', union, intersect, L1, L2, mapped_nodes_1, mapped_nodes_2)
        if union == 0 or max(L1, L2) == 0:
            return 0

        return intersect/ union * min(L1, L2)  / max(L1, L2)

    ## similarity of 2 nodes
    def sim_uni(self, g1_node: int, g2_node: int):
        sim_local = self.sim_local(g1_node, g2_node)
        sim_global = self.sim_global(g1_node, g2_node)
        alpha = math.log(self.g1.global_rank[g1_node]/self.full_graph_size + self.C)
        return alpha * sim_local + (1 - alpha) * sim_global

    ## seed exploration
    def explore_seed(self):
        loop_count = len(self.seed_list)
        print('need to find {} nodes'.format(loop_count))
        print(len([u for u in self.g1_to_g2_link.keys() if self.g1_to_g2_link[u]!=-1]))
        while loop_count <= self.desired_seed_size:
            # for u in range(len(self.g1.global_rank)):
            for u in self.g1.node_by_rank:
                # if not self.g1.mapped[u]:
                if self.g1_to_g2_link[u] == -1:
                    # find v in g2
                    rk = int(self.g1.rank_by_node[u])
                    # print(rk, type(rk))
                    last_idx = min(2 * rk, len(self.g2.node_by_rank))
                    CL = self.g2.node_by_rank[:last_idx]#.copy()
                    best_v =  -1
                    best_sim = -10000
                    for v in CL:
                        if self.g2.mapped[v]:
                            continue
                        sim = self.sim_uni(u, v)
                        if sim > best_sim:
                            best_sim = sim
                            best_v = v
                    # print('v found, now start to look up u on g1', u, best_v)
                    # using v, find u' in g1, if u' == u then (u, v) is matched
                    rk = int(self.g2.rank_by_node[best_v])
                    CL = self.g1.node_by_rank[:2 * rk]#.copy()
                    best_u = -1
                    best_sim = -1000
                    for u_ in CL:
                        if self.g1.mapped[u_]:
                            continue
                        sim = self.sim_uni(u_, best_v)
                        if sim > best_sim:
                            best_sim = sim
                            best_u = u_
                    if best_u == u:
                        self.g1_to_g2_link[u] = best_v
                        self.g2_to_g1_link[best_v] = u
                        self.g1_to_g2_link_prob[u] = best_sim
                        self.g2_to_g1_link_prob[best_v] = best_sim
                        self.g1.mapped[u]=True
                        self.g2.mapped[best_v] = True
                        # print('looked up u', best_u)
                        print('add ({},{}) with prob {}'.format(self.g1.node_list[u], self.g2.node_list[best_v], best_sim), u, best_v, len([u for u in algo.g1_to_g2_link.keys() if algo.g1_to_g2_link[u]!=-1]), loop_count)
                        loop_count += 1
                        if loop_count >=self.desired_seed_size:
                            print('loop count', loop_count)
                            return
            pass

    ## seed expansion on one seed
    def expand_seed_one_seed(self, root):
        expanded_set = {}
        expanded_set_prob = {}
        # threshold = 5
        total_count = 0
        print('preparing data for expansion')
        g1_mapped = deepcopy(self.g1.mapped)
        g2_mapped = deepcopy(self.g2.mapped)
        g1_to_g2_link = deepcopy(self.g1_to_g2_link)
        g1_to_g2_link_prob = deepcopy(self.g1_to_g2_link_prob)
        g1_neighbors = deepcopy(self.g1.neighbors)
        g2_neighbors = deepcopy(self.g2.neighbors)
        print('expansion started')
        queue = Queue()
        queue.put(root)
        while not queue.empty():
            r = queue.get()
            # get all neighbors of r and get the node which has the highest global rank
            # print(type(g1_neighbors[r]))
            r_neighbors = np.array([t for t in list(g1_neighbors[r]) if len(list(g1_neighbors[t])) >= self.threshold and not g1_mapped[t]])
            # print('r_neighbors', r_neighbors)
            r_neighbors_ranks = np.array([self.g1.global_rank[u] for u in r_neighbors])
            sorted_neighbors = r_neighbors[list(reversed(np.argsort(r_neighbors_ranks)))]
            # print('sorted neighbors')
            # print('sorted neighbors', sorted_neighbors)
            # best_u_rank = r_neighbors_ranks[u_best]
            # get all mapped neighbors of u
            # u_best = np.argmax(r_neighbors_ranks)
            for u_best in sorted_neighbors:
                u_neighbors = list([t for t in g1_neighbors[u_best] if g1_mapped[t]])
                CL = []
                for u_quote in u_neighbors:
                    v_quote = g1_to_g2_link[u_quote] 
                    if v_quote == -1:
                        continue
                    # print('v_quote',v_quote,  self.g2_to_g1_link[u_quote], u_quote, r)
                    v_quote_neighbors = [t for t in g2_neighbors[v_quote] if not g2_mapped.get(t, False)]
                    for v in v_quote_neighbors:
                        CL.append(v)
                    pass
                # calculate similarity of u and all the v node in CL
                # print('CL', CL)
                if len(CL) == 0:
                    continue
                u_v_sims = np.array([self.sim_uni(u_best, v) for v in CL])
                # get the best v node
                best_v_idx = np.argmax(u_v_sims)
                v_best = CL[best_v_idx]
                best_u_v_sim = u_v_sims[best_v_idx]
                # add (u, v) to mapped node with prob P(u) = P(r)*Sim_uni(u,v)
                p_u = g1_to_g2_link_prob.get(r, 1) * best_u_v_sim
                g1_mapped[u_best] = True
                g2_mapped[v_best] = True
                expanded_set[u_best] = ([u_best, v_best])
                expanded_set_prob[u_best] = p_u
                queue.put(u_best)
                # print('added ({},{}) to queue with prob {}'.format(u_best, v_best, p_u))
                # self.map.add(u_best)
                with open('tmp.txt', 'a') as f:
                    f.write(str(u_best) + '\n')
                # print(self.map)
                total_count += 1
        print('expand on one seed ended', total_count)
        return expanded_set, expanded_set_prob

    ## do post-process (conflict resolution)
    ## replace (u,v) with (u, v') if (u, v') has higher probability
    def post_process_results(self, results):
        print('do post-process')
        for result in results:
            pairs, probs = result
            for u in tqdm.tqdm(pairs.keys()):
                v_tmp = pairs[u][1]
                prob_tmp = probs[u]
                v_curr = self.mapped_pairs.get(u, None)
                if v_curr is None:
                    v_curr = v_tmp
                    prob_curr = probs[u]
                else:
                    prob_curr = self.mapped_probs.get(u, -1)
                    if prob_curr < prob_tmp:
                        v_curr = v_tmp
                        prob_curr = prob_tmp
                print('do post_process', v_curr, type(v_curr), u, type(u))
                self.mapped_pairs[u] = v_curr
                self.mapped_probs[u] = prob_curr
                self.g1_to_g2_link[u] = v_curr
                self.g1_to_g2_link_prob[u] = prob_curr
                self.g2_to_g1_link[v_curr] = u
                self.g2_to_g1_link_prob[v_curr] = prob_curr
    
    ## expand nodes using multiprocess
    def expand_seed_multi_seed(self, roots):
        print('expand seed multi')
        results = []
        # pairs, probs = [], []
        for root in roots:
            result = self.expand_seed_one_seed(root)
            # pairs.append(pair)
            # probs.append(prob)
            results.append(result)
        print('expand multi-seed ended', len(results))
        return results

    ## do seed expansion in parallel
    def expand_parallel(self, pool):
        print('start expanding')
        t1 = datetime.datetime.now()
        roots = [u for u in self.g1_to_g2_link.keys() if self.g1_to_g2_link[u] != -1]
        # partitions = []
        # split roots
        root_count_per_part = len(roots) // self.process_num
        for i in range(self.process_num):
            start = i * root_count_per_part
            end = min(start + root_count_per_part, len(roots))
            tmp = pool.apply_async(self.expand_seed_multi_seed, args=(roots[start:end],), callback=self.post_process_results)
            # tmp.wait()
            print(start, end)
        pool.close()
        pool.join()
        t2 = datetime.datetime.now()
        print('time consumed {}'.format(t2 - t1))
        # return partitions

    def export_results(self, data_type):
        # undo aliasing then write to file
        print(len([u for u in self.g1_to_g2_link.keys() if self.g1_to_g2_link[u]!=-1]))
        with open('result_{}.txt'.format(data_type), 'w') as f:
            data = []
            for(u, v) in self.g1_to_g2_link.items():
                if v == -1:
                    continue
                unaliased_u = self.g1.node_list[u]
                unaliased_v = self.g2.node_list[v]
                data.append('{}\t{}'.format(unaliased_u, unaliased_v))
            text_data = '\n'.join(data)
            f.write(text_data)

# g1 = Graph.load('data_full/1.0_1.0/graph_0.pickle')
# g2 = Graph.load('data_full/1.0_1.0/graph_1.pickle')
# g3 = Graph.load('data_full/graph_full.pickle')
g1 = Graph.load('data_10000/{}/graph_0.pickle'.format(argv[1]))
g2 = Graph.load('data_10000/{}/graph_1.pickle'.format(argv[1]))
g3 = Graph.load('data_10000/graph_full.pickle')
# g1 = Graph.load('graph_0.pickle')
# g2 = Graph.load('graph_1.pickle')
# g3 = Graph.load('graph_2.pickle')
# inter = list(set(g1.node_list).intersection(set(g2.node_list)))
# inter = shuffle(inter, random_state=int(time.time()))
arr = np.array(list(range(6000)))
np.random.shuffle(arr)
joint_nodes = set(g1.node_list)
joint_nodes = list(joint_nodes.intersection(set(g2.node_list)))
count = 0
# seed_list = [[g1.node_alias[i], g2.node_alias[i]] for i in joint_nodes[:80]]
seed_list = []
for i in range(len(g3.node_by_rank)):
    node = g3.node_by_rank[i]
    if g1.node_alias.get(node, None) is None:
        continue
    if g2.node_alias.get(node, None) is None:
        continue
    seed_list.append([g1.node_alias[node], g2.node_alias[node]])
    count += 1
    if count == int(math.sqrt(len(g3.global_rank))):
        break
print(len(seed_list))
algo = Algorithm(g1, g2, seed_list=seed_list, desired_seed_size=len(g3.global_rank) * 2/100, process_num=6, threshold=2, full_graph_size=g3.size)
# start algorithm
pool = mp.Pool()
print('start seed exploration')
algo.explore_seed()
print(len([u for u in algo.g1_to_g2_link.keys() if algo.g1_to_g2_link[u]!=-1]))
print('start seed expansion')
algo.expand_parallel(pool)
print('write result')
algo.export_results(argv[1])
# print(algo.map)
# print(len(g1.global_rank), len(g2.global_rank))
# algo.expand_seed_one_seed(seed_list[0][0])