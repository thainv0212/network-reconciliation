import dill
import numpy as np
# the graph that will be used in the algorithm
class Graph():
    def __init__(self, links_data, global_rank, link_strengs, node_alias, nodes, neighbors, node_embedding=None):
        self.links_data = links_data
        self.global_rank = global_rank
        self.link_strengs = link_strengs
        self.size = len(global_rank)
        ## sort nodes by global_rank
        ## map rank to node
        self.node_by_rank = np.array(list(reversed(np.argsort(self.global_rank))))
        ## map node to rank
        self.rank_by_node = np.zeros(shape=(len(self.node_by_rank), ), dtype='int')
        for i in range(len(self.node_by_rank)):
            self.rank_by_node[self.node_by_rank[i]] = i

        self.mapped = {}
        for i in range(len(global_rank)):
            self.mapped[i] = False
        self.node_alias = node_alias
        self.node_list = nodes
        self.neighbors = neighbors
        self.node_embedding = None

    @staticmethod
    def load(data_file):
        return dill.load(open(data_file, 'rb'))
    def save(self, data_file):
        dill.dump(self, open(data_file, 'wb'))