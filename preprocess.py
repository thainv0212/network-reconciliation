import dask
from dask import array
import gzip
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import tqdm
import datetime
import time
import dill
from scipy import sparse
import os.path as path

from data import  Graph
node_count = 63731
# network modeling
# read graph data
link_columns = ['user_1', 'user_2']
# read facebook links data
print('reading facebook links data')
file = gzip.GzipFile('facebook-links.txt.gz', 'rb')
data = file.read()
text_data = data.decode('ascii')
links_data_arr = []
lines = text_data.split('\n')
for line in tqdm.tqdm(lines):
    try:
        splits = line.split('\t')
        user_1 = int(splits[0]) - 1
        user_2 = int(splits[1]) - 1
        if user_1 > user_2:
            user_1 += user_2
            user_2 = user_1 - user_2
            user_1 = user_1 - user_2
#         timestamp = splits[2]
        links_data_arr.append([user_1, user_2])
    except:
        continue
print('convert data to dataframe')
links_data_arr = np.array(links_data_arr)
link_data_df = pd.DataFrame(links_data_arr, columns=link_columns)
link_data_df = link_data_df.drop_duplicates() # remove duplicated links
node_count = 1000 # reduce the number of nodes to 6000
print('reduce graph size to {} nodes'.format(node_count))
link_data_df = link_data_df[(link_data_df.user_1 < node_count) & (link_data_df.user_2 < node_count)]

# generate sub graph in which every node has degree greater or equal to 1 and with an edge selection probability of pe
def gen_subgraph(graph, pe=0.5):
    g1 = shuffle(graph.copy(),random_state=int(time.time()))
    g1_size = graph.shape[0] * pe
    degree = graph.groupby('user_1')['user_1'].count().to_frame('count').reset_index().append(
graph.groupby('user_2')['user_2'].count().to_frame('count').reset_index().rename(columns={'user_2':'user_1'}).reset_index()).groupby('user_1').sum()['count'].to_dict()

    g1 = g1.reset_index().assign(drop=pd.Series(np.zeros(g1.shape[0])))
    count = 0
    drop_dict = g1['drop'].to_dict()
    t1 = datetime.datetime.now()
    for i in range(0, g1.shape[0]):
        row = g1.iloc[i]
        user_1 = row['user_1']
        user_2 = row['user_2']
        if degree[user_1] >= 2 and degree[user_2] >= 2 and drop_dict[i] != 1:
            drop_dict[i] = 1
            degree[user_1] -= 1
            degree[user_2] -= 1
            count += 1
            if count % 10000 == 0:
                print(count, datetime.datetime.now() - t1)
                t1 = datetime.datetime.now()
            if count >= g1_size:
                break
    g1 = g1[~pd.Series(drop_dict).astype(bool)][['user_1', 'user_2']]
    node_values = list(set(g1.values.flatten()))
    node_alias = {}
    count = 0
    for val in node_values:
        node_alias[val] = count
        count += 1
    return g1, node_alias, node_values
def gen_subgraph_new(graph, pe=0.5):
    g = graph.copy()
    g = shuffle(g, random_state=int(time.time() * 100000 % 100000)).reset_index()[['user_1', 'user_2']]
    size = int(g.shape[0] * pe)
    print('size',size)
    start = random.randint(0, g.shape[0] - size)
    g = g.iloc[start:start + size]
    node_values = list(set(g.values.flatten()))
    node_alias = {}
    count = 0
    for val in node_values:
        node_alias[val] = count
        count += 1
    return g, node_alias, node_values
print('generate subgraph')
# g1, node_alias_1, node_values_1 = gen_subgraph(link_data_df, 0.9)
# g2, node_alias_2, node_values_2 = gen_subgraph(link_data_df, 0.9)
# g3, node_alias_3, node_values_3 = gen_subgraph(link_data_df, 1)
g1, node_alias_1, node_values_1 = gen_subgraph_new(link_data_df, 0.9)
g2, node_alias_2, node_values_2 = gen_subgraph_new(link_data_df, 0.9)
g3, node_alias_3, node_values_3 = gen_subgraph_new(link_data_df, 1)
# g1 = link_data_df.copy()
# g2 = link_data_df.copy()
print('start globalrank computation process')
for idx, (g, node_alias, node_values) in enumerate(zip((g1, g2, g3), (node_alias_1, node_alias_2, node_alias_3), (node_values_1, node_values_2, node_values_3))):
    g['weights'] = g['user_1'].copy()
    print('process matrix {}'.format(idx), g.shape)
    
    node_count = len(node_alias)
    data_matrix = np.zeros((node_count, node_count), dtype='float')
    edge_matrix = np.zeros((node_count, node_count), dtype=np.uint8)
    t1 = datetime.datetime.now()
    g1_values = g.values.copy().astype(float)
    print('get neighbors for every node')
    neighbors_map = {}
    for i in tqdm.trange(g1_values.shape[0]):
        row = g.iloc[i]
        ## neighbors for user_1
        user_1 = node_alias[row.user_1]
        user_2 = node_alias[row.user_2]
        user_1_neighbors = neighbors_map.get(user_1, set())
        user_1_neighbors.add(user_2)
        neighbors_map[user_1] = user_1_neighbors
        ## neighbors for user_2
        user_2_neighbors = neighbors_map.get(user_2, set())
        user_2_neighbors.add(user_1)
        neighbors_map[user_2] = user_2_neighbors
    print('calculate graph weights')
    for i in range(g1_values.shape[0]):
        row = g1_values[i]
        user_1 = node_alias[int(row[0])]
        user_2 = node_alias[int(row[1])]
        user_1_neighbors = neighbors_map[user_1]
        user_2_neighbors = neighbors_map[user_2]
        # w = (1 + len(user_1_neighbors.intersection(user_2_neighbors)))/(1 + len(user_1_neighbors.union(user_2_neighbors)))
        w = (len(user_1_neighbors.intersection(user_2_neighbors)))/(len(user_1_neighbors.union(user_2_neighbors)))
        row[2] = w
        # row[2] = 1
        
    # for i in range(g1_values.shape[0]):
    #     row = g1_values[i]
    #     row[0] = alias[row[0]]
    #     row[1] = alias[row[1]]

    for row in g1_values:
        user_1 = node_alias[int(row[0])]
        user_2 = node_alias[int(row[1])]
        data_matrix[user_1,user_2] = row[2]
        data_matrix[user_2,user_1] = row[2]
        edge_matrix[user_1,user_2] = 1
        edge_matrix[user_2,user_1] = 1

    print(datetime.datetime.now() - t1)
    print('calculate sums')
    total_weight_sum = data_matrix.sum()
    neighbor_count = data_matrix.sum(axis=1)

    total_neighbor_strengs = np.array([data_matrix[i][edge_matrix[i] > 0].sum() for i in tqdm.trange(node_count)])
    print((total_neighbor_strengs == 0).astype(int).sum())
    print('calculate forward and jump matrix')    

    print('forward matrix')
    ## old way
    forward_index_1 = []
    forward_index_2 = []
    forward_data = []
    forward_bin = []
    jump_index_1=[]
    jump_index_2=[]
    jump_data = []
    for k in tqdm.trange(g1_values.shape[0]):
        row = g1_values[k]
        i = node_alias[int(row[0])]
        j = node_alias[int(row[1])]
        forward_index_1.append(i)
        forward_index_2.append(j)
        # if total_neighbor_strengs[i] != 0:
            # forward_data.append(neighbor_count[j] * 1.0 / total_neighbor_strengs[i])
        if neighbor_count[i] > 0:
            forward_data.append(data_matrix[i, j]/ neighbor_count[i])
        else: forward_data.append(0)
                
        forward_index_1.append(j)
        forward_index_2.append(i)
        # if total_neighbor_strengs[j] != 0:
            # forward_data.append(neighbor_count[i] * 1.0 / total_neighbor_strengs[j])
        if neighbor_count[j] > 0:
            forward_data.append(data_matrix[i, j]/ neighbor_count[j])
        else: forward_data.append(0)
        
    for k in tqdm.trange(node_count):
        forward_index_1.append(k)
        forward_index_2.append(k)
        forward_data.append(0)
        # if total_neighbor_strengs[k] > 0:
        #     # u = node_values[k]
        #     u = k
        #     forward_index_1.append(u)
        #     forward_index_2.append(u)
        #     forward_data.append(neighbor_count[u] * 1.0 / total_neighbor_strengs[u])
    forward_matrix = sparse.csc_matrix((forward_data, (forward_index_1, forward_index_2)))
    forward_data_file = 'forward_data_file.tmp'  
    forward_data_memmap = np.memmap(forward_data_file, dtype='float32', mode='w+', shape=(node_count, node_count))
    forward_data_memmap[:] = forward_matrix.toarray()[:]
    forward_data_memmap.flush()

    print('jump matrix')
    jump_data_file = 'jump_data_file.tmp'
    jump_data_memmap = np.memmap(jump_data_file, dtype='float32', mode='w+', shape=(node_count, node_count))
    tmp = np.array(total_neighbor_strengs).sum()
    for  i in tqdm.trange(0, node_count):
        jump_data_memmap[i] = neighbor_count/tmp
    jump_data_memmap.flush()

    jump_data_file = 'jump_data_file.tmp'
    jump_data_memmap = np.memmap(jump_data_file, dtype='float32', mode='r+', shape=(node_count, node_count))
    forward_data_file = 'forward_data_file.tmp'  
    forward_data_memmap = np.memmap(forward_data_file, dtype='float32', mode='r+', shape=(node_count, node_count))

    # calculate T matrix
    print('calculating T matrix')
    T_file = 'T{}.tmp'.format(idx)
    pf = 0.85
    pj = 0.15
    T = np.memmap(T_file, dtype='float32', mode='w+', shape=(node_count, node_count))
    for i in tqdm.trange(node_count):
        T[i] = jump_data_memmap[i] * pj + forward_data_memmap[i] * pf
    ## write to memmap file
    T.flush()
    jump_data_memmap.flush()
    forward_data_memmap.flush()
    del jump_data_memmap
    del forward_data_memmap

    # GlobalRank computation
    t1 = datetime.datetime.now()
    R = np.array(neighbor_count)/total_weight_sum #* factor#/total_weight_sum
    dill.dump(R, open('R.pickle', 'wb'))
    R_dask = dask.array.from_array(R, chunks=500) # old = 1650
    threshold = 0.0001
    count = 0
    T_dask = dask.array.from_array(T, chunks=500)
    while True:
        t = datetime.datetime.now()
        count += 1
        R_dask = dask.array.from_array(R, chunks=500)
        R_dask_tmp = dask.array.dot(T_dask, R_dask)
        R_tmp = R_dask_tmp.compute()
        delta = R_tmp - R
        sigma = (delta**2).sum()#.compute()
        print(count, R_tmp, datetime.datetime.now() - t, sigma)
        R = R_tmp
        if sigma <= threshold:
            break
        print('new')
    print('consumed time:', datetime.datetime.now() - t1)
    print('save graph data')
    dill.dump(R, open('R_{}.pickle'.format(idx), 'wb'))
    ##convert neighbors_map to dict<int, list>
    for key in neighbors_map.keys():
        val = neighbors_map[key]
        val = list(val)
        neighbors_map[key] = val
    g_aliased_values = []
    for row in g1_values:
        u_1 = node_alias[row[0]]
        u_2 = node_alias[row[1]]
        g_aliased_values.append([u_1, u_2])
    g_new = pd.DataFrame(np.array(g_aliased_values), columns=['user_1', 'user_2'])
    graph = Graph(links_data=g_new, global_rank=R, link_strengs=neighbor_count, node_alias=node_alias, nodes=node_values, neighbors=neighbors_map)
    graph.save('graph_{}.pickle'.format(idx))
