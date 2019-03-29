import dill
import gzip
import numpy as np
import pandas as pd
import tqdm

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
        links_data_arr.append([user_1, user_2])
    except:
        continue
print('convert data to dataframe')
links_data_arr = np.array(links_data_arr)
graph = pd.DataFrame(links_data_arr, columns=link_columns)
graph = graph.drop_duplicates() # remove duplicated links

degree = graph.groupby('user_1')['user_1'].count().to_frame('count').reset_index().append(
graph.groupby('user_2')['user_2'].count().to_frame('count').reset_index().rename(columns={'user_2':'user_1'}).reset_index()).groupby('user_1').sum()['count']
degree_sorted_nodes = list(degree.sort_values().index)
with open('degree.txt', 'w') as f:
    f.write('\n'.join([str(a) for a in degree_sorted_nodes]))