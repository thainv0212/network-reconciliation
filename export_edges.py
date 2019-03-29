import dill
import pandas as pd
from sys import argv
if len(argv) < 3:
    print('python export_edges.py file_name output_format')
    exit(0)
g = dill.load(open(argv[1], 'rb'))
idx = argv[1].rfind('.')
if argv[2] == 'txt':
    text = '\n'.join([' '.join([str(i) for i in l]) for l in g.links_data.values])
    with open(argv[1][:idx] + '_edges.txt', 'w') as f:
        f.write(text)
else:
    df = pd.DataFrame(g.links_data.values, columns=['node_1', 'node_2'])
    df.to_csv(argv[1][:idx] + '_edges.csv', index=False)