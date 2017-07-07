import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names= header)

n_users= df.user_id.unique().shape[0]
n_items= df.item_id.unique().shape[0]
print('Number of users:' + str(n_users)+  '| Numbers of movies =' + str(n_items))

sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print ('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')