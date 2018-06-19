#python3
import requests
import json
import pandas as pd
import os

DATA_ROOT = '/media/henan.wang/workspace/dataset/log_preprocess'
TRAIN = os.path.join(DATA_ROOT, 'train.txt')
EVAL = os.path.join(DATA_ROOT, 'test.txt')
TEST = os.path.join(DATA_ROOT, 'test.txt')
MOVIE_TITLES = os.path.join(DATA_ROOT,'contentid_titles.txt')


#dict_query = {1:32768,1:2307,2:11598}

df = pd.read_csv(EVAL, names=['CustomerID','MovieID','Rating'], sep='\t')
#print('df.shape:',df.shape)
#print('df.head():',df.head())
df2 = pd.read_csv(TEST, names=['CustomerID','MovieID','Rating'], sep='\t')
#print('df2.shape:',df2.shape)
#print('df2.head():',df2.head())
titles = pd.read_csv(MOVIE_TITLES, names=['MovieID','ContentID','Year','Title'], encoding = "latin")
#print('titles.head():',titles.head())
target = df2[df2['CustomerID'] == 8286]
#print('target:',target)
df_customer = pd.merge(target, titles, on='MovieID', how='left', suffixes=('_',''))
#print('df_customer:',df_customer)
df_customer.drop(['ContentID','Year','Title'], axis=1, inplace=True)
df_query = df_customer.drop(['CustomerID'], axis=1).set_index('MovieID')
dict_query = df_query.to_dict()['Rating']
dict_query['user_id'] = 1 # add user_id
#print('dict_query:', dict_query)

end_point = 'http://127.0.0.1:5000/'
end_point_recommend = "http://127.0.0.1:5000/recommend"
headers = {'Content-type':'application/json'}
data=json.dumps(dict_query)
print('data:',data)
res = requests.post(end_point_recommend, data=data, headers=headers)
print(res.ok)
print(json.dumps(res.json(), indent=2))
