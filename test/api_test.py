#python3
import requests
import json
import pandas as pd
import os
import time

DATA_ROOT = '/media/henan.wang/workspace/dataset/log_preprocess'
TRAIN = os.path.join(DATA_ROOT, 'train.txt')
EVAL = os.path.join(DATA_ROOT, 'test.txt')
TEST = os.path.join(DATA_ROOT, 'test.txt')
TRAIN_MUID = os.path.join(DATA_ROOT, 'train_muids_map.txt')
TRAIN_CONTENT_ID = os.path.join(DATA_ROOT, 'train_content_ids_map.txt')
MOVIE_TITLES = os.path.join(DATA_ROOT,'contentid_titles.txt')
MODEL_OUTPUT_DIR = 'model_save_funny'
INFER_OUTPUT = os.path.join(MODEL_OUTPUT_DIR, 'preds_api.txt')

end_point = 'http://127.0.0.1:5000/'
end_point_recommend_old = "http://127.0.0.1:5000/recommend_old"
end_point_recommend = "http://127.0.0.1:5000/recommend"
headers = {'Content-type':'application/json'}


def load_train_muid_and_content_id(muids_map_file, content_ids_map_file):
    muids_map = {}
    muids_inverted_map = {}
    with open(muids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            muids_map[params[0]] = int(params[1])
            muids_inverted_map[int(params[1])] = params[0]
    content_ids_map = {}
    content_ids_inverted_map = {}
    with open(content_ids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            content_ids_map[params[0]] = int(params[1])
            content_ids_inverted_map[int(params[1])] = params[0]
    return muids_map, muids_inverted_map, content_ids_map, content_ids_inverted_map


def infer_one_old(dict_query, fout=None):
    data=json.dumps(dict_query)
    #print('data:',data)
    res = requests.post(end_point_recommend_old, data=data, headers=headers)
    #print(res.ok)
    res_json = res.json()
    #print(json.dumps(res_json, indent=2))
    one_hit = 0
    one_not_hit = 0
    two_hit = 0
    two_not_hit = 0
    for k,v in dict_query.items():
        if k != 'user_id':
            infered_score = 1
            if float(res_json[str(k)]) >= 1.1:
                infered_score = 2
            if v == 1:
                if infered_score == 1:
                    one_hit += 1
                else:
                    one_not_hit += 1
            if v == 2:
                if infered_score == 2:
                    two_hit += 1
                else:
                    two_not_hit += 1
            s = str(dict_query['user_id'])+'\t'+str(k)+'\t'+str(res_json[str(k)])+'\t'+str(v)+'\n'
            if fout != None:
                fout.write(s)
            else:
                print(s)
                #pass
    return one_hit, one_not_hit, two_hit, two_not_hit


def infer_all_old(muids_map, content_ids_map):
    df = pd.read_csv(EVAL, names=['CustomerID','MovieID','Rating'], sep='\t')
    #print('df.shape:',df.shape)
    #print('df.head():',df.head())
    df2 = pd.read_csv(TEST, names=['CustomerID','MovieID','Rating'], sep='\t')
    print('df2.shape:',df2.shape)
    #print('df2.head():',df2.head())
    titles = pd.read_csv(MOVIE_TITLES, names=['MovieID','ContentID','Year','Title'], encoding = "latin")
    #print('titles.head():',titles.head())
    all_one_hit = 0
    all_one_not_hit = 0
    all_two_hit = 0
    all_two_not_hit = 0
    user_ids = {}
    fout = open(INFER_OUTPUT, 'w')
    start = time.time()
    for index, row in df2.iterrows():
        user_id = int(row['CustomerID'])
        if user_id not in user_ids:
            user_ids[user_id] = 1
        else:
            continue
        if len(user_ids) % 100 == 0:
            print('Step:',len(user_ids))
        #print('user_id:',user_id)
        target = df2[df2['CustomerID'] == user_id]
        #print('target:',target)
        df_customer = pd.merge(target, titles, on='MovieID', how='left', suffixes=('_',''))
        #print('df_customer:',df_customer)
        df_customer.drop(['ContentID','Year','Title'], axis=1, inplace=True)
        df_query = df_customer.drop(['CustomerID'], axis=1).set_index('MovieID')
        dict_query = df_query.to_dict()['Rating']
        dict_query['user_id'] = user_id # add muid
        
        #print('dict_query:', dict_query)

        one_hit, one_not_hit, two_hit, two_not_hit = infer_one_old(dict_query, fout)

        all_one_hit += one_hit
        all_one_not_hit += one_not_hit
        all_two_hit += two_hit
        all_two_not_hit += two_not_hit
        #one_hit = one_hit/float(one_hit+one_not_hit)
        #two_hit = two_hit/float(two_hit+two_not_hit)
        #print('Precision:',(one_hit+two_hit)/float(one_hit+one_not_hit+two_hit+two_not_hit))

    end = time.time()                
    print(all_one_hit, all_one_not_hit)
    print(all_two_not_hit, all_two_hit)
    print('user_ids:',len(user_ids))
    print('Final Precision:',(all_one_hit+all_two_hit)/float(all_one_hit+all_one_not_hit+all_two_hit+all_two_not_hit))
    print('Final Time:',end-start,'s')
    print('Avg Time:',(end-start)*1000/float(len(user_ids)), 'ms')
    fout.close()

    
def infer_one(query):
    data=json.dumps(query)
    #print('data:',data)
    res = requests.post(end_point_recommend, data=data, headers=headers)
    #print(res.ok)
    res_json = res.json()
    #print(json.dumps(res_json, indent=2))
    return res_json


def infer_all(muids_map, muids_inverted_map, content_ids_map, content_ids_inverted_map):
    df = pd.read_csv(EVAL, names=['CustomerID','MovieID','Rating'], sep='\t')
    #print('df.shape:',df.shape)
    #print('df.head():',df.head())
    df2 = pd.read_csv(TEST, names=['CustomerID','MovieID','Rating'], sep='\t')
    print('df2.shape:',df2.shape)
    #print('df2.head():',df2.head())
    titles = pd.read_csv(MOVIE_TITLES, names=['MovieID','ContentID','Year','Title'], encoding = "latin")
    #print('titles.head():',titles.head())
    all_one_hit = 0
    all_one_not_hit = 0
    all_two_hit = 0
    all_two_not_hit = 0
    user_ids = {}
    fout = open(INFER_OUTPUT, 'w')
    start = time.time()
    for index, row in df2.iterrows():
        user_id = int(row['CustomerID'])
        if user_id not in user_ids:
            user_ids[user_id] = 1
        else:
            continue
        if len(user_ids) % 100 == 0:
            print('Step:',len(user_ids))
        #print('user_id:',user_id)
        target = df2[df2['CustomerID'] == user_id]
        #print('target:',target)
        df_customer = pd.merge(target, titles, on='MovieID', how='left', suffixes=('_',''))
        #print('df_customer:',df_customer)
        df_customer.drop(['ContentID','Year','Title'], axis=1, inplace=True)
        df_query = df_customer.drop(['CustomerID'], axis=1).set_index('MovieID')
        dict_query = df_query.to_dict()['Rating']
        
        query = {}
        query['muid'] = muids_inverted_map[user_id] # add muid
        query['content_ids'] = []
        for k in dict_query.keys():
            query['content_ids'].append(content_ids_inverted_map[k])
        
        #print('query:', query)

        res_json = infer_one(query)
        
        one_hit = 0
        one_not_hit = 0
        two_hit = 0
        two_not_hit = 0
        for i in range(len(query['content_ids'])):
            k = content_ids_map[query['content_ids'][i]]
            v = dict_query[k]
            infered_score = 1
            if float(res_json['ratings'][i]) >= res_json['threshold']:
                infered_score = 2
            if v == 1:
                if infered_score == 1:
                    one_hit += 1
                else:
                    one_not_hit += 1
            if v == 2:
                if infered_score == 2:
                    two_hit += 1
                else:
                    two_not_hit += 1
            s = str(user_id)+'\t'+str(k)+'\t'+str(res_json['ratings'][i])+'\t'+str(v)+'\n'
            #print(s)
            fout.write(s)

        all_one_hit += one_hit
        all_one_not_hit += one_not_hit
        all_two_hit += two_hit
        all_two_not_hit += two_not_hit
        #one_hit = one_hit/float(one_hit+one_not_hit)
        #two_hit = two_hit/float(two_hit+two_not_hit)
        #print('Precision:',(one_hit+two_hit)/float(one_hit+one_not_hit+two_hit+two_not_hit))

    end = time.time()                
    print(all_one_hit, all_one_not_hit)
    print(all_two_not_hit, all_two_hit)
    print('user_ids:',len(user_ids))
    print('Final Precision:',(all_one_hit+all_two_hit)/float(all_one_hit+all_one_not_hit+all_two_hit+all_two_not_hit))
    print('Final Time:',end-start,'s')
    print('Avg Time:',(end-start)*1000/float(len(user_ids)), 'ms')
    fout.close()
    
if __name__=='__main__':
#    muids_map, muids_inverted_map, content_ids_map, content_ids_inverted_map = load_train_muid_and_content_id(TRAIN_MUID, TRAIN_CONTENT_ID)
#    infer_all(muids_map, muids_inverted_map, content_ids_map, content_ids_inverted_map)
    
    query = {}
    query['muid'] = 'dece108c-fb25-428f-800f-2ddb4dcbe8fc' # 12288
    #query['muid'] = 'abcd' # not existed muid
    #query['content_ids'] = ['fdc603c20c70fcec20ffe5e7692e4049'] # 9
    query['content_ids'] = ['fdc603c20c70fcec20ffe5e7692e4049', 'efgh'] # 9 and not existed content_id
    #query['content_ids'] = 'fdc603c20c70fcec20ffe5e7692e4049' # ErrorNo 400
    res_json = infer_one(query)
    print(res_json)
    
    #infer_all_old()
    
#    dict_query = {9:1,'user_id':12288}
#    one_hit, one_not_hit, two_hit, two_not_hit = infer_one_old(dict_query)
#    print('Final Precision:',(one_hit+two_hit)/float(one_hit+one_not_hit+two_hit+two_not_hit))

    
