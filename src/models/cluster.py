from src.models.warm_start import WARM_FEW_API 
import os
import tempfile
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from src.utils.ezr import *
from math import exp

def density(args, method ='LLM'):

  scores = {
    'explore' :  lambda B,R: (exp(B) + exp(R))/abs(exp(B) - exp(R) + 1E-30),
    'exploit' : lambda B,R : B
  }

  score = scores['explore']

  "Sequential model optimization."
  def _ranked(lst:rows) -> rows:
    "Sort `lst` by distance to heaven. Called by `_smo1()`."
    lst = sorted(lst, key = lambda r:d2h(i,r))
    return lst

  def _guess(todo:rows, done:rows) -> rows:
    "Divide `done` into `best`,`rest`. Use those to guess the order of unlabelled rows. Called by `_smo1()`."
    cut  = int(.5 + len(done) ** the.N)
    best = clone(i, done[:cut])
    rest = clone(i, done[cut:])
    key  = lambda r: score(loglikes(best, r, len(done), 2),
                           loglikes(rest, r, len(done), 2))


    
    random.shuffle(todo) # optimization: only sort a random subset of todo 
    return  sorted(todo[:the.any], key=key, reverse=True) + todo[the.any:]

    #return sorted(todo,key=key,reverse=True)

  def _smo1(todo:rows, done:rows) -> rows:
    "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
    for k in range(args.last - args.label):
      if len(todo) < 3: break
      top,*todo = _guess(todo, done)
      #print(d2h(i,top))
      #most = top if not most or d2h(i,top) < d2h(i,most) else most
      done += [top]
      done = _ranked(done)
    return done#,most

  # remove any  bias from older runs
  most = []
  #i = DATA(csv(args.dataset))
  #done, new_done ,todo = WARM_FEW_API(i, args, method = method)

  #Clustering into dense and sparse regions
  data = pd.read_csv(args.dataset)
  scaler = StandardScaler()
  numeric_data = data.select_dtypes(include=['float64', 'int64'])
  normalized_data = scaler.fit_transform(numeric_data)
  dbscan = DBSCAN(eps=2, min_samples=5)  # Adjust parameters as needed
  labels = dbscan.fit_predict(normalized_data)

  data['cluster'] = labels
  dense_data = data[data['cluster'] != -1].drop(columns=['cluster'])  # Dense regions
  sparse_data = data[data['cluster'] == -1].drop(columns=['cluster'])

  with tempfile.TemporaryDirectory() as temp_dir:
    # Save the datasets to the temporary folder
    dense_path = os.path.join(temp_dir, "dense_regions.csv")
    sparse_path = os.path.join(temp_dir, "sparse_regions.csv")
    
    dense_data.to_csv(dense_path, index=False)
    sparse_data.to_csv(sparse_path, index=False)
    args.last = 30
    i = DATA(csv(dense_path))

    done, new_done ,todo = WARM_FEW_API(i, args, method = method)
    final_results = _smo1(todo, _ranked(new_done))


    #i_sparse = DATA(csv(sparse_path))

    #set1 = set(map(tuple, new_done))
    #set2 = set(map(tuple, i_sparse.rows))

    # Find rows in list2 that are not in list1
    #difference = set2 - set1

    # Convert the result back to a list of lists
    #sparse_todo = list(map(list, difference))

    # args.last = 24
    # results =  _smo1(sparse_todo, _ranked(new_done))

    # set1 = set(map(tuple, results))
    # set2 = set(map(tuple, i_dense.rows))

    # difference = set2 - set1

    # dense_todo = list(map(list, difference))

    # args.last = 24
    # scores = ['exploit']
    # final_results = _smo1(dense_todo, _ranked(results))


  return final_results

    



  # new active learning
  
  
