from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize2 
from src.prompts.synthetic import SYNTHETIC
from statistics import mode
import warnings
import json
import time
from src.models.smo import SMO
import numpy as np
from sklearn.cluster import KMeans

def _post_process(result : str) -> dict:
    "Converts the output from the model to usable"
    json_start = result.find('{')
    json_end = result.rfind('}') + 1
    json_str = result[json_start:json_end]
    data = json.loads(json_str)
    best , rest = [], []

    for bst,rst in zip(data['better_examples'], data['poorer_examples']):
        best.append(bst['features'])
        rest.append(rst['features'])   

        return best,rest
def _markdown_to_rows(markdown):
    "Converts a Markdown table to a list of lists (rows)."
    
    lines = markdown.strip().split('\n')
    lines.pop(1)
    rows = [line.strip('|').split('|') for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    rows = rows[1:]

    best, rest = [], []

    for r in rows:
        if r[-1] == 'Best':
            best.append([float(val) if val.replace('.', '', 1).isdigit() else val for val in r[:-1]])
        elif r[-1] == 'Rest':
            rest.append([float(val) if val.replace('.', '', 1).isdigit() else val for val in r[:-1]])        
        
    return best,rest
def calculate_mean(example_list):
    "Calculate the mean of a list of rows"

    n_examples = len(example_list)
    n_features = len(example_list[0])
    
    mean = [0] * n_features
    for example in example_list:
        for i in range(n_features):
            mean[i] += example[i]
    
    mean = [x / n_examples for x in mean]
    return mean

def calculate_mean_or_mode(example_list, is_numeric):
    "Calculate the mean for numeric columns and mode for symbolic columns."
    n_features = len(example_list[0])
    
    result = [0] * n_features
    for i in range(n_features):
        column = [example[i] for example in example_list if example[i] != '?']  # Ignore missing values
        if len(column) == 0:  # If all values in the column are missing
            result[i] = '?'  # Handle case where all values are missing
            continue

        if is_numeric[i] == 1:  # If the column is numeric
            column = list(map(float, column))  # Convert to float
            result[i] = sum(column) / len(column)
        elif is_numeric[i] == 0:  # If the column is symbolic
            result[i] = mode(column) if column else '?'  # Use mode for symbolic data
    return result

def linear_extrapolation(i, done, scale=0.5):
    """Perform linear extrapolation to generate better and worse examples."""
    
    cut = int(.5 + len(done) ** 0.5)
    best = clone(i, done[:cut]).rows
    rest = clone(i, done[cut:]).rows

    dff = 0

    if(len(i.cols.names) != len(i.cols.x) + len(i.cols.y)):
        dff = len(i.cols.names)  - len(i.cols.x) - len(i.cols.y)

    best = [b[:len(i.cols.x) + dff] for b in best]
    rest = [r[:len(i.cols.x) + dff] for r in rest]

    # Check if each column is numeric or symbolic
    is_numeric = [0] * len(best[0])


    find = lambda col, i: next((c for c in i.cols.x if c.txt == col), None)

    names = i.cols.names[:len(i.cols.x) + dff]

    for _, col in enumerate(names):
        if col in [x.txt for x in i.cols.x]:
            c = find(col, i)
            is_numeric[_] = 1 if c.this == NUM else 0
        else:
            is_numeric[_] = 2

    # Calculate mean for numeric columns and mode for symbolic ones
    mean_best = calculate_mean_or_mode(best, is_numeric)
    mean_rest = calculate_mean_or_mode(rest, is_numeric)

    # Generate better and worse examples based on column type
    better_examples = [
        [
            float(b[idx] + scale * (mean_best[idx] - mean_rest[idx])) if is_numeric[idx] == 1 and b[idx] != '?' else mean_best[idx]
            for idx in range(len(b))
        ] for b in best
    ]
    worse_examples = [
        [
            float(r[idx] - scale * (mean_best[idx] - mean_rest[idx])) if is_numeric[idx] == 1 and r[idx] != '?' else mean_rest[idx]
            for idx in range(len(r))
        ] for r in rest
    ]


    return better_examples + worse_examples


def WARM_FEW_API(i: data, args, method = 'LLMExtra'):

    def calculate_mean(example_list):
        "Calculate the mean of a list of rows"

        n_examples = len(example_list)
        n_features = len(example_list[0])
        
        mean = [0] * n_features
        for example in example_list:
            for i in range(n_features):
                mean[i] += example[i]
        
        mean = [x / n_examples for x in mean]
        return mean

    def calculate_mean_or_mode(example_list, is_numeric):
        "Calculate the mean for numeric columns and mode for symbolic columns."
        n_features = len(example_list[0])
        
        result = [0] * n_features
        for i in range(n_features):
            column = [example[i] for example in example_list if example[i] != '?']  # Ignore missing values
            if len(column) == 0:  # If all values in the column are missing
                result[i] = '?'  # Handle case where all values are missing
                continue

            if is_numeric[i] == 1:  # If the column is numeric
                column = list(map(float, column))  # Convert to float
                result[i] = sum(column) / len(column)
            elif is_numeric[i] == 0:  # If the column is symbolic
                result[i] = mode(column) if column else '?'  # Use mode for symbolic data
        return result
    
    def _ranked(lst:rows, cur:row = None) -> rows:
        "Sort `lst` by distance to heaven. Called by `_smo1()`."
        lst = sorted(lst, key = lambda r:d2h(i,r))
        #callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur))
        return lst


    def _post_process(result : str) -> dict:
        "Converts the output from the model to usable"
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_str = result[json_start:json_end]
        data = json.loads(json_str)
        best , rest = [], []

        for bst,rst in zip(data['better_examples'], data['poorer_examples']):
            best.append(bst['features'])
            rest.append(rst['features'])   

        return best,rest

    def _markdown_to_rows(markdown):
        "Converts a Markdown table to a list of lists (rows)."
      
        lines = markdown.strip().split('\n')
        lines.pop(1)
        rows = [line.strip('|').split('|') for line in lines]
        rows = [[cell.strip() for cell in row] for row in rows]
        rows = rows[1:]

        best, rest = [], []

        for r in rows:
            if r[-1] == 'Best':
                best.append([float(val) if val.replace('.', '', 1).isdigit() else val for val in r[:-1]])
            elif r[-1] == 'Rest':
                rest.append([float(val) if val.replace('.', '', 1).isdigit() else val for val in r[:-1]])       
        
        return best,rest

    

    # def linear_extrapolation(done, scale=0.5):
    #     "Perform linear extrapolation to generate better and worse examples."

    #     cut = int(.5 + len(done) ** 0.5)
    #     best = clone(i, done[:cut]).rows
    #     rest = clone(i, done[cut:]).rows

    #     best = [b[:len(i.cols.x)] for b in best]
    #     rest = [r[:len(i.cols.x)] for r in rest]
    #     #print(best)
    #     #print(rest)

    #     mean_best = calculate_mean(best)
    #     mean_rest = calculate_mean(rest)
        
    #     difference = [mean_best[idx] - mean_rest[idx] for idx in range(len(mean_best))]
    #     better_examples = [[b[idx] + scale * difference[idx]  
    #     if i.cols.x[idx].this == NUM else b[idx]
    #     for idx in range(len(b))] for b in best]
    #     worse_examples = [[r[idx] - scale * difference[idx]  
    #     if i.cols.x[idx].this == NUM else r[idx]
    #     for idx in range(len(r))] for r in rest]
        
    #     return better_examples + worse_examples  


    def _synthesise(done: rows):
        "Synthesise better examples based on the initial random samples"
        #(model, dir) =  load_model(args).get_pipeline()
        model = load_model(args, name = 'gemini').get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        

        sythetic = SYNTHETIC(i, best, rest)
        messages = sythetic.get_template_correlation()
        #print(messages)

      
        result = model.invoke(messages).content
        #print(result)
        
        best, rest = _markdown_to_rows(result)
        #print(best, rest) 

        return best + rest 
        
    def n_examples(todo:rows, done:rows):
        "get the 4 start samples ready for active learning"
        results = _synthesise(done) if method == 'LLM' else linear_extrapolation(i,done)

        dff = 0

        if(len(i.cols.names) != len(i.cols.x) + len(i.cols.y)):
            dff = len(i.cols.names)  - len(i.cols.x) - len(i.cols.y)

        x_size = len(i.cols.x) + dff

        new_done = []
        for record in results:
            random.shuffle(todo)
            key = lambda r : dists(i, record, r[:x_size])
            top, *todo= sorted(todo, key=key, reverse=False)
            new_done.append(top)

        combined = _ranked(done+new_done)
        new_done = [combined[0],combined[1],combined[-1],combined[-2]]
        return done, new_done ,todo

    random.shuffle(i.rows) # remove any  bias from older runs
    return n_examples(i.rows[args.label:],_ranked(i.rows[:args.label]))

    
def WARM_FEW_API(i: data, args,  todo:rows, done:rows, method = 'LLMExtra', iscombined = True):
    def _ranked(lst:rows, cur:row = None) -> rows:
        "Sort `lst` by distance to heaven. Called by `_smo1()`."
        lst = sorted(lst, key = lambda r:d2h(i,r))
        #callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur))
        return lst

    def _synthesise(done: rows):
        "Synthesise better examples based on the initial random samples"
        #(model, dir) =  load_model(args).get_pipeline()
        model = load_model(args, name = args.llm).get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        # print("old best :", best)
        # print("old rest :", rest)
        sythetic = SYNTHETIC(i, best, rest)
        messages = sythetic.get_template_markdown()
        #print(messages)

      
        result = model.invoke(messages).content
        #print(result)
        
        best, rest = _markdown_to_rows(result)
        #print(best, rest) 

        # print("best:",best)
        # print("rest:", rest)
        # print("synsize", len(best[0]), len(rest[0]))
        return best + rest 
        
        
    def n_examples(todo:rows, done:rows):
        "get the 4 start samples ready for active learning"
        results = _synthesise(done) if method == 'LLM' else linear_extrapolation(i,done)

        dff = 0

        if(len(i.cols.names) != len(i.cols.x) + len(i.cols.y)):
            dff = len(i.cols.names)  - len(i.cols.x) - len(i.cols.y)

        x_size = len(i.cols.x) + dff
        # print("true size", x_size)
        new_done = []
        for record in results:
            random.shuffle(todo)
            key = lambda r : dists(i,record, r[:x_size] )
            top, *todo= sorted(todo, key=key, reverse=False)
            new_done.append(top)

        if iscombined:
            combined = _ranked(done+new_done)
            new_done = [combined[0],combined[1],combined[-1],combined[-2]]
        return done, new_done ,todo

    random.shuffle(i.rows) # remove any  bias from older runs
    return n_examples(todo,_ranked(done))

    


def warm_smo_plus(args, score = lambda B,R,I,N : B-R, method = 'LLM', start = 'Random'):
    def _ranked(lst:rows) -> rows:
        "Sort `lst` by distance to heaven. Called by `_smo1()`."
        lst = sorted(lst, key = lambda r:d2h(i,r))
        return lst

    
    if start == 'Random':
        i = DATA(csv(args.dataset))
        random.shuffle(i.rows)

        # points = random.choices(i.rows, k = 20)
        # sorted(points, key = lambda p : chebyshev(i,p))

        # best_points = points[:10]
        # rest_points = points[10:]

        # #done, new_done ,todo = WARM_FEW_API(i, args, method = method)
        # all = []
        # k = 0
        # while k  < 20:
        #     done = random.choices(best_points, k = 2) + random.choices(rest_points, k= 2)
        #     done = set((tuple(_) for _ in done))
        #     todo = set((tuple(_) for _ in i.rows)) - done


        #     done, new_done, todo = WARM_FEW_API(i, args,  list(todo), list(done), method = method)
        #     all += done + new_done
        #     k += 2


        all = []
        k = 0
        while k < 20:
            done = random.choices(i.rows, k=4)
            done = set((tuple(_) for _ in done))
            todo = set((tuple(_) for _ in i.rows)) - done

            done, new_done, todo = WARM_FEW_API(i, args,  list(todo), list(done), method = method)
            print("best of new done",chebyshev(i, new_done[0]))
            all += done + new_done
            k += 2


         

    elif start == 'Diversity':
        i = DATA(csv(args.dataset))
        random.shuffle(i.rows)

        d = dendogram(i,stop = 4)
        points = leafs(d,centroids=[])
        
        points = sorted(points, key = lambda p : chebyshev(i,p))

        # X = np.array(i.rows)
        

        # kmeans = KMeans(n_clusters=10,  n_init=10)
        # clusters = kmeans.fit_predict(X)

        # centroids = kmeans.cluster_centers_
        # points = centroid_rows = np.array([X[np.argmin(np.linalg.norm(X - centroid, axis=1))] for centroid in centroids])
        

        # points = sorted(points, key = lambda p : chebyshev(i,p))
        # for p in points:
        #     print(chebyshev(i,p))
        best_points = points[:len(points)//2]
        rest_points = points[len(points)//2:]

        


        #print("best of points", chebyshev(i,best_points[0]))

        #done, new_done ,todo = WARM_FEW_API(i, args, method = method)
        all = []
        k = 0
        while k  < 30:
            done = random.choices(best_points, k = 2) + random.choices(rest_points, k= 2)
            done = set((tuple(_) for _ in done))
            todo = set((tuple(_) for _ in i.rows)) - done


            done, new_done, todo = WARM_FEW_API(i, args,  list(todo), list(done), method = method)
            #print("best of new done",chebyshev(i, new_done[0]))


            all += done + new_done
            k += 2

        # i = DATA(csv(args.dataset))
        # random.shuffle(i.rows)
        # random.shuffle(i.rows)
        # best, rest, n = branch(i, i.rows, 15)


        # temp_best = random.choices(best, k=4)
        # temp_rest = random.choices(rest, k=4)

        # done = sorted(temp_best + temp_rest, key=lambda p: chebyshev(i, p))
        # print("best of intial", chebyshev(i,done[0]))
        # done = set(tuple(_) for _ in done)
        # todo = set(tuple(_) for _ in i.rows) - done
        # all = []

        # done = list(done)
        # todo = list(todo)

        # #print("best of points", chebyshev(i,best_points[0]))

        # #done, new_done ,todo = WARM_FEW_API(i, args, method = method)
        # all = []
        # k = 0
        # while k  < 28:
        #     # done = random.choices(best_points, k = 2) + random.choices(rest_points, k= 2)
        #     # done = set((tuple(_) for _ in done))
        #     # todo = set((tuple(_) for _ in i.rows)) - done


        #     done, new_done, todo = WARM_FEW_API(i, args,  list(todo), list(done), method = method)
        #     print("best of new done",chebyshev(i, new_done[0]))


        #     all += done + new_done
        #     k += 2
        
    elif start == 'Surrogate':
        besties = lambda B, R: B - R
        resties = lambda B, R: R - B

        i = DATA(csv(args.dataset))

        random.shuffle(i.rows)
        best, rest, n = branch(i, i.rows, 8)
        print("rest",len(rest))
        print("best", len(best))


        temp_best = random.choices(best, k=4)
        temp_rest = random.choices(rest, k=4)

        done = sorted(temp_best + temp_rest, key=lambda p: chebyshev(i, p))

        #print("best of intial", chebyshev(i,done[0]))
        done = set(tuple(_) for _ in done)
        todo = set(tuple(_) for _ in i.rows) - done
        all = []

        done = list(done)
        todo = list(todo)
        k = 0
        while k < 44:
            cut = int(.5 + len(done) ** 0.5)
            done = sorted(done, key=lambda p: chebyshev(i, p))
            best = clone(i, done[:cut])
            rest = clone(i, done[cut:])
            
            new_best = []
            score = besties
            key = lambda r: score(loglikes(best, r, len(done), 2),
                                loglikes(rest, r, len(done), 2))

            random.shuffle(todo)  # optimization: only sort a random subset of todo 
            todo = sorted(todo[:100], key=key, reverse=True) + todo[100:]
            new_best.append(todo[0])

            random.shuffle(todo)  # optimization: only sort a random subset of todo 
            todo = sorted(todo[:100], key=key, reverse=True) + todo[100:]
            new_best.append(todo[0])



            score = resties
            new_rest = []
            key = lambda r: score(loglikes(best, r, len(done), 2),
                                loglikes(rest, r, len(done), 2))

            random.shuffle(todo)  # optimization: only sort a random subset of todo 
            todo = sorted(todo[:100], key=key, reverse=True) + todo[100:]
            new_rest.append(todo[0])
            random.shuffle(todo)  # optimization: only sort a random subset of todo 
            todo = sorted(todo[:100], key=key, reverse=True) + todo[100:]
            new_rest.append(todo[0])

            # for _ in new_best + new_rest:
            #     print("chebyshev of points", chebyshev(i,_))
           

            done, new_done, todo = WARM_FEW_API(i, args, todo, sorted(new_best + new_rest, key = lambda p : chebyshev(i,p)), method=method, iscombined=False)
            # if new_done:
            #     print("extra best",chebyshev(i,new_done[0]))
            # done += new_done  # Add new_done rows to done
            # todo = [row for row in todo if row not in new_done]  # Remove new_done from todo

            all += new_done
            k += 2


    return _ranked(all)


    
    




def warm_smo(args, score=lambda B,R,I,N: B-R, method ='LLM'):
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
                           loglikes(rest, r, len(done), 2), len(done) - the.label, the.Last)


    
    random.shuffle(todo) # optimization: only sort a random subset of todo 
    return  sorted(todo[:the.any], key=key, reverse=True) + todo[the.any:]

    #return sorted(todo,key=key,reverse=True)

  def _smo1(todo:rows, done:rows, most: row) -> rows:
    "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
    
    for k in range(args.last - args.label):
      if len(todo) < 3: break
      top,*todo = _guess(todo, done)
      #print(d2h(i,top))
      most = top if not most or d2h(i,top) < d2h(i,most) else most
      done += [top]
      done = _ranked(done)
    return done,most

  # remove any  bias from older runs
  most = []
  i = DATA(csv(args.dataset))
  todo, done = i.rows[args.label:],_ranked(i.rows[:args.label])
  done, new_done ,todo = WARM_FEW_API(i, args,todo, done, method = method)
  for _ in new_done:
    print("row : ", _ , "chebys : ", chebyshev(i, _))
  results, most =  _smo1(todo, _ranked(new_done), most)
  
  return results, [i, [most], new_done[:2], new_done[2:], results]

    