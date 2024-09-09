from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize2 
from src.prompts.synthetic import SYNTHETIC
import warnings
import json
import time

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
            if r[-1] == 'Best': best.append([float(val) for val in r[:-1]])
            elif r[-1] == 'Rest': rest.append([float(val) for val in r[:-1]])        
        
        return best,rest

    def linear_extrapolation(done, scale=0.5):
        "Perform linear extrapolation to generate better and worse examples."

        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        best = best[:len(i.cols.x)]
        rest = rest[:len(i.cols.x)]

        mean_best = calculate_mean(best)
        mean_rest = calculate_mean(rest)
        
        difference = [mean_best[idx] - mean_rest[idx] for idx in range(len(mean_best))]
        better_examples = [[b[idx] + scale * difference[idx]  for idx in range(len(b))] for b in best]
        worse_examples = [[r[idx] - scale * difference[idx]  for idx in range(len(r))] for r in rest]
        
        return better_examples + worse_examples 


    def _synthesise(done: rows):
        "Synthesise better examples based on the initial random samples"
        #(model, dir) =  load_model(args).get_pipeline()
        model = load_model(args, name = 'gemini').get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        sythetic = SYNTHETIC(i, best, rest)
        messages = sythetic.get_template_correlation()
      
        result = model.invoke(messages).content
        
        best, rest = _markdown_to_rows(result)
        #print(best, rest)

        return best + rest 
        
    def n_examples(todo:rows, done:rows):
        "get the 4 start samples ready for active learning"
        results = _synthesise(done) if method == 'LLM' else linear_extrapolation(done)

        x_size = len(i.cols.x)
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
  done, new_done ,todo = WARM_FEW_API(i, args, method = method)
  results, most =  _smo1(todo, _ranked(new_done), most)
  if(False):
        time.sleep(5)
        visualize2(
        i, 
        [most], 
        new_done[:2], 
        new_done[2:], 
        results, 
        policy='warm_explore'
        )
  return results

    