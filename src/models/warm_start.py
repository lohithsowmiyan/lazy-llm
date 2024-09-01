from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
from src.prompts.synthetic import SYNTHETIC
import warnings
import json
import time

def WARM_FEW_API(i: data, args):
    
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
        """
        Converts a Markdown table to a list of lists (rows).

        :param markdown: String containing the Markdown table.
        :return: List of lists where each sublist represents a row in the table.
        """
        lines = markdown.strip().split('\n')
        lines.pop(1)
        rows = [line.strip('|').split('|') for line in lines]
        rows = [[cell.strip() for cell in row] for row in rows]
        rows = rows[1:]

        best, rest = [], []

        for r in rows:
            if r[-1] == 'Best': best.append([int(val) for val in r[:-1]])
            elif r[-1] == 'Rest': rest.append([int(val) for val in r[:-1]])        
        
        return best,rest

    def _synthesise(done: rows):
        "Synthesise better examples based on the initial random samples"
        #(model, dir) =  load_model(args).get_pipeline()
        model = load_model(args, name = 'gemini').get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        sythetic = SYNTHETIC(i, best, rest)
        messages = sythetic.get_template_markdown()

        
        result = model.invoke(messages).content
        
        best, rest = _markdown_to_rows(result)
        #print(best, rest)

        

        return best + rest 
       
        #unload_model(model, dir)

        
    def n_examples(todo:rows, done:rows):
        results = _synthesise(done)

        x_size = len(i.cols.x)
        new_done = []
        for record in results:
            random.shuffle(todo)
            key = lambda r : dists(i, record, r[:x_size])
            top, *todo= sorted(todo, key=key, reverse=False)
            new_done.append(top)

        #print(results)
        #print(new_done)
        return done, new_done, todo

    random.shuffle(i.rows) # remove any  bias from older runs
    return n_examples(i.rows[args.label:],_ranked(i.rows[:args.label]))

def WARM_FEW_L(i:data, args):
     
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
        return json.loads(json_str)



    def _synthesise(done: rows):
        "Synthesise better examples based on the initial random samples"
        #(model, dir) =  load_model(args).get_pipeline()
        model = load_model(args)
        pipe = model.get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        sythetic = SYNTHETIC(i, best, rest)

        messages = sythetic.get_template()
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512,  do_sample=True, temperature=0.7, top_p=0.9) #eos_token_id=terminators,
        print(outputs)
        outputs = outputs[0]['generated_text'][len(prompt):]
        #result = model.invoke(messages).content
        #print(result)
        
        data = _post_process(outputs)

        best , rest = [], []

        for bst,rst in zip(data['better_examples'], data['poorer_examples']):
            best.append(bst['features'])
            rest.append(rst['features'])

        return best + rest

        #unload_model(model, dir)


    def n_examples(todo:rows, done:rows):
        results = _synthesise(done)

        x_size = len(i.cols.x)
        new_done = []
        for record in results:
            random.shuffle(todo)
            key = lambda r : dists(i, record, r[:x_size])
            top, *todo= sorted(todo, key=key, reverse=False)
            new_done.append(top)
        
        
        #print(results)
        #print(new_done)
        return done, new_done, todo

    random.shuffle(i.rows) # remove any  bias from older runs
    return n_examples(i.rows[args.label:],_ranked(i.rows[:args.label]))



def warm_smo(args, score=lambda B,R,I,N: B-R, callBack=lambda x:x):
  "Sequential model optimization."
  def _ranked(lst:rows) -> rows:
    "Sort `lst` by distance to heaven. Called by `_smo1()`."
    lst = sorted(lst, key = lambda r:d2h(i,r))
    callBack([d2h(i,r) for r in lst])
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

  def _smo1(todo:rows, done:rows) -> rows:
    "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
    for k in range(args.last - args.label):
      if len(todo) < 3: break
      top,*todo = _guess(todo, done)
      #print(d2h(i,top))
      done += [top]
      done = _ranked(done)
    return done

  # remove any  bias from older runs
  i = DATA(csv(args.dataset))
  done, new_done ,todo = WARM_FEW_API(i, args)
  return _smo1(todo, _ranked(new_done))

    