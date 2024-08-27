from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
from src.language.llms import unload_model
from src.prompts.synthetic import SYNTHETIC
import warnings
import time

def WARM_FEW(args):

    
    def _ranked(lst:rows, cur:row = None) -> rows:
        "Sort `lst` by distance to heaven. Called by `_smo1()`."
        lst = sorted(lst, key = lambda r:d2h(i,r))
        #callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur))
        return lst


    def _synthesise(i : data, done: rows):
        "Synthesise better examples based on the initial random samples"
        (model, dir) =  load_model(args).get_pipeline()
        cut = int(.5 + len(done) ** 0.5)
        best = clone(i, done[:cut]).rows
        rest = clone(i, done[cut:]).rows

        sythetic = SYNTHETIC(i, best, rest)
        messages = sythetic.get_template()

        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model(prompt, max_new_tokens=512,  do_sample=True, temperature=0.7, top_p=0.9) #eos_token_id=terminators,
        print(outputs[0]['generated_text'])

        unload_model(model, dir)



    def n_examples(i: data, todo:rows, done:rows) -> rows:
        "Guess the `top`  unlabeled row, add that to `done`, resort `done`,and repeat"
        results = _synthesise(i, done)
        #print(i)

        
        

    i = DATA(csv(args.dataset))
    random.shuffle(i.rows) # remove any  bias from older runs
    return n_examples(i, i.rows[args.label:],_ranked(i.rows[:args.label]))