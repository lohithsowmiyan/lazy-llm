from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
import warnings
import time

def ZERO(args, save_results = False, k = 100, model = None):
    warnings.filterwarnings("ignore")
    loaded_here = False
    if model == None:
        loaded_here = True
        model =  load_model(args)
        pipe = model.get_pipeline()
    #random.seed(args.seed)
    records = []
    def _tile(lst, curd2h, budget):
        num = adds(NUM(),lst)
        n=100
        print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
        sd=int(num.sd*n/2)
        mu=int(num.mu*n)
        print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")
        record = o(the = "result", N = len(lst), Mu = format(num.mu,".3f"), Sd = format(num.sd, ".3f"), Var = " "*(mu-sd) + "-"*sd + "+"*sd, Curd2h = format(curd2h, ".3f"), Budget = budget)
        records.append(record)

    def learner(i:data, callBack=lambda x,y,z:x):
        """
        
        """
        def _ranked(lst:rows, cur:row = None, budget:int = 0) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur), budget)
            # print(d2h of the best row)
            return lst

        def llm_guesser(current: row, done: rows) -> row:
            cut = int(.5 + len(done) ** 0.5)
            best = clone(i, done[:cut]).rows
            rest = clone(i, done[cut:]).rows
            best = [b[:len(i.cols.x)] for b in best]
            rest = [r[:len(i.cols.x)] for r in rest]
            messages = load_prompt(args.dataset).getZeroShot(current[:len(i.cols.x)], cols = i.cols.x)
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=256,  do_sample=True, temperature=0.5, top_p=0.9) #eos_token_id=terminators,
            print(outputs[0]['generated_text']) if args.intermediate else None
            if "best" in outputs[0]['generated_text'][len(prompt):].lower(): return current
            return None
            
        
        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            count = 0
            for k in todo:
                count += 1
                if len(done) >= args.last: break
                top = llm_guesser(k, done)
                if(top == None): continue
                btw(d2h(i,top))
                done += [top]
                done = _ranked(done, top, count)
            return done

        i_sampled = random.choices(i.rows, k = k)
        return _smo1(i_sampled[args.label:], _ranked(i_sampled[:args.label]))

    
    if(save_results):
        results = learner(DATA(csv(args.dataset)), _tile)
        save_results_txt(model = args.model + "_" + args.llm, dataset = args.dataset, records =  records)
        time.sleep(5)
        visualize(dataset = args.dataset[args.dataset.rfind('/')+1:-4], show = 'All', save_fig= True, display = False)
    else: results = learner(DATA(csv(args.dataset)))
    if loaded_here:
        model.unload_model()
    return results