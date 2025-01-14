from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
import warnings
import time

def SMO(args):
     
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

    def smo(i:data, score=lambda B,R: B-R, callBack=lambda x,y,z:x ):
        "Sequential model optimization."
        def _ranked(lst:rows, cur:row = None, budget:int = 0) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur), budget)
            return lst

        def _guess(todo:rows, done:rows) -> rows:
            "Divide `done` into `best`,`rest`. Use those to guess the order of unlabelled rows. Called by `_smo1()`."
            cut  = int(.5 + len(done) ** 0.5)
            best = clone(i, done[:cut])
            rest = clone(i, done[cut:])
            key  = lambda r: score(loglikes(best, r, len(done), 2),
                           loglikes(rest, r, len(done), 2))
            
            random.shuffle(todo) # optimization: only sort a random subset of todo 
            todo= sorted(todo[:100], key=key, reverse=True) + todo[100:]
            return  todo[:int(len(todo) *1)]
             #return sorted(todo,key=key,reverse=True)

        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`,and repeat"
            count = 0
            for k in range(the.Last - the.label):
                count += 1 
                if len(todo) < 3: break
                top,*todo = _guess(todo, done)
                done += [top]
                done = _ranked(done, top, count)
            return done

        random.shuffle(i.rows) # remove any  bias from older runs
        return _smo1(i.rows[4:], _ranked(i.rows[:4]))

    return smo(DATA(csv(args.dataset)))
    # save_results_txt(model = args.model, dataset = args.dataset, records =  records)
    # time.sleep(5)
    # visualize(dataset = args.dataset[args.dataset.rfind('/')+1:-4], show = 'All', save_fig= True, display = False)
    # return True
