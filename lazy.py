from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.models import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
import warnings
import time


# def vanilla(args):
#     warnings.filterwarnings("ignore")
#     model = load_model(args).get_model_with_quantization() if args.quantization else load_model(args).get_model()

#     def _tile(lst):
#         num = adds(NUM(),lst)
#         n=100
#         print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
#         sd=int(num.sd*n/2)
#         mu=int(num.mu*n)
#         print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")

#     def learner(i:data, callBack=lambda x:x):
#         """
        
#         """
#         def _ranked(lst:rows) -> rows:
#             "Sort `lst` by distance to heaven. Called by `_smo1()`."
#             lst = sorted(lst, key = lambda r:d2h(i,r))
#             _tile([d2h(i,r) for r in lst])
#             # print(d2h of the best row)
#             return lst

#         def llm_guesser(current: row, done: rows) -> row:
#             cut = int(.5 + len(done) ** 0.5)
#             best = clone(i, done[:cut]).rows
#             rest = clone(i, done[cut:]).rows
#             prompt = load_prompt[args.dataset].getZeroShot(best, rest)
#             chain = prompt | model
#             output = chain.invoke({"input": f"{current[:len(i.cols.x)]}"}) # data.cols.x
#             print(output)
#             if "best" in output.lower(): return current
#             return None
            
        
#         def _smo1(todo:rows, done:rows) -> rows:
#             "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
#             for k in todo:
#                 if len(done) > 30: break
#                 top = llm_guesser(k, done)
#                 if(top == None): continue
#                 print(d2h(i,top))
#                 done += [top]
#                 done = _ranked(done)
#             return done

#         random.shuffle(i.rows)
#         return _smo1(i.rows[4:], _ranked(i.rows[:4]))

#     learner(DATA(csv(args.dataset)), _tile)

def vanilla1(args):
    warnings.filterwarnings("ignore")
    model =  load_model(args).get_pipeline()
    random.seed(args.seed)
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

    def learner(i:data, callBack=lambda x:x):
        """
        
        """
        def _ranked(lst:rows, cur:row = None, budget:int = 0) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            _tile([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur), budget)
            # print(d2h of the best row)
            return lst

        def llm_guesser(current: row, done: rows) -> row:
            cut = int(.5 + len(done) ** 0.5)
            best = clone(i, done[:cut]).rows
            rest = clone(i, done[cut:]).rows
            best = [b[:len(i.cols.x)] for b in best]
            rest = [r[:len(i.cols.x)] for r in rest]
            messages = load_prompt(args.dataset).getTemplate(best, rest, current[:len(i.cols.x)], cols = i.cols.x)
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model.model.config.pad_token_id = model.model.config.eos_token_id
            outputs = model(prompt, max_new_tokens=256,  do_sample=True, temperature=0.5, top_p=0.9) #eos_token_id=terminators,
            print(outputs[0]['generated_text']) if args.intermediate else None
            if "best" in outputs[0]['generated_text'][len(prompt):].lower(): return current
            return None
            
        
        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            count = 0
            for k in todo:
                count += 1
                if len(done) >= 30: break
                top = llm_guesser(k, done)
                if(top == None): continue
                print(d2h(i,top))
                done += [top]
                done = _ranked(done, top, count)
            return done

        random.shuffle(i.rows)
        return _smo1(i.rows[4:], _ranked(i.rows[:4]))

    learner(DATA(csv(args.dataset)), _tile)
    return save_results_txt(model = args.model + "_" + args.llm, dataset = args.dataset, records =  records)
    time.sleep(3)
    visualize(dataset = args.dataset.rfind('/'), show = 'All', save_fig= True, display = False)

def SMO(args):
    random.seed(args.seed)
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

    smo(DATA(csv(args.dataset)),callBack = _tile)
    save_results_txt(model = args.model, dataset = args.dataset, records =  records)
    time.sleep(3)
    visualize(dataset = args.dataset.rfind('/'), show = 'All', save_fig= True, display = False)
    return True

        
if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    if(args.model == 'vanilla'):
        vanilla1(args)
    if(args.model == 'smo'):
        SMO(args)
    #print(save_results())



