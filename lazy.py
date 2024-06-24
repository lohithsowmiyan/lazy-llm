from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.models import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
import time


def vanilla(args):
    model = load_model(args).get_model()

    def _tile(lst):
        num = adds(NUM(),lst)
        n=100
        print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
        sd=int(num.sd*n/2)
        mu=int(num.mu*n)
        print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")

    def learner(i:data, callBack=lambda x:x):
        def _ranked(lst:rows) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            _tile([d2h(i,r) for r in lst])
            return lst

        def llm_guesser(current: rows, done: rows) -> rows:
            cut = int(.5 + len(done) ** 0.5)
            best = clone(i, done[:cut]).rows
            rest = clone(i, done[cut:]).rows
            chain = load_prompt[args.dataset].getZeroShot(best, rest) | model
            output = chain.invoke({"input": f"{current[:5]}"})
            if "rest" in output.content.lower() or "outlier" in output.content.lower(): return None
            return current
            
        
        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            for k in todo:
                if len(done) > 30: break
                top = llm_guesser(k, done)
                time.sleep(5)
                if(top == None): continue
                done += [top]
                done = _ranked(done)
            return done

        random.shuffle(i.rows)
        return _smo1(i.rows[4:], _ranked(i.rows[:4]))

    learner(DATA(csv(args.dataset)), _tile)

if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    vanilla(args)

