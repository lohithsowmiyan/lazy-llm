from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.models import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt



def vanilla(args):
    model = load_model(args)

    def _tile(lst):
        num = adds(NUM(),lst)
        n=100
        print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
        sd=int(num.sd*n/2)
        mu=int(num.mu*n)
        print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")

    def learner():
        def _ranked(lst:rows) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            _tile([d2h(i,r) for r in lst])
            return lst

        
        
    prompt = load_prompt[args.dataset].getZeroShot()

if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    vanilla(args)

