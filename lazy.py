from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize,visualize2 
import warnings
import time
from src.models.few import FEW
from src.models.smo import SMO
from src.models.zero import ZERO
from src.models.warm_start import warm_smo,warm_smo_plus
from src.models.cluster import density
from src.models.gpm import gpms
from math import exp


def explore(B, R):
    return (exp(B) + exp(R)) / (abs(exp(B) - exp(R)) + 1E-30)

def exploit(B, R):
    return B

def norm(x, shift):
    min_x = min(x)
    max_x = max(x)
    normalized = [(xi - min_x) / (max_x - min_x) + shift for xi in x]
    return normalized

def m(i, n, shift):
    exp_value = math.exp(0.25 * i)
    exp_values = [math.exp(0.25 * j) for j in range(n)]
    normalized_values = norm(exp_values, shift)
    return normalized_values[i]

def fews(args):
    "try different sample sizes"
    policies = dict(exploit = lambda B,R: B-R,
                    EXPLORE = lambda B,R: (e**B + e**R)/abs(e**B - e**R + 1E-30))
    repeats=20
    d = DATA(csv(args.dataset))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[d2h(d,row) for row in d.rows])
    rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best,_,_ = branch(d,d.rows,4); rxs[rx].add(d2h(d,best[0]))

    tests = [
        {"name": "phi3-mini", "lasts" : [20, 30], "repeats" : 10},
        {"name": "llama3-8b", "lasts" : [20, 30], "repeats" : 10},
        #{"name" : "phi3-medium", "lasts" : [15], "repeats" : 1}
    ]
    
    k = min(500, len(d.rows))
    for test in tests:
        args.llm = test["name"]
        (model, dir) =  load_model(args).get_pipeline()
        for last in test["lasts"]:
            rx =f"llm_{args.llm}({load_model(args).get_params(model)}),{last}"
            rxs[rx] = SOME(txt= rx)
            for _ in range(test["repeats"]):
                args.last = last
                rxs[rx].add(d2h(d, FEW(args, False, k, model)[0]))
        unload_model(model, dir)

    

    
    scoring_policies = [
        ('exploit', lambda B, R, I, N: exploit(B, R)),
        ('explore', lambda B, R, I, N: explore(B, R)),
        #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
        ('SimAnneal', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
        #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
    ]
            
    
    # rx =f"llm,phi3-medium,15"
    # args.llm = 'phi3-medium'
    # rxs[rx] = SOME(txt= rx)
    # args.last = 15
    # rxs[rx].add(d2h(d, vanilla1(args, False, k)[0]))

    
    for last in [20,25,30,35,40,45,50,55,60]:
      the.Last= last
      guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
      rx=f"random,{last}"
      rxs[rx] = SOME(txt=rx, inits=[d2h(d,guess()) for _ in range(repeats)])
      for  guessFaster in [True]:
        for what,how in  scoring_policies:
          the.GuessFaster = guessFaster
          rx=f"{what},{the.Last}"
          rxs[rx] = SOME(txt=rx)
          for _ in range(repeats):
             btw(".")
             rxs[rx].add(d2h(d,smo(d,how)[0]))
          btw("\n")
      
    report(rxs.values())

def zeros(args):
    "try different sample sizes"
    policies = dict(exploit = lambda B,R: B-R,
                    EXPLORE = lambda B,R: (e**B + e**R)/abs(e**B - e**R + 1E-30))
    repeats=20
    d = DATA(csv(args.dataset))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[d2h(d,row) for row in d.rows])
    rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best,_,_ = branch(d,d.rows,4); rxs[rx].add(d2h(d,best[0]))

    tests = [
        {"name": "phi3-mini", "lasts" : [20, 30, 40], "repeats" : 5},
        {"name": "llama3-8b", "lasts" : [20, 30], "repeats" : 5},
        #{"name" : "phi3-medium", "lasts" : [15], "repeats" : 1}
    ]
    
    k = min(500, len(d.rows))
    for test in tests:
        args.llm = test["name"]
        (model, dir) =  load_model(args).get_pipeline()
        for last in test["lasts"]:
            rx =f"llm_{args.llm}({load_model(args).get_params(model)}),{last}"
            rxs[rx] = SOME(txt= rx)
            for _ in range(test["repeats"]):
                args.last = last
                rxs[rx].add(d2h(d, ZERO(args, False, k, model)[0]))
        unload_model(model, dir)

    

    
    scoring_policies = [
        ('exploit', lambda B, R, I, N: exploit(B, R)),
        ('explore', lambda B, R, I, N: explore(B, R)),
        #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
        ('SimAnneal', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
        #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
    ]
            
    
    # rx =f"llm,phi3-medium,15"
    # args.llm = 'phi3-medium'
    # rxs[rx] = SOME(txt= rx)
    # args.last = 15
    # rxs[rx].add(d2h(d, vanilla1(args, False, k)[0]))

    
    for last in [20,25,30,35,40,45,50,55,60]:
      the.Last= last
      guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
      rx=f"random,{last}"
      rxs[rx] = SOME(txt=rx, inits=[d2h(d,guess()) for _ in range(repeats)])
      for  guessFaster in [True]:
        for what,how in  scoring_policies:
          the.GuessFaster = guessFaster
          rx=f"{what},{the.Last}"
          rxs[rx] = SOME(txt=rx)
          for _ in range(repeats):
             btw(".")
             rxs[rx].add(d2h(d,smo(d,how)[0]))
          btw("\n")
      
    report(rxs.values())

def warms(args):

    repeats=20
    d = DATA(csv(args.dataset))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[chebyshev(d,row) for row in d.rows])
    rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best,_,_ = branch(d,d.rows,4); rxs[rx].add(chebyshev(d,best[0]))

    scoring_policies = [
        ('exploit', lambda B, R, I, N: exploit(B, R)),
        ('explore', lambda B, R, I, N: explore(B, R)),
       
        #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
        #('Focus', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
        #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
    ]

    both = lambda B,R, I, N: explore(B, R) if I/N < 0.5 else exploit(B, R)
    
    for last in [20,25,30]:
      args.last= last
      guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
      rx=f"random,{last}"
      rxs[rx] = SOME(txt=rx, inits=[chebyshev(d,guess()) for _ in range(repeats)])
      
      gps = [('UCB_GPM'), ('PI_GPM'), ('EI_GPM')]
      for guesFaster in [True]:
        for what in gps:
            rx = f"{what},{args.last}"
            rxs[rx] = SOME(txt=rx)
            for _ in range(repeats):
                btw(".")
                rxs[rx].add(chebyshev(d, gpms(args, what)[0]))
            btw("\n")
      
      graphs = {'exploit' : [], 'LINEAR/exploit' : [], 'LLM/exploit' : []}

      for  guessFaster in [True]:
        for what,how in  scoring_policies:
          the.GuessFaster = guessFaster
          rx=f"{what},{args.last}"
          rxs[rx] = SOME(txt=rx)
          for _ in range(repeats):
            btw(".")
            res,data = smo(d,how)
            rxs[rx].add(chebyshev(d,res[0]))
            if last == 20 and what in graphs.keys() : graphs[what].append(data)
        btw("\n")

      
      
    #   for  guessFaster in [True]:
    #     for start in ['LLM']:
    #         for what,how in  scoring_policies:
    #             the.GuessFaster = guessFaster
    #             rx=f"{start}/{what},{args.last}"
    #             rxs[rx] = SOME(txt=rx)
    #             for _ in range(repeats):
    #                 btw(".")
    #                 #time.sleep(10)
    #                 if start == 'LLM' and len(d.rows) < 50:
    #                     res,data = smo(d,how) # this heuristic works because LLM warm start performs poorly across all small datasets
    #                     rxs[rx].add(chebyshev(d,res[0]))
    #                 else :
    #                     res, data = warm_smo(args,how,method = start)
    #                     rxs[rx].add(chebyshev(d,res[0]))
    #                     print("Best Row : ", res[0], "chebys :" , chebyshev(d, res[0]))
    #                 if last == 20 and f'{start}/{what}' in graphs.keys(): graphs[f'{start}/{what}'].append(data)
    #         btw("\n")


    for guessFaster in [True]:
        for start in ['LINEAR']:
            the.GuessFaster = guessFaster
            rx = f"{start}++,{30}"
            rxs[rx] = SOME(txt = rx)
            for _ in range(repeats):
                btw(".")
                res = warm_smo_plus(args, both, method = start, start = 'Diversity')
                rxs[rx].add(chebyshev(d,res[0]))
                    

       


    report(rxs.values())

    if args.graph:
        "plot the graph depending on the config"
        for n, _ in graphs.items():
            print(n, len(_))
        visualize2(args.dataset, graphs)


def clusters(args):
    repeats=20
    d = DATA(csv(args.dataset))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[chebyshev(d,row) for row in d.rows])
    rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best,_,_ = branch(d,d.rows,4); rxs[rx].add(chebyshev(d,best[0]))

    scoring_policies = [
        ('exploit', lambda B, R, I, N: exploit(B, R)),
        ('explore', lambda B, R, I, N: explore(B, R)),
        #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
        #('Focus', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
        #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
    ]
    
    for last in [20]:
      the.Last= last
      guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
      rx=f"random,{last}"
      rxs[rx] = SOME(txt=rx, inits=[chebyshev(d,guess()) for _ in range(repeats)])
      
      gps = [('UCB_GPM', ucbs(args)), ('PI_GPM', pis(args)), ('EI_GPM', eis(args))]
      for guesFaster in [True]:
        for what, how in gps:
            rx = f"{what},{the.Last}"
            rxs[rx] = SOME(txt=rx)
            for _ in range(repeats):
                btw(".")
                rxs[rx].add(d2h(d, how[0]))
            btw("\n")

    rx = f"LLM+Density/30"
    rxs[rx] = SOME(txt = rx)

    for _ in range(repeats):
        btw(".")
        res = density(args)
        rxs[rx].add(chebyshev(d, res[20]))

    report(rxs.values())



        
if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    if(args.model == 'few'):
        FEW(args, False)
    if(args.model == 'smo'):
        SMO(args)
    if(args.model == 'fews'):
        fews(args)
    if(args.model == 'zero'):
        ZERO(args,True)
    if(args.model == 'zeros'):
        zeros(args)

    if(args.model == 'warms'):
        warms(args)

    if(args.model == 'density'):
        clusters(args)
    #print(save_results())



