from src.prompts.prompts import *

prompts = {
    "data/misc/auto93.csv" : Auto93Template(),
    "data/hpo/healthCloseIsses12mths0011-easy.csv" : HpoEasyTemplate(),
    "data/hpo/healthCloseIsses12mths0001-hard.csv" : HpoEasyTemplate(),
}

def load_prompt(dataset : str = ''):

    if dataset in prompts.keys():
        return prompts[dataset]
    else:
        return Template()

