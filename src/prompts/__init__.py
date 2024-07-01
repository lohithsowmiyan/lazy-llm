from src.prompts.prompts import *

load_prompt = {
    "data/misc/auto93.csv" : Auto93Template(),
    "data/hpo/healthCloseIsses12mths0011-easy.csv" : HpoEasyTemplate(),
    "data/hpo/healthCloseIsses12mths0001-hard.csv" : HpoEasyTemplate(),
}

