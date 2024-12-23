
# Can Large Language Models Improve SE Active Learning via Warm-Starts 


## Abstract
When SE data is scarce,  active learners   use   models learned from tiny samples of the data to find the next most informative example to label. In this way, effective models can be generated using very little data.
For multi-objective software engineering (SE) tasks,
active learning can benefit from an effective 
 set of initial guesses (also known as warm starts).  This paper explores the use of Large Language Models (LLMs) for creating warm-starts. Those results are  compared   against   Gaussian Process Models and Tree of Parzen Estimators. For 49 SE  tasks, LLM-generated warm starts significantly improved the performance of low- and medium-dimensional tasks. However, LLM effectiveness  diminishes in high-dimensional problems, where Bayesian methods like Gaussian Process Models perform best. 


This repo geenrates the reports needed to address research questions 1,2,3 fro our recent [![paper]]([https://lohithsowmiyan.com/](https://github.com/lohithsowmiyan/lazy-llm/blob/main/docs/paper.pdf)) on combining LLMs with active learning for SE multi-objective optimization problems.
## Environment Setup



```bash
#clone the repository
git clone https://github.com/lohithsowmiyan/lazy-llm.git
cd lazy-llm

#Install necessary modules
pip install -r requirements.txt --no-warn-script-location

#create a .env file and add all the necessary tokens
touch .env

## inside the .env file add these environemt variables
HF_TOKEN = XXXXX XXXXX (Your token from Huggingface)
GOOGLE_API_KEY = XXXXX XXXXX (Your key for gemini)
OPENAI_API_KEY = XXXXX XXXXX (Your key from Open AI)
```





## Usage/Examples

```bash
#Example 1
python lazy.py --model vanilla --llm llama3-8b --dataset data/misc/auto93.csv
#Example 2
python lazy.py --model vanilla --llm gpt-3.5-turbo --dataset data/misc/wine_quality.csv
```


## Configurations

#### Experiment Settings

| Parameter | Values     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `vanilla` | Simple Greedy Selector (LLM) |
| `model` | `smo` | Sequential Model Optimization (Baseline) |
| `llm` | `llama3-8b`, `gemini-pro`, `phi3-medium`, `gpt-4`  | **Can add any model via llm.py file** |
| `dataset` | `data/misc/auto93.csv` | Enter the full path for any of the datasets in repository|
| `explanation`      | `true` | Log / Display the rational behind the model decisons for every label |

#### LLM Settings

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `temperature`      | `0 - 1` | Controls the creativity of the model |
| `max_tokens`      | `50` | Leave it at default values |
| `top_p`      | `0 - 1` | Focus levels on the core of the prompt |



#### Optional Settings

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `quantization`      | `True` or  `False` | May affect  model performance |
| `q_bits`      | `4` or  `8` | Lower value results in low operating cost |


## Visualization/Examples

```bash
#Example 1
python3 graph.py auto93.csv All
#Example 2
python3 graph.py healthCloseIsses12mths0011-easy Mu
```


## Acknowledgements

 - [EZR](https://github.com/timm/ezr/tree/24Jun14?tab=readme-ov-file)

