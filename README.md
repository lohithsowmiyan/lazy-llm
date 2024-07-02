
# LazyLLM (LLMs For SMO)



## Environment Setup



```bash
#clone the repository
git clone https://github.com/lohithsowmiyan/lazy-llm.git
cd lazy-llm

#Install necessary modules
pip install -r requirements.txt

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





## Acknowledgements

 - [EZR](https://github.com/timm/ezr/tree/24Jun14?tab=readme-ov-file)

