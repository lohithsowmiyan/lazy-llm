import torch
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class LLM:
    """
      Class LLM: it is the parent class of all the LLM Models we are going to use for this project
      params: it just takes the parameters like model names and basic configuration of the LLMS
    """
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def __repr__(self):
        return {"model" : self.model_name,
         "temperature" : self.temperature, "max token" : self.max_tokens,
          "top p" : self.top_p}


class Local_LLM(LLM):
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, **kwargs):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_model():
        pipe = pipeline(
               "text-generation",
                model= self.model,
                tokenizer= self.tokenizer,
                max_new_tokens= self.max_tokens, 
                temperature= self.temperature, 
                top_p= self.top_p  
               )

        return HuggingFacePipeline(pipeline = pipe)




        
