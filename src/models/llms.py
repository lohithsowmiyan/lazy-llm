import os
import torch
import getpass
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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
        return f'LLM(model = {self.model_name}, temperature = {self.temperature}, max token = {self.max_tokens}, top p = {self.top_p})'
    
class Local_LLM(LLM):
    """

    """
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, **kwargs):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.quantization = kwargs.get('quantization', False)
        self.quant_bits = kwargs.get('nbits', 8)
        
        if self.quantization and self.quant_bits == 4:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype = torch.bfloat16
            )
        
        elif self.quantization and self.quant_bits == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="int8",
                bnb_8bit_compute_dtype=torch.bfloat16  
            )

    def get_model(self) -> HuggingFacePipeline:
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        pipe = pipeline(
               "text-generation",
                model= self.model,
                tokenizer= self.tokenizer,
                max_new_tokens= self.max_tokens, 
                temperature= self.temperature, 

                top_p= self.top_p,
                do_sample = True 
               )

        return HuggingFacePipeline(pipeline = pipe)

    def get_pipeline(self):
        pipe = pipeline(
         "text-generation",
         model=self.model_name,
         #device_map = "auto",
        )

        return pipe

    def get_model_with_quantization(self) -> HuggingFacePipeline:
        if not self.quantization:
            raise Exception("Your model does not have necessary quantization config setup")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = self.bnb_config)
        pipe = pipeline(
               "text-generation",
                model= self.model,
                tokenizer= self.tokenizer,
                max_new_tokens= self.max_tokens, 
                temperature= self.temperature, 
                top_p= self.top_p, 
                do_sample = True  
               )


class API_LLM(LLM):
    """

    """
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, **kwargs):
        super().__init__(model_name, temperature, max_tokens, top_p)

    def get_model(self):
        if "gemini" in self.model_name:
            if "GOOGLE_API_KEY" not in os.environ:
                print("Unable to find the API key please enter here:")
                os.environ["GOOGLE_API_KEY"] = getpass.getpass()
            return ChatGoogleGenerativeAI(
                        model= self.model_name,
                        temperature= self.temperature,  
                        max_tokens= self.max_tokens, 
                        top_p= self.top_p
                    )

        elif "gpt" in self.model_name:
            if "OPENAI_API_KEY" not in os.environ:
                print("Unable to find the API key please enter here:")
                os.environ["OPENAI_API_KEY"] = getpass.getpass()
            return OpenAI(
                        model= self.model_name,
                        temperature= self.temperature,  
                        max_tokens= self.max_tokens, 
                        top_p= self.top_p
                )


        