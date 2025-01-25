from src.language.llms import Local_LLM, API_LLM, LLM

api_model_path = {
    "gemini" : "gemini-1.5-pro",
    'gpt' : "gpt-4o"
}

local_model_path = {
    "llama2-7b" : "meta-llama/Llama-2-7b-chat-hf",
    "llama3-8b" : "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral" : "filipealmeida/Mistral-7B-Instruct-v0.1-sharded",
    "phi3-mini" : "microsoft/Phi-3-mini-4k-instruct",
    "phi3-small" : "microsoft/Phi-3-small-8k-instruct",
    "phi3-medium" : "microsoft/Phi-3-medium-4k-instruct",
    "mistral-7b" : "mistralai/Mistral-7B-Instruct-v0.3"
}

def load_model(args, name = None) -> LLM:
    """

    """

    if args.llm in api_model_path.keys() or name in api_model_path.keys():
        return API_LLM(api_model_path[args.llm] if name == None else api_model_path[name], args.temperature, args.max_tokens, args.top_p)

    elif args.llm in local_model_path.keys() or name in local_model_path.keys():
        return Local_LLM(local_model_path[args.llm] if name == None else local_model_path[name], args.temperature, args.max_tokens, args.top_p, args.cache, quantization =  args.quantization, nbits = args.q_bits)

    else:
        raise Exception("Model Not Found. Add the Model to src/models/__init__.py")

    
