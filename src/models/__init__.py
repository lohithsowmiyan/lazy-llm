from src.models.llms import Local_LLM, API_LLM, LLM

api_model_path = {
    "gemini" : "gemini-1.5-flash",
}

local_model_path = {
    "llama2-7b" : "meta-llama/Llama-2-7b-chat-hf",
    "llama3-8b" : "meta-llama/Meta-Llama-3-8B",
    "mistral" : "filipealmeida/Mistral-7B-Instruct-v0.1-sharded",
    "phi3" : "microsoft/Phi-3-mini-4k-instruct"
}

def load_model(args) -> LLM:
    """

    """

    if args.llm in api_model_path.keys():
        return API_LLM(api_model_path[args.llm], args.temperature, args.max_tokens, args.top_p)

    elif args.llm in local_model_path.keys():
        return Local_LLM(local_model_path[args.llm], args.temperature, args.max_tokens, args.top_p, quantization =  args.quantization, nbits = args.q_bits)

    else:
        raise Exception("Model Not Found. Add the Model to src/models/__init__.py")

    