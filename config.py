import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Base deployment (normal Azure model)
BASE_DEPLOYMENT = "gpt-4o"

# Fine-tuned deployment
FT_DEPLOYMENT = "gpt-4o-finetune-model"

# Base model for planning / routing / reasoning
smart_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_deployment=BASE_DEPLOYMENT,
    temperature=0.2
)

# Base model for fast operations
fast_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_deployment=BASE_DEPLOYMENT,
    temperature=0.0
)

# Fine-tuned model ONLY for final answer
ft_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_deployment=FT_DEPLOYMENT,
    temperature=0.2
)

# default fallback
llm = smart_llm