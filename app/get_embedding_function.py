from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings



def get_embedding_function():
    #For AWS bedrock platform Embedding model setup
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    #For Local Embedding model setup
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2", #model downloaded to local server to support
        model_kwargs={'device': 'cpu'},  # or 'cuda' if we have a GPU
        encode_kwargs={'normalize_embeddings': True},
        cache_folder="models"
    )    
    return embeddings


