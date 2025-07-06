from fastapi import FastAPI
from pydantic import BaseModel
from query_data import query_rag, expand_query

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/ask")
def ask_model(query: Query):
    expanded = expand_query(query.text)
    print(f"[Expanded Query] {expanded}")
    result = query_rag(expanded)
    return {"expanded_query": expanded, "response": result}

@app.post("/expand")
def expand_query_endpoint(query: Query):
    expanded_text = expand_query(query.text)
    return {"expanded_query": expanded_text}