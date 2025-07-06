from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, b_FastAPI"}

@app.get("/hello/{name}")
def greet(name: str):
    return {"greeting": f"Hello, {name}!"}

@app.get("/hello/{name}- FAST api learning")
def greet(name: str):
    return {"greeting": f"Hello, {name}!"}