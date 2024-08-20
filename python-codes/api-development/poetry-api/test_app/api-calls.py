from fastapi import FastAPI

app = FastAPI()

@app.get("/hi") 
def greet():
    return "Hello? World?"

@app.get("/hi/{name}")
def greet_with_name(name: str):
    return "Hello? World, " + name