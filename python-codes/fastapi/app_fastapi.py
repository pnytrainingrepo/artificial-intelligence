# Step No 1: Install FastAPI and Uvicorn:
# pip install fastapi uvicorn

# Step No 2: Create the FastAPI application: 

from fastapi import FastAPI


app = FastAPI(title="Hello World API", 
    version="0.0.1",
    servers=[
        {
            "url": "http://0.0.0.0:8000", # ADD NGROK URL Here Before Creating GPT Action
            "description": "Development Server"
        }
        ])
        
#app = FastAPI(title="Hello World API", version="0.0.1")
              
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/pny")
def pny():
    return {"organization": "PNY Training"}

# Step No 3: # Run the application: 
# uvicorn main:app --reload

# Step No 4: Access the application: 
# goto API Access: http://127.0.0.1:8000/
# # goto API DOCS Access: http://127.0.0.1:8000/docs