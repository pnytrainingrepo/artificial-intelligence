# STEP NO 1: Step 1: Set Up Your Environment
# pip install fastapi uvicorn sqlmodel psycopg2-binary


# fastapi: The web framework.
# 
# # uvicorn: The ASGI server to run the application.

# sqlmodel: The ORM and Pydantic model wrapper for SQLAlchemy.
# 
# psycopg2-binary: The PostgreSQL database adapter for Python.

# STEP NO 2: Set Up PostgreSQL
# CREATE DATABASE todo_db;

# STEP NO 3: Create the FastAPI Application (main.py)


from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import List, Optional

# Define the ToDo model
class ToDo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    completed: bool = Field(default=False)

# Database URL
DATABASE_URL = "postgresql://postgres:speed123@localhost/todo_db"

# Create the database engine
engine = create_engine(DATABASE_URL, echo=True)

# Create the FastAPI app
app = FastAPI()

# Dependency to get the database session
def get_session():
    with Session(engine) as session:
        yield session

# Initialize the database
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Create a new task
@app.post("/todos/", response_model=ToDo)
def create_todo(todo: ToDo, session: Session = Depends(get_session)):
    session.add(todo)
    session.commit()
    session.refresh(todo)
    return todo

# Read all tasks
@app.get("/todos/", response_model=List[ToDo])
def read_todos(session: Session = Depends(get_session)):
    todos = session.exec(select(ToDo)).all()
    return todos

# Read a specific task by ID
@app.get("/todos/{todo_id}", response_model=ToDo)
def read_todo(todo_id: int, session: Session = Depends(get_session)):
    todo = session.get(ToDo, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="ToDo not found")
    return todo

# Update a task
@app.put("/todos/{todo_id}", response_model=ToDo)
def update_todo(todo_id: int, updated_todo: ToDo, session: Session = Depends(get_session)):
    todo = session.get(ToDo, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="ToDo not found")
    todo.title = updated_todo.title
    todo.description = updated_todo.description
    todo.completed = updated_todo.completed
    session.commit()
    session.refresh(todo)
    return todo

# Delete a task
@app.delete("/todos/{todo_id}", response_model=dict)
def delete_todo(todo_id: int, session: Session = Depends(get_session)):
    todo = session.get(ToDo, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="ToDo not found")
    session.delete(todo)
    session.commit()
    return {"ok": True}


# STEP NO 4: Run the Application
# 1: uvicorn main:app --reload
# 2: http://127.0.0.1:8000
# 3: 


# STEP NO 5: Interact with the API
# Access ENDPOINTS
# POST /todos/: Create a new task.
# GET /todos/: Retrieve all tasks.

