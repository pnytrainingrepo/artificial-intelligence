# CRUD OPERATION

from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import Session, select
from .database import create_db_and_tables, get_session
from .models import TodoItem

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/todos/", response_model=TodoItem)
def create_todo(*, session: Session = Depends(get_session), todo: TodoItem):
    session.add(todo)
    session.commit()
    session.refresh(todo)
    return todo

@app.get("/todos/", response_model=list[TodoItem])
def read_todos(*, session: Session = Depends(get_session)):
    todos = session.exec(select(TodoItem)).all()
    return todos

@app.get("/todos/{todo_id}", response_model=TodoItem)
def read_todo(*, session: Session = Depends(get_session), todo_id: int):
    todo = session.get(TodoItem, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.put("/todos/{todo_id}", response_model=TodoItem)
def update_todo(*, session: Session = Depends(get_session), todo_id: int, todo: TodoItem):
    db_todo = session.get(TodoItem, todo_id)
    if not db_todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    db_todo.title = todo.title
    db_todo.description = todo.description
    db_todo.is_completed = todo.is_completed
    session.commit()
    session.refresh(db_todo)
    return db_todo

@app.delete("/todos/{todo_id}")
def delete_todo(*, session: Session = Depends(get_session), todo_id: int):
    todo = session.get(TodoItem, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    session.delete(todo)
    session.commit()
    return {"ok": True}
