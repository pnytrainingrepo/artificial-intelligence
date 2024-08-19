from sqlmodel import SQLModel, create_engine, Session
from .models import TodoItem

DATABASE_URL = "postgresql://username:password@localhost/todo_app"

engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
