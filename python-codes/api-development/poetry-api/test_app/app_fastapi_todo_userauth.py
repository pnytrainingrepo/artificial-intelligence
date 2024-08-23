from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import SQLModel, Field, create_engine, Session, select
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

# SQLModel classes for User and TodoItem
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str

class TodoItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")

# Pydantic models for token
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# FastAPI app initialization
app = FastAPI(title="TODO API", version="1.0.0")

# Database URL
DATABASE_URL = "postgresql://postgres:speed123@localhost/todo_db"

# Create the database engine
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = "PNYTraining"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper functions for password hashing and token creation
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if user is None:
            raise credentials_exception
    return user

# Run this function to create the database and tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Routes for user registration, login, and todo management

# WELCOME ROUTE
@app.get("/")
def welcome():
    return "Welcome to my TODO API"

# User registration route
@app.post("/register/")
def register(user: User):
    with Session(engine) as session:
        hashed_password = get_password_hash(user.hashed_password)
        db_user = User(username=user.username, hashed_password=hashed_password)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user

# User login route to get access token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == form_data.username)).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
        return {"access_token": access_token, "token_type": "bearer"}

# Create new todo item
@app.post("/todos/", response_model=TodoItem)
async def create_todo(todo: TodoItem, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        todo.user_id = current_user.id
        session.add(todo)
        session.commit()
        session.refresh(todo)
        return todo

# Get all todos for current user
@app.get("/todos/", response_model=List[TodoItem])
async def get_todos(current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        todos = session.exec(select(TodoItem).where(TodoItem.user_id == current_user.id)).all()
        return todos

# Update a todo item
@app.put("/todos/{todo_id}", response_model=TodoItem)
async def update_todo(todo_id: int, todo: TodoItem, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        db_todo = session.get(TodoItem, todo_id)
        if not db_todo or db_todo.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Todo not found")
        db_todo.title = todo.title
        db_todo.description = todo.description
        db_todo.completed = todo.completed
        session.commit()
        session.refresh(db_todo)
        return db_todo

# Delete a todo item
@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int, current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        db_todo = session.get(TodoItem, todo_id)
        if not db_todo or db_todo.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Todo not found")
        session.delete(db_todo)
        session.commit()
        return {"detail": "Todo deleted"}


# To run the app, use: uvicorn main:app --reload
