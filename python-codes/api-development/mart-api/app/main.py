from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
from datetime import timedelta
from . import models, crud, auth
from .auth import create_access_token, authenticate_user, get_current_user
from .database import create_db_and_tables, get_session

app = FastAPI()

# Create the database and tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# User Registration
@app.post("/users/", response_model=models.User)
def register_user(user: models.User, session: Session = Depends(get_session)):
    db_user = crud.get_user_by_username(session, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    return crud.create_user(session, user)

# User Login for JWT token
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = crud.get_user_by_username(session, form_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# Get current user information
@app.get("/users/me/", response_model=models.User)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

# Protected Product Creation (Requires Authentication)
@app.post("/products/", response_model=models.Product)
def add_product(product: models.Product, session: Session = Depends(get_session), current_user: models.User = Depends(get_current_user)):
    product.userid = current_user.userid
    return crud.create_product(session, product)

# Get all products
@app.get("/products/", response_model=list[models.Product])
def list_products(session: Session = Depends(get_session)):
    return crud.get_products(session)

# Create Order (Requires Authentication)
@app.post("/orders/", response_model=models.Order)
def place_order(order: models.Order, session: Session = Depends(get_session), current_user: models.User = Depends(get_current_user)):
    order.userid = current_user.userid
    return crud.create_order(session, order)

# Get all orders
@app.get("/orders/", response_model=list[models.Order])
def list_orders(session: Session = Depends(get_session)):
    return crud.get_orders(session)
