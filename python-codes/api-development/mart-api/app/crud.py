from sqlmodel import Session, select
from .models import User, Product, Order
from .auth import hash_password

def create_user(session: Session, user: User):
    user.password = hash_password(user.password)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

def get_user_by_username(session: Session, username: str):
    statement = select(User).where(User.username == username)
    result = session.exec(statement)
    return result.first()

def create_product(session: Session, product: Product):
    session.add(product)
    session.commit()
    session.refresh(product)
    return product

def get_products(session: Session):
    return session.exec(select(Product)).all()

def create_order(session: Session, order: Order):
    session.add(order)
    session.commit()
    session.refresh(order)
    return order

def get_orders(session: Session):
    return session.exec(select(Order)).all()
