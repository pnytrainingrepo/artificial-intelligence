from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime


class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    password: str
    creation_date: datetime = Field(default_factory=datetime.utcnow)

class User(UserBase, table=True):
    userid: Optional[int] = Field(default=None, primary_key=True)
    
    products: List["Product"] = Relationship(back_populates="owner")
    orders: List["Order"] = Relationship(back_populates="user")
    
class UserCreate(UserBase):
    pass

class UserPublic(UserBase):
    id: int

class Product(SQLModel, table=True):
    product_id: Optional[int] = Field(default=None, primary_key=True)
    product_name: str
    product_desc: str
    product_price: float
    product_qty: int
    product_created_date: datetime = Field(default_factory=datetime.utcnow)
    userid: int = Field(foreign_key="user.userid")

    owner: Optional[User] = Relationship(back_populates="products")

class Order(SQLModel, table=True):
    order_id: Optional[int] = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="product.product_id")
    order_qty: int
    order_created_date: datetime = Field(default_factory=datetime.utcnow)
    userid: int = Field(foreign_key="user.userid")

    user: Optional[User] = Relationship(back_populates="orders")
