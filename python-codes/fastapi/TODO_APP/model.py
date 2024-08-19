# Create the FastAPI Application

from sqlmodel import SQLModel, Field
from typing import Optional

class TodoItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    is_completed: bool = Field(default=False)

