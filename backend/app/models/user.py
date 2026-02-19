from pydantic import BaseModel


class User(BaseModel):
    id: str
    username: str
    email: str


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str
