from fastapi import APIRouter

from app.models.user import UserCreate, UserLogin

router = APIRouter()


@router.post("/register")
async def register(user: UserCreate):
    return {"message": "Registration not yet implemented", "username": user.username}


@router.post("/login")
async def login(credentials: UserLogin):
    return {"message": "Login not yet implemented", "email": credentials.email}


@router.get("/me")
async def get_current_user():
    return {"message": "Auth not yet implemented", "user": None}
