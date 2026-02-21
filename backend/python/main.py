from fastapi import FastAPI

from app.middleware.cors import add_cors_middleware
from app.routes import health, tracks, katas, execute, execute_stream, auth

app = FastAPI(title="Python AI Katas", version="0.1.0")

add_cors_middleware(app)

app.include_router(health.router, tags=["health"])
app.include_router(tracks.router, prefix="/api", tags=["tracks"])
app.include_router(katas.router, prefix="/api", tags=["katas"])
app.include_router(execute.router, prefix="/api", tags=["execute"])
app.include_router(execute_stream.router, prefix="/api", tags=["execute"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
