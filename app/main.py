from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.v1.endpoints import router as api_router
from app.core.config import settings
import os

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(api_router, prefix="/api/v1")

static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))
