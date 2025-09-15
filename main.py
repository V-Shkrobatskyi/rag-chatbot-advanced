from fastapi import FastAPI
from app.routes import router
from app.utils import clear_stored_txt
from app.state import AppState
from app.config import PROJECT_NAME

app = FastAPI(title=PROJECT_NAME)


@app.on_event("startup")
async def startup_event():
    app.state = AppState()
    clear_stored_txt()

# Import all endpoints from routes.py
app.include_router(router)
