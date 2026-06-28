
from contextlib import asynccontextmanager
from fastapi    import FastAPI

from database.mongo_store  import init_database
from routes.pipeline       import router as pipeline_router
from routes.alerts         import AlertRouter as alerts_router
from routes.search         import SearchRouter as search_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initialising MongoDB…")
    init_database()
    yield

app = FastAPI(
    title       = "Drone Security Agent",
    description = "AI-powered drone footage analysis pipeline",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.include_router(pipeline_router, prefix="/api",        tags=["Pipeline"])
app.include_router(alerts_router,   prefix="/api/alerts",       tags=["Alerts"])
app.include_router(search_router,   prefix="",        tags=["Search & Chat"])