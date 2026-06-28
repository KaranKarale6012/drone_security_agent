"""
Starts the FastAPI server.

"""

import uvicorn 
from app import app
from database.mongo_store import init_database


########Startup Event
@app.on_event("startup")
async def on_startup():
    try:
        init_database()
    except Exception as e:
        print(f"⚠️  MongoDB init skipped: {e}")



if __name__=="__main__":
    uvicorn.run(
        "app:app",     # filename : fastapi_instance
        host = '0.0.0.0',
        port = 8000,
        reload = True  #auto restart the server when we edit files
    )