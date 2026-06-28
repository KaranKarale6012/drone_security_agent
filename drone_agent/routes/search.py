from fastapi  import APIRouter
from pydantic import BaseModel
from typing   import Optional


from agents.security_agent import run_chat_agent
from database.chroma_store import semantic_search


SearchRouter = APIRouter()

class SearchRequest(BaseModel):
    query:str
    video_name: Optional[str] = None   # None = search all videos
    n_results:  int = 5

class ChatRequest(BaseModel):
    question:   str
    video_name: Optional[str] = None

@SearchRouter.post("/api/search")
def search_frames(request: SearchRequest):
    """
    Semantic search — finds frames by meaning not keywords.
    Example: “person near fence at night" finds relevant frames
    even if those exact words weren’t in the description.
    """
    results = semantic_search(
    query      = request.query,
    video_name = request.video_name,
    n_results  = request.n_results
    )


    return {
        "query":   request.query,
        "results": results,
        "count":   len(results)
    }


@SearchRouter.post("/api/chat")
def chat(request: ChatRequest):
    """
    Ask a natural language question about the footage.
    The LangGraph ChatAgent searches ChromaDB + MongoDB
    and Claude Sonnet synthesizes the answer.

    Example questions:
    "Were there any suspicious people last night?"
    "How many times did a vehicle appear?"
    "What happened near the gate?"
    """
    answer = run_chat_agent(
        question   = request.question,
        video_name = request.video_name
    )

    return {
        "question": request.question,
        "answer":   answer
    }
