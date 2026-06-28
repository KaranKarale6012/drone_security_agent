import os
import chromadb
import numpy as np
from agents.embedding import embed_text

# ── CONFIG ────────────────────────────────────────────────────
CHROMA_PATH = os.path.join("output", "chromadb")


### ── Embedding for loitering────────────────────────────────────────────────────
LOITERING_CONCEPT = """
A person staying in the same location for an extended period of time without a clear purpose,
showing minimal movement or repeatedly appearing in the same area,
waiting, lingering, or observing surroundings in a suspicious or unusual manner.
Includes behavior such as standing idle near restricted areas, slowly pacing,
hesitating, repeatedly approaching and leaving a spot, or monitoring activity
without engaging in a normal task.
Often associated with surveillance, reconnaissance, or intent to perform suspicious activity.
"""

LOITERING_EMBED = embed_text(LOITERING_CONCEPT)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── CLIENT ────────────────────────────────────────────────────
def _get_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection(video_name: str):
    client = _get_client()
    return client.get_or_create_collection(
        name=f"video_{video_name}",
        metadata={"hnsw:space": "cosine"}
    )


# ── STORE ─────────────────────────────────────────────────────
def store_embeddings(video_name: str, embedded_frames: list[dict]) -> int:

    if not embedded_frames:
        print("  No frames to store in ChromaDB.")
        return 0

    collection = _get_collection(video_name)

    ids, embeddings, documents, metadatas = [], [], [], []

    for frame in embedded_frames:

        if not frame.get("description") or not frame.get("embedding"):
            continue

        print("frames in chroma store :", frame)

        # ✅ USE DESCRIPTION (MAIN FIX)
        description = frame.get("description", "").lower()

        # ── Extract objects from text ─────────────────────────
        objects = []

        if "person" in description or "individual" in description:
            objects.append("person")

        if any(v in description for v in ["car", "truck", "bus", "motorcycle", "vehicle"]):
            objects.append("vehicle")

        # ── Suspicion detection ───────────────────────────────
        if "suspicion level: suspicious" in description:
            is_suspicious = True
        elif "suspicion level: non-suspicious" in description:
            is_suspicious = False
        else:
            is_suspicious = "suspicious" in description


        # ── Loitering detection using embeddings ─────────────────────
        desc_embed = frame["embedding"]

        score = cosine_sim(desc_embed, LOITERING_EMBED)

        loitering = score > 0.65   # ✅ tune if needed


        # ── Metadata ──────────────────────────────────────────
        metadata = {
            "video_name": video_name,
            "frame_id": frame["frame_id"],
            "timestamp_sec": float(frame.get("timestamp_sec", 0)),

            "objects": ",".join(objects),
            
            "has_person": int("person" in objects),
            "has_vehicle": int("vehicle" in objects),
            "loitering": int(loitering),
            "is_suspicious": int(is_suspicious)

        }

        print("metadata chromadb:", metadata)

        ids.append(f"{video_name}_{frame['frame_id']}")
        embeddings.append(frame["embedding"])
        documents.append(frame["description"])
        metadatas.append(metadata)

    if not ids:
        print("  No valid frames to store in ChromaDB.")
        return 0

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"  ✅ Stored {len(ids)} frames → ChromaDB 'video_{video_name}'")
    return len(ids)


# ── SEARCH ────────────────────────────────────────────────────
def semantic_search(
    query: str,
    video_name: str = None,
    n_results: int = 5,
    filters: dict = None
) -> list[dict]:

    client = _get_client()
    query_embed = embed_text(query)

    if video_name:
        try:
            collections = [client.get_collection(f"video_{video_name}")]
        except Exception:
            print(f"  ⚠️  Collection 'video_{video_name}' not found")
            return []
    else:
        all_cols = client.list_collections()
        collections = [
            client.get_collection(col.name)
            for col in all_cols
            if col.name.startswith("video_")
        ]

    if not collections:
        print("  ⚠️  No collections found in ChromaDB")
        return []

    all_results = []

    for col in collections:

        if col.count() == 0:
            continue

        actual_n = min(n_results, col.count())

        query_params = {
            "query_embeddings": [query_embed],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"]
        }

        if filters:
            query_params["where"] = filters

        results = col.query(**query_params)

        for i, doc in enumerate(results["documents"][0]):
            similarity = round(1 - results["distances"][0][i], 3)

            all_results.append({
                "description": doc,
                "metadata": results["metadatas"][0][i],
                "similarity": similarity
            })

    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return all_results[:n_results]


# ── DELETE ────────────────────────────────────────────────────
def delete_video_collection(video_name: str) -> None:
    client = _get_client()

    try:
        client.delete_collection(f"video_{video_name}")
        print(f"  ✅ Deleted ChromaDB collection 'video_{video_name}'")
    except Exception:
        print(f"  ⚠️  Collection 'video_{video_name}' not found — nothing to delete")