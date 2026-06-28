"""  
    convert all the frames description into vector embedding using OpenCLIP
"""


import open_clip
import torch
import json
import os
from pathlib import Path
from safetensors.torch import load_file

MODEL_NAME = "ViT-B-32"
checkpoint_path = r"OpenCLIP\0_CLIPModel\model.safetensors"


# create_model() builds the architecture without any weights
_model = open_clip.create_model(MODEL_NAME)

# ── Step 2: Load safetensors weights ──────────────────────
checkpoint = load_file(checkpoint_path)


# Sometimes keys have prefixes — fix if needed
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")  # safe cleanup
    new_state_dict[new_key] = v

_model.load_state_dict(new_state_dict, strict=False)



# 3. Image preprocessing
_preprocess = open_clip.image_transform(
    _model.visual.image_size,
    is_train=False
)

# 4. Tokenizer
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# 5. Eval mode
_model.eval()


######Embedding function

def embed_text(text:str) -> list[float]:

    with torch.no_grad():
        tokens = _tokenizer([text])
        features = _model.encode_text(tokens)

        # normalize (important for cosine similarity)
        features = features / features.norm(dim=-1, keepdim=True)

        return features[0].tolist()


def run_embeddings(description_results: list[dict]) -> list:
    if not description_results:
        print("No descriptions to embed.")
        return []

    results = []
    frames_dir = str(Path(description_results[0]["filepath"]).parent)
    total = len(description_results)

    for i, frame in enumerate(description_results):
        frame_id = frame["frame_id"]
        description = frame.get("description", "")

        if not description:
            embedding = [0.0] * 512  # ViT-B-32 = 512 dim
        else:
            embedding = embed_text(description)

        results.append({
            **frame,
            "embedding": embedding
        })

        print(f"[{i+1}/{total}] Embedded {frame_id}")

    # Save lightweight metadata
    lightweight = [
        {k: v for k, v in r.items() if k != "embedding"}
        for r in results
    ]

    output_path = os.path.join(frames_dir, "embeddings_meta.json")

    with open(output_path, "w") as f:
        json.dump(lightweight, f, indent=2)

    print(f"\nEmbeddings ready — {len(results)} vectors generated")
    print(f"Reference saved → {output_path}")

    return results





