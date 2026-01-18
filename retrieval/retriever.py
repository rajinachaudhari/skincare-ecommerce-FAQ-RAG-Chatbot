# import faiss
# import yaml
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# # ---------------- Load config ----------------
# with open("config/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)

# # Paths from config
# chunks_csv = config["paths"]["chunks_csv"]
# faiss_index_file = config["paths"]["vector_index"]

# # Model config
# model_name = config["embedding"]["model_name"]
# device = config["embedding"]["device"]

# # ---------------- Load model and data ----------------
# model = SentenceTransformer(model_name, device=device)
# chunks_df = pd.read_csv(chunks_csv)

# # Load FAISS index
# index = faiss.read_index(faiss_index_file)

# # ---------------- Simple Retriever Function ----------------
# def semantic_search_faiss(query: str, top_k: int = 3):
#     """
#     Returns top-k relevant chunks from FAISS index with similarity scores
#     """

#     # 1️⃣ Encode the query
#     query_embedding = model.encode([query], convert_to_numpy=True)

#     # 2️⃣ Normalize embeddings if FAISS index was built with inner product
#     faiss.normalize_L2(query_embedding)

#     # 3️⃣ Search FAISS
#     distances, indices = index.search(query_embedding, top_k)

#     # 4️⃣ Collect results
#     results = []
#     for score, idx in zip(distances[0], indices[0]):
#         text = chunks_df.iloc[idx]["text"]
#         domain = chunks_df.iloc[idx]["domain"] if "domain" in chunks_df.columns else "default"
#         results.append({"domain": domain, "text": text, "score": float(score)})

#     return results





import faiss
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer

###loading config.yaml file###
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

faiss_index_file = config["paths"]["vector_index"]


###Loading chunked DataFrame directly###
chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

###selecting models directly###
model = SentenceTransformer(config["embedding"]["model_name"], device=config["embedding"]["device"])

# Load FAISS index
index = faiss.read_index(faiss_index_file)

###Retriever Function###
def semantic_search_faiss(query: str, top_k: int = 3) -> pd.DataFrame:
    """
    Cosine similarity retriever (scores ∈ [-1, 1])
    """

   ###encoding query with normalization for cosine similarity###
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    ###FAISS cosine search###
    scores, indices = index.search(query_embedding, top_k)

    rows = []
    for score, idx in zip(scores[0], indices[0]):
        rows.append({
            "domain": chunks_df.iloc[idx]["domain"],
            "text": chunks_df.iloc[idx]["text"],
            "score": float(score)
        })

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
