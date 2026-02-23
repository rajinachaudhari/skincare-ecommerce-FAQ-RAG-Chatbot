
# import faiss
# import pandas as pd
# import yaml
# from sentence_transformers import SentenceTransformer

# ###loading config.yaml file###
# with open("config/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)
# #loading chunk and embedding#
# chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

# model = SentenceTransformer(
#     config["embedding"]["model_name"],
#     device=config["embedding"]["device"]
# )

# ###encoding chunks with normalization for cosine similarity###
# embeddings = model.encode(
#     chunks_df["text"].tolist(),
#     convert_to_numpy=True,
#     normalize_embeddings=True
# )

# dim = embeddings.shape[1]

# ###Inner Product index ###
# ###NORMALIZED VECTORS + INNER PRODUCT = COSINE SIMILARITY###
# ###Note: FAISS does not have a dedicated cosine similarity index it does euclidean distance.###
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)  ###this is where the embeddings are added to the index to find cosine similarity###

# faiss.write_index(index, config["paths"]["vector_index"])

# print("FAISS index rebuilt successfully")

import faiss
import yaml
import numpy as np
from pathlib import Path


def load_config():
    """Load configuration from YAML file."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_faiss_index():
    """
    Build FAISS index from precomputed normalized embeddings.
    This is an OFFLINE step.
    """

    config = load_config()

    embeddings_path = config["paths"]["embeddings_file"]
    index_path = config["paths"]["vector_index"]

    # Load normalized embeddings
    embeddings = np.load(embeddings_path)

    # FAISS requires float32
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    # Inner Product index (works as cosine similarity because vectors are normalized)
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to index
    index.add(embeddings)

    # Ensure directory exists
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, index_path)

    print("FAISS index built successfully.")
    print(f"Indexed vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")


if __name__ == "__main__":
    build_faiss_index()
