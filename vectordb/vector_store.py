
import faiss
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

###loading config.yaml file###
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

model = SentenceTransformer(
    config["embedding"]["model_name"],
    device=config["embedding"]["device"]
)

###encoding chunks with normalization for cosine similarity###
embeddings = model.encode(
    chunks_df["text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

dim = embeddings.shape[1]

###Inner Product index ###
###NORMALIZED VECTORS + INNER PRODUCT = COSINE SIMILARITY###
###Note: FAISS does not have a dedicated cosine similarity index it does euclidean distance.###
index = faiss.IndexFlatIP(dim)
index.add(embeddings)  ###this is where the embeddings are added to the index to find cosine similarity###

faiss.write_index(index, config["paths"]["vector_index"])

print("FAISS index rebuilt successfully")
