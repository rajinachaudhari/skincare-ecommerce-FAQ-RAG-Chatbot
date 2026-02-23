
# import faiss
# import yaml
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# ###loading config.yaml file###
# with open("config/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)

# faiss_index_file = config["paths"]["vector_index"]


# ###Loading chunked DataFrame directly###
# chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

# ###selecting models directly###
# model = SentenceTransformer(config["embedding"]["model_name"], device=config["embedding"]["device"])

# # Load FAISS index
# index = faiss.read_index(faiss_index_file)

# ###Retriever Function###
# def semantic_search_faiss(query: str, top_k: int = 3) -> pd.DataFrame:
#     """
#     Cosine similarity retriever (scores ∈ [-1, 1])
#     """

#    ###encoding query with normalization for cosine similarity###
#     query_embedding = model.encode(
#         [query],
#         convert_to_numpy=True,
#         normalize_embeddings=True
#     )

#     ###FAISS cosine search###
#     scores, indices = index.search(query_embedding, top_k)

#     rows = []
#     for score, idx in zip(scores[0], indices[0]):
#         rows.append({
#             "domain": chunks_df.iloc[idx]["domain"],
#             "text": chunks_df.iloc[idx]["text"],
#             "score": float(score)
#         })

#     return (
#         pd.DataFrame(rows)
#         .sort_values("score", ascending=False)
#         .reset_index(drop=True)
#     )



import faiss
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer


class FaissSemanticRetriever:
    """
    Runtime FAISS retriever for semantic search.
    Loads model + index once.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)

        # Load chunks dataframe (metadata)
        self.chunks_df = pd.read_csv(self.config["paths"]["chunks_csv"])

        # Load embedding model (for query encoding only)
        self.model = SentenceTransformer(
            self.config["embedding"]["model_name"],
            device=self.config["embedding"]["device"]
        )

        # Load FAISS index
        self.index = faiss.read_index(self.config["paths"]["vector_index"])

    @staticmethod
    def _load_config(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def semantic_search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """
        Perform cosine similarity search using FAISS.
        Returns top_k most relevant chunks.
        """

        # Encode query (must be normalized)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "domain": self.chunks_df.iloc[idx]["domain"],
                "text": self.chunks_df.iloc[idx]["text"],
                "score": float(score)
            })

        return (
            pd.DataFrame(results)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )