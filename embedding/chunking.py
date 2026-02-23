
from typing import List
import yaml
import pandas as pd
from pathlib import Path


###loading config.yaml file###
def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


###chunking function###
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


###main chunking runner/process###
def run_chunking():
    config = load_config()

    input_file = config["paths"]["input_documents"]
    output_file = config["paths"]["output_chunks"]
    processed_dir = config["paths"]["processed_dir"]

    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["overlap"]
    domain = config["chunking"]["domain"]
    encoding = config["chunking"]["encoding"]

    
    with open(input_file, "r", encoding=encoding) as f:
        raw_text = f.read()   #opens the input file and reads the content as a single string

    
    documents = [
        doc.strip()
        for doc in raw_text.split("\n---\n")
        if doc.strip()
    ]

    chunked_docs = []

  
    for doc in documents:
        chunks = chunk_text(
            text=doc,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:
            chunked_docs.append({
                "domain": domain,
                "text": chunk
            })

    
    df = pd.DataFrame(chunked_docs)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)


    df.to_csv(output_file, index=False, encoding=encoding)

    print("Chunking completed successfully")
    print(f"Total Q&A documents: {len(documents)}")
    print(f"Total chunks created: {len(df)}")
    print(df.head())


###run the chunking process###
if __name__ == "__main__":
    run_chunking()

