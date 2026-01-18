import pandas as pd
import yaml
from pathlib import Path

def csv_to_documents():
    # Load config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_path = config["paths"]["input_csv"]
    output_path = config["paths"]["output_documents"]
    processed_dir = config["paths"]["processed_dir"]

    q_col = config["processing"]["question_column"]
    a_col = config["processing"]["answer_column"]

    
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        question = str(row[q_col]).strip()
        answer = str(row[a_col]).strip()

        doc = f"""Question: {question}
Answer: {answer}
"""
        documents.append(doc)

    # checking if directory exists
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Write to txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(documents))

    print(f"Converted {len(documents)} Q&A pairs into documents.txt")

if __name__ == "__main__":
    csv_to_documents()
