import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

# Function to load documents from an Excel file
def load_documents_from_excel(file_path, sheet_name=0, text_column="source"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[text_column].dropna().tolist()  # Drop missing values and return as a list

def average_embeddings(documents, batch_size=32, model_max_length=512, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)

    model.eval()
    average_embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding vocabulary"):
            batch_inputs = tokenizer(batch, padding="max_length", max_length=model_max_length, truncation=True, return_tensors="pt")
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            last_hidden_state = model(**batch_inputs).last_hidden_state
            avg_embedding = last_hidden_state.mean(dim=1)
            average_embeddings.append(avg_embedding.cpu().numpy())

    document_vectors = normalize(np.vstack(average_embeddings))
    return document_vectors

# Example usage:
file_path = "research_data.xlsx"  # Change to your actual file path
documents = load_documents_from_excel(file_path, text_column="source")  # Adjust column name
document_vectors = average_embeddings(documents)

# Print first few embeddings
print(document_vectors[:5])