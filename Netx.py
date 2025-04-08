import pandas as pd
import networkx as nx

# === 1. Load OpenAlex corpus ===
# Assuming you have a JSON or CSV file with OpenAlex data
# with columns like 'authors', 'title', 'id', etc.

df = pd.read_csv("research_data.CSV")  # or use read_csv if needed

# === 2. Create a co-authorship graph ===
G = nx.Graph()

for idx, row in df.iterrows():
    authors = row.get("authorships", [])
    author_ids = [a["author"]["id"] for a in authors if a.get("author")]
    
    # Add nodes and edges for co-authorship
    for i in range(len(author_ids)):
        G.add_node(author_ids[i])
        for j in range(i + 1, len(author_ids)):
            if G.has_edge(author_ids[i], author_ids[j]):
                G[author_ids[i]][author_ids[j]]["weight"] += 1
            else:
                G.add_edge(author_ids[i], author_ids[j], weight=1)

# === 3. Export to Gephi-readable format ===
nx.write_gexf(G, "coauthorship_network.gexf")

print("Graph exported to 'coauthorship_network.gexf'. You can now open it in Gephi.")
