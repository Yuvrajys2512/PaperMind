from ingestion.query_router import route_query
from ingestion.generator import generate_answer
 
r = route_query("How is relevance computed between tokens?", "attention-is-all-you-need")
g = generate_answer(r["query"], r["chunks"], r["intents"])
print(g["answer"][:500])
