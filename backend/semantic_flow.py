from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

# Load the pre-trained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

"""
Semantic Search Flow Implementation

1. User submits query
2. Analyze intent and context
3. Extract intent and relationships
4. Return intent and relationships
5. Retrieve relevant data
6. Rank data based on relevance
7. Return ranked results
8. Present generated content/output
"""

def user_submits_query(query):
    # Step 1: Entry point
    return query

def analyze_intent_and_context(query):
    # Step 2: Analyze intent and context using LLM or classifier
    # Stub: Replace with actual intent analysis
    intent = "definition"  # Example intent
    context = "general"    # Example context
    return intent, context

def extract_intent_and_relationships(query, intent, context):
    # Step 3: Extract relationships between terms
    # Stub: Replace with actual semantic parsing
    relationships = ["term1-term2"]
    return relationships

def return_intent_and_relationships(intent, relationships):
    # Step 4: Return to LLM (or next module)
    return {"intent": intent, "relationships": relationships}

def retrieve_relevant_data(intent_info, documents=None):
    # Step 5: Retrieve relevant data (semantic search or IR)
    # Accept documents as an argument for real use
    if documents is not None:
        return documents
    # For demo, use a static list of documents
    demo_documents = [
        "Semantic search uses embeddings to find similar meanings.",
        "Traditional search relies on keyword matching.",
        "Sentence transformers generate vector representations of text.",
        "Ranking algorithms sort results by relevance."
    ]
    return demo_documents

def paragraph_chunking(text):
    """Split text into paragraphs using double newlines or single newlines as fallback."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

def rank_data_based_on_relevance(data, query):
    # Step 6: Rank data using semantic similarity
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(data, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_data = [data[i] for i in ranked_indices]
    return ranked_data

def return_ranked_results(ranked_data):
    # Step 7: Return ranked results to LLM
    return ranked_data

def present_generated_content(ranked_data):
    # Step 8: Present output to user
    # Stub: Replace with actual output formatting
    return "\n".join(ranked_data)

def generate_query_ngrams(query):
    """Generate all contiguous n-grams (1 to N words) from the query."""
    words = query.split()
    ngrams = set()
    for n in range(1, len(words)+1):
        for i in range(len(words)-n+1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
    return list(ngrams)

def semantic_search_flow(query, documents=None):
    # Query chunking: generate all n-grams
    ngram_queries = generate_query_ngrams(query)
    # Collect results for all n-grams
    all_results = []
    seen = set()
    for ngram in ngram_queries:
        intent, context = analyze_intent_and_context(ngram)
        relationships = extract_intent_and_relationships(ngram, intent, context)
        intent_info = return_intent_and_relationships(intent, relationships)
        data = retrieve_relevant_data(intent_info, documents)
        ranked_data = rank_data_based_on_relevance(data, ngram)
        for item in ranked_data:
            if item not in seen:
                all_results.append(item)
                seen.add(item)
    # Optionally, re-rank all_results by relevance to the full query
    all_results = rank_data_based_on_relevance(all_results, query)
    results = return_ranked_results(all_results)
    output = present_generated_content(results)
    return output

def find_most_similar_document(query, documents):
    # Generate embeddings
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(documents, convert_to_tensor=True)
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    # Find the index of the most similar document
    best_idx = np.argmax(similarities)
    return documents[best_idx], similarities[best_idx]

# Example usage:
if __name__ == "__main__":
    user_query = "How does semantic search work? What are its advantages?"
    print("--- All Ranked Results ---")
    print(semantic_search_flow(user_query))
    print("\n--- Most Similar Document ---")
    docs = retrieve_relevant_data({}, None)
    best_doc, score = find_most_similar_document(user_query, docs)
    print(f"Most similar: {best_doc}\nCosine similarity: {score:.4f}") 