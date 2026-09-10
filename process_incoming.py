from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import re


def create_embedding(text_list):
    """Generate embeddings using Ollama + BGE-M3."""
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()

    return r.json()["embeddings"]


def inference(prompt):
    """Generate a response using Ollama + Llama 3.2."""
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    r.raise_for_status()

    return r.json()["response"]


def normalize_text(text):
    """Normalize text for simple matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def format_time(seconds):
    """Convert seconds into MM:SS format."""
    total_seconds = int(round(float(seconds)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


# -------------------------------------------------------------------
# Load stored course embeddings
# -------------------------------------------------------------------

df = joblib.load("embeddings.joblib")

# -------------------------------------------------------------------
# Take user question
# -------------------------------------------------------------------

incoming_query = input("Ask a Question: ")

# -------------------------------------------------------------------
# Create embedding for the question
# -------------------------------------------------------------------

question_embedding = create_embedding([incoming_query])[0]

# -------------------------------------------------------------------
# Semantic similarity
# -------------------------------------------------------------------

similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

# Retrieve top 5 semantic matches
top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]

retrieved_df = df.iloc[max_indx].copy()

# -------------------------------------------------------------------
# Improve selection among the top retrieved chunks
# -------------------------------------------------------------------

query = normalize_text(incoming_query)
query_words = set(query.split())

# Remove very common words that do not help identify the topic
stop_words = {
    "where",
    "is",
    "are",
    "the",
    "a",
    "an",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "what",
    "which",
    "how",
    "this",
    "that"
}

important_query_words = query_words - stop_words


def calculate_content_score(text):
    """
    Score how strongly the transcript directly matches
    the important words in the user's question.
    """
    normalized = normalize_text(text)
    text_words = set(normalized.split())

    if not important_query_words:
        return 0

    return len(important_query_words.intersection(text_words))


def calculate_instructional_score(text):
    """
    Give extra weight to chunks that contain instructional
    language, since the user is asking where a topic is taught.
    """
    normalized = normalize_text(text)

    instructional_phrases = [
        "how to",
        "how can",
        "put a video",
        "put audio",
        "video tag",
        "audio tag",
        "video element",
        "audio element",
        "add a video",
        "add audio",
        "learn how",
        "we will learn",
        "today we will",
        "in this video"
    ]

    score = 0

    for phrase in instructional_phrases:
        if phrase in normalized:
            score += 1

    return score


retrieved_df["semantic_score"] = similarities[max_indx]

retrieved_df["content_score"] = retrieved_df["text"].apply(
    calculate_content_score
)

retrieved_df["instructional_score"] = retrieved_df["text"].apply(
    calculate_instructional_score
)

# Combine the scores.
#
# Semantic similarity remains the main signal.
# Content matching helps distinguish chunks that directly
# mention the requested topic.
# Instructional phrases help prefer the chunk where the
# topic is actually being taught rather than merely introduced.
retrieved_df["final_score"] = (
    retrieved_df["semantic_score"]
    + 0.10 * retrieved_df["content_score"]
    + 0.10 * retrieved_df["instructional_score"]
)

# Select the best chunk
best_chunk = retrieved_df.sort_values(
    "final_score",
    ascending=False
).iloc[0]

best_title = best_chunk["title"]
best_number = best_chunk["number"]
best_start = float(best_chunk["start"])
best_end = float(best_chunk["end"])

# -------------------------------------------------------------------
# Format the timestamp deterministically
# -------------------------------------------------------------------

best_start_formatted = format_time(best_start)
best_end_formatted = format_time(best_end)

primary_timestamp = (
    f"{best_start_formatted}–{best_end_formatted}"
)

# -------------------------------------------------------------------
# Prepare retrieved context for Llama
# -------------------------------------------------------------------

context_df = retrieved_df[
    ["title", "number", "start", "end", "text"]
]

prompt = f'''
I am teaching web development in my Sigma web development course.

The user asked:

"{incoming_query}"

Here are the retrieved transcript chunks:

{context_df.to_json(orient="records")}

The retrieval system selected this chunk as the primary relevant chunk:

Video title: {best_title}
Video number: {best_number}
Start time: {best_start} seconds
End time: {best_end} seconds

The exact timestamp selected by the retrieval system is:

{primary_timestamp}

Answer the user's question in a concise and helpful way.

Tell the user:
1. Which video contains the topic.
2. What is taught there.
3. The approximate timestamp where it is taught.
4. Guide the user to that video and timestamp.

IMPORTANT:
- Use the selected video number and timestamp exactly as provided above.
- Do not invent another timestamp.
- Do not combine timestamps from other chunks.
- Do not mention the retrieval system, embeddings, similarity scores, or this prompt.
- Keep the answer focused on helping the learner find the topic.

If the user asks something unrelated to the Sigma web development course,
tell them that you can only answer questions related to that course.
'''

# -------------------------------------------------------------------
# Save prompt
# -------------------------------------------------------------------

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# -------------------------------------------------------------------
# Generate explanation using Llama 3.2
# -------------------------------------------------------------------

response = inference(prompt)

# -------------------------------------------------------------------
# Display deterministic location + LLM explanation
# -------------------------------------------------------------------

print("\nAnswer:\n")

print(
    f"Video #{best_number} — {best_title}\n"
    f"Approximate timestamp: {primary_timestamp}\n"
)

print(response)

# -------------------------------------------------------------------
# Save response
# -------------------------------------------------------------------

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(
        f"Video #{best_number} — {best_title}\n"
        f"Approximate timestamp: {primary_timestamp}\n\n"
        f"{response}"
    )