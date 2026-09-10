from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    # Generate embeddings using Ollama + BGE-M3
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    r.raise_for_status()

    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    # Generate response using Ollama + Llama 3.2
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    r.raise_for_status()

    response = r.json()
    return response["response"]

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])[0]

# Find similarities between the question and stored course embeddings
similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

# Retrieve top 5 most relevant chunks
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_indx]

prompt = f'''I am teaching web development in my Sigma web development course.
Here are video subtitle chunks containing video title, video number,
start time in seconds, end time in seconds, and the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

-----------------------------

"{incoming_query}"

The user asked this question related to the video chunks.

Answer in a human and helpful way. Tell the user:
1. Where the relevant content is taught.
2. Which video it is in.
3. The approximate timestamp where it is taught.
4. Guide the user to go to that particular video.

If the user asks something unrelated to the Sigma web development course,
tell them that you can only answer questions related to that course.
'''

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

print("\nAnswer:\n")
print(response)