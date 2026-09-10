# RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) based AI teaching assistant that helps learners find relevant topics, videos, and timestamps from a web development course.

The system converts course videos into audio, transcribes them using Whisper, creates semantic embeddings using BGE-M3, retrieves the most relevant content using cosine similarity, and generates contextual responses using Llama 3.2 through Ollama.

> **Note:** Whisper is used locally for speech-to-text transcription. This project does not require the OpenAI API or paid API credentials.

## Features

- 🎥 Converts course videos into MP3 audio using FFmpeg
- 🎙️ Transcribes audio using Whisper (`large-v2`)
- ⏱️ Preserves timestamps for transcript segments
- ✂️ Groups transcript segments into larger contextual chunks
- 🧠 Generates semantic embeddings using BGE-M3
- 🔎 Performs semantic search using cosine similarity
- 📚 Retrieves the top 5 most relevant course chunks
- 🤖 Generates answers using Llama 3.2
- 🖥️ Runs LLM inference locally through Ollama
- 🎯 Identifies the relevant course video and approximate timestamp
- 🚫 Handles questions unrelated to the course context

## How It Works

The project follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
Course Videos
      │
      ▼
Video → MP3
   (FFmpeg)
      │
      ▼
Speech-to-Text
   (Whisper)
      │
      ▼
Timestamped Transcripts
      │
      ▼
Chunk Merging
      │
      ▼
BGE-M3 Embeddings
      │
      ▼
Semantic Retrieval
(Cosine Similarity)
      │
      ▼
Top 5 Relevant Chunks
      │
      ▼
Context-Aware Prompt
      │
      ▼
Llama 3.2
   (Ollama)
      │
      ▼
Course-Specific Answer
 + Video Number
 + Timestamp
```

## Project Architecture

### 1. Video to Audio

**File:** `video_to_mp3.py`

Converts course videos into MP3 audio files using FFmpeg.

### 2. Speech-to-Text

**File:** `mp3_to_json.py`

Uses Whisper (`large-v2`) locally to transcribe the audio and stores timestamped transcript segments containing:

- Video number
- Video title
- Start timestamp
- End timestamp
- Transcript text

### 3. Chunk Processing

**File:** `merge_chunks.py`

Groups multiple transcript segments together to create larger chunks of text for improved retrieval context.

### 4. Embedding Generation

**File:** `preprocess_json.py`

Uses the BGE-M3 embedding model through Ollama to convert transcript chunks into numerical vectors.

The generated embeddings are stored using Joblib.

### 5. Semantic Retrieval and Answer Generation

**File:** `process_incoming.py`

When a user asks a question:

1. The question is converted into an embedding using BGE-M3.
2. Cosine similarity is calculated between the question embedding and stored course embeddings.
3. The top 5 most relevant transcript chunks are retrieved.
4. The retrieved chunks are added to a context-aware prompt.
5. Llama 3.2 generates the response through Ollama.
6. The response identifies the relevant course video and approximate timestamp.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Whisper | Speech-to-text transcription |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Cosine similarity |
| Joblib | Saving and loading embeddings |
| Requests | Communication with Ollama |
| BGE-M3 | Text embeddings |
| Llama 3.2 | Response generation |
| Ollama | Local LLM and embedding model serving |
| FFmpeg | Audio extraction |

## Project Structure

```text
RAG-Based-AI-Teaching-Assistant/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── video_to_mp3.py
├── mp3_to_json.py
├── merge_chunks.py
├── preprocess_json.py
└── process_incoming.py
```

### Generated and Local Files

The following files and folders are intentionally excluded from version control:

```text
Audios/
Sample Videos/
jsons/
newjsons/
embeddings.joblib
prompt.txt
response.txt
```

These files are either generated during processing, contain course-derived data, or consist of large media files.

## Prerequisites

Before running the project, install:

- Python 3.x
- FFmpeg
- Ollama

The project uses the following Ollama models:

```text
bge-m3
llama3.2
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dipti-Abes/rag-based-ai-teaching-assistant.git
cd rag-based-ai-teaching-assistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the required Ollama models

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Verify the installed models:

```bash
ollama list
```

## Running the Pipeline

The project is designed as a sequential processing pipeline.

### Step 1 — Convert Videos to Audio

Place legally obtained course videos inside:

```text
Sample Videos/
```

Then run:

```bash
python video_to_mp3.py
```

This generates MP3 files inside:

```text
Audios/
```

### Step 2 — Generate Transcripts

Run:

```bash
python mp3_to_json.py
```

Whisper generates timestamped transcript data from the extracted audio.

### Step 3 — Merge Transcript Chunks

Run:

```bash
python merge_chunks.py
```

This combines multiple transcript segments into larger contextual chunks.

### Step 4 — Generate Embeddings

Run:

```bash
python preprocess_json.py
```

This generates BGE-M3 embeddings using Ollama and stores them in:

```text
embeddings.joblib
```

### Step 5 — Ask Questions

Run:

```bash
python process_incoming.py
```

Then enter a question related to the course.

## Example

### Question

```text
Where is video and audio in HTML taught?
```

### Result

```text
Video #10 — Video, Audio & Media in HTML

The relevant content is taught approximately around
00:54–01:12.
```

The assistant identifies the relevant course video and approximate timestamp based on the retrieved transcript chunks.

## Why RAG?

A language model alone may not know the exact content or structure of a particular educational course.

This project uses **Retrieval-Augmented Generation (RAG)** to first retrieve relevant information from course transcripts and then provide that information to the language model as context.

This helps the assistant generate answers grounded in the available course material and makes it easier for learners to navigate long educational content.

## Limitations

- The current interface is command-line based.
- Retrieval uses cosine similarity over stored embeddings rather than a dedicated vector database.
- The system retrieves the top 5 chunks for each query.
- Course-specific relevance filtering is currently handled through the prompt rather than a separate similarity threshold or classifier.
- Reproducing the complete preprocessing pipeline requires locally available course content.
- Response quality depends on transcript quality, retrieved context, and the local language model.

## Future Improvements

- Add a similarity threshold for improved irrelevant-query detection
- Replace the current embedding storage approach with a vector database such as FAISS or Chroma
- Build a web interface using Streamlit or Flask
- Add clickable video timestamps
- Add conversation history
- Implement hybrid keyword and semantic search
- Add retrieval and answer-quality evaluation metrics
- Allow users to upload their own educational videos

## Learning Outcomes

Through this project, I gained practical experience with:

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- Text embeddings
- Semantic search
- Cosine similarity
- Speech-to-text processing
- Local Large Language Models
- Prompt engineering
- Data preprocessing
- Python-based ML pipelines

## Acknowledgement

This project was developed as part of the **Ultimate Job Ready Data Science Course by CodeWithHarry**.

The project was used as a learning exercise to understand RAG pipelines, embeddings, semantic retrieval, speech-to-text processing, and local LLM inference.

## Disclaimer

This repository is intended for educational and portfolio purposes.

Course videos, audio files, transcripts, and other third-party course materials are **not included** in this repository.