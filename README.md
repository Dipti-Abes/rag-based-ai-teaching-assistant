# RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) based AI teaching assistant that helps learners find relevant topics, videos, and timestamps from a web development course.

The system converts course videos into audio, transcribes them using OpenAI Whisper, creates semantic embeddings using BGE-M3, retrieves the most relevant content using cosine similarity, and generates a contextual response using Llama 3.2 through Ollama.

## Features

- 🎥 Converts course videos into MP3 audio using FFmpeg
- 🎙️ Transcribes audio into text using OpenAI Whisper
- ✂️ Groups transcript segments into larger semantic chunks
- 🧠 Generates text embeddings using BGE-M3
- 🔎 Performs semantic search using cosine similarity
- 📚 Retrieves the top 5 most relevant course chunks
- 🤖 Generates answers using Llama 3.2 via Ollama
- ⏱️ Identifies the relevant course video and approximate timestamp
- 🚫 Handles questions unrelated to the course

## How It Works

The project follows the following pipeline:

```text
Course Videos
     │
     ▼
Video → MP3
     │
     │  FFmpeg
     ▼
Audio Transcription
     │
     │  Whisper
     ▼
Timestamped Transcript
     │
     ▼
Chunk Merging
     │
     ▼
BGE-M3 Embeddings
     │
     ▼
Semantic Retrieval
     │
     │  Cosine Similarity
     ▼
Top 5 Relevant Chunks
     │
     ▼
Context-Aware Prompt
     │
     ▼
Llama 3.2
     │
     │  Ollama
     ▼
Course-Specific Answer
+ Video Number
+ Timestamp

Project Architecture
1. Video to Audio

video_to_mp3.py

Converts course videos into MP3 files using FFmpeg because Whisper processes the extracted audio.

2. Speech-to-Text

mp3_to_json.py

Uses OpenAI Whisper (large-v2) to transcribe the audio and stores transcript segments along with:

Video number
Video title
Start timestamp
End timestamp
Transcript text
3. Chunk Processing

merge_chunks.py

Groups multiple transcript segments together to create larger chunks of text. This provides more context during semantic retrieval.

4. Embedding Generation

preprocess_json.py

Uses the locally hosted BGE-M3 embedding model through Ollama to convert transcript chunks into numerical vectors.

The resulting embeddings are stored using joblib.

5. Semantic Retrieval and Answer Generation

process_incoming.py

When a user asks a question:

The question is converted into an embedding using BGE-M3.
Cosine similarity is calculated between the question embedding and stored course embeddings.
The top 5 most relevant chunks are retrieved.
The retrieved chunks are added to a context-aware prompt.
Llama 3.2 generates the final response through Ollama.
The response identifies the relevant video and approximate timestamp.
Tech Stack
Technology	Purpose
Python	Core programming language
OpenAI Whisper	Speech-to-text transcription
Pandas	Data processing
NumPy	Numerical computation
Scikit-learn	Cosine similarity
Joblib	Saving/loading embeddings
Requests	Communication with Ollama API
BGE-M3	Text embeddings
Llama 3.2	Response generation
Ollama	Local model serving
FFmpeg	Audio extraction
Project Structure
RAG-Based-AI-Teaching-Assistant/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── video_to_mp3.py
├── mp3_to_json.py
├── merge_chunks.py
├── preprocess_json.py
├── process_incoming.py
└── speech_to_text.py

Course videos, audio files, transcript data, generated embeddings, and temporary output files are excluded from version control.

Prerequisites

Before running the project, install:

Python 3.x
FFmpeg
Ollama

The project uses the following Ollama models:

bge-m3
llama3.2
Installation
1. Clone the repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd RAG-Based-AI-Teaching-Assistant
2. Install Python dependencies
pip install -r requirements.txt
3. Install Ollama models

Pull the required models:

ollama pull bge-m3
ollama pull llama3.2

Verify the installed models:

ollama list
Running the Pipeline

The project is designed as a sequential processing pipeline.

Step 1: Convert videos to audio

Place your legally obtained course videos inside:

Sample Videos/

Then run:

python video_to_mp3.py

This generates MP3 files inside:

Audios/
Step 2: Transcribe the audio

Run:

python mp3_to_json.py

This uses Whisper to generate timestamped transcript data.

Step 3: Merge transcript chunks

Run:

python merge_chunks.py

This combines multiple transcript segments into larger chunks.

Step 4: Generate embeddings

Run:

python preprocess_json.py

This generates BGE-M3 embeddings using Ollama and stores them in:

embeddings.joblib
Step 5: Ask questions

Run:

python process_incoming.py

Then enter a question related to the course:

Ask a Question: Where is video and audio in HTML taught?

The system retrieves relevant transcript chunks and generates a response containing the relevant video and approximate timestamp.

Example
Input
Where is video and audio in HTML taught?
Output
Video #10 — Video, Audio & Media in HTML

The relevant content is covered in Video #10.
Go to the indicated timestamp to find the section discussing
video and audio elements in HTML.

The exact response and timestamp depend on the retrieved transcript chunks and model output.

Why RAG?

Instead of asking the language model to answer from its general knowledge, this project first retrieves relevant information from the course transcripts.

This helps the assistant provide answers grounded in the available course material and makes it useful for navigating long educational content.

Limitations
The current interface is command-line based.
Retrieval currently uses cosine similarity over stored embeddings rather than a dedicated vector database.
The system retrieves the top 5 chunks for each query.
Course-specific relevance filtering is currently handled through the prompt rather than a separate similarity threshold or classifier.
The project requires locally available course content to reproduce the complete preprocessing pipeline.
Response quality depends on the quality of the transcript, retrieved chunks, and local language model.
Future Improvements
Add a similarity threshold for better irrelevant-query detection
Replace the current embedding storage approach with a vector database
Build a web interface using Streamlit or Flask
Provide clickable video timestamps
Add conversation history
Improve retrieval using hybrid keyword + semantic search
Add evaluation metrics for retrieval and answer quality
Allow users to upload their own educational videos
Learning Outcomes

Through this project, I worked with:

Retrieval-Augmented Generation (RAG)
Natural Language Processing
Text embeddings
Semantic search
Cosine similarity
Speech-to-text transcription
Local Large Language Models
Prompt engineering
Data preprocessing
Python-based ML pipelines
Acknowledgement

This project was developed as part of the Ultimate Job Ready Data Science Course by CodeWithHarry.

The project was used as a learning exercise to understand RAG pipelines, embeddings, semantic retrieval, speech-to-text processing, and local LLM inference.

License

This repository is intended for educational and portfolio purposes.

Course videos, audio, transcripts, and other third-party course materials are not included in this repository.
