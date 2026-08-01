# InternGuide — RAG-Based Document Q&A Chatbot

**BetaBytez AI/ML Internship — Task 4**
**Author:** Sabiha Metlo
**Repo:** `betabytez-aiml-task4-sabiha`

---

## Overview

InternGuide is a Retrieval-Augmented Generation (RAG) chatbot that answers questions **only** from a specific set of documents I provided — the BetaBytez TaskBook, the Guidelines PDF, and my own Task 1–3 READMEs. Unlike a general-purpose chatbot, it does not answer from its own training knowledge. If the answer isn't in the documents, it says so instead of guessing.

The interface is a custom dark-themed Streamlit chat app with a sidebar for managing multiple saved conversations (star, rename, delete), styled to resemble a modern chat product.

---

## Document Set Chosen

I used four sources, combined into a single knowledge base:

1. **AI-ML TaskBook PDF** — the official task descriptions for all 5 tasks
2. **Guidelines PDF** — program structure, submission rules, evaluation criteria
3. **My Task 1, 2, and 3 READMEs** — my own project documentation (`.md` files)

**Why this set:** I already had these documents on hand, understood their content deeply (making it easy to write honest, verifiable test questions), and it created a genuinely useful chatbot — one that can answer questions about both the internship program itself and my own past project work. This also let me test retrieval across two different document types (PDF and Markdown) rather than just one.

---

## Chunking Strategy

- **Chunk size:** 300 characters
- **Chunk overlap:** 50 characters
- **Splitter:** `RecursiveCharacterTextSplitter` (LangChain)

**Reasoning:** My documents are short (a few-page TaskBook, short READMEs), not long-form text, so large chunks weren't necessary. I estimated that a reasonable chunk should hold roughly 2–4 sentences — enough to capture one complete idea (like a single task's tech stack or a single deliverable) without pulling in unrelated content. At ~80–120 characters per sentence, that put me in the low hundreds of characters, so I chose 300. The 50-character overlap (~15% of chunk size) helps ensure a sentence split across a chunk boundary still appears whole in at least one chunk.

**Known trade-off:** This chunk size sometimes split a single logical answer across two chunks (see Limitations below), and the TaskBook's continuous numbered list (deliverables numbered 1–28 across all 5 tasks, not restarting per task) occasionally confused the model about which task a chunk belonged to.

---

## Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (via `HuggingFaceEmbeddings`)
- **Why:** Free and runs entirely locally — no API key, no billing, no rate limits. Given my 2-day timeline, this removed all signup/cost friction. It's a well-established, lightweight model that's more than sufficient for a small, single-domain document set like mine.

---

## Vector Database

- **Choice:** FAISS (`faiss-cpu`), via LangChain's `FAISS.from_documents()`
- **Why:** Free, local, no server setup required (unlike ChromaDB's optional server mode), and simple to save/load from disk (`save_local()`), which fit a short deadline well.

---

## LLM

- **Provider:** Groq API
- **Model:** `llama-3.1-8b-instant`
- **Why:** Groq offers a genuinely free tier with fast inference, and LangChain has a direct integration (`langchain-groq`) — no complex setup, and fast enough for a responsive chat experience.

**Grounding mechanism:** The LLM is instructed via prompt to answer *only* using the retrieved context, and to explicitly say "I don't have information about that in the provided documents" if the answer isn't present. After discovering a hallucination during testing (see below), I added an additional instruction: *"Do not infer numbers, task names, or facts that are not explicitly and clearly stated in the context."*

---

## Testing — 5 Questions (3 In-Scope, 2 Out-of-Scope)

### ✅ In-scope questions (answerable from the documents)

**Q1: What is the tech stack for the RAG chatbot task?**
> A: The tech stack for the RAG chatbot task is Python, LangChain or LlamaIndex, FAISS or ChromaDB, LLM API, and Streamlit.

Correct and fully grounded — matched the Task 4 tech line in the TaskBook exactly.

**Q2: Which model performed best for house price prediction, Random Forest, Gradient Boosting, or Linear Regression?**
> A: Gradient Boosting.

Correct — pulled directly from my Task 3 README's model comparison section, including the reasoning (lowest error, highest R²).

**Q3: How many weeks does the BetaBytez internship last?**
> A: The internship has a duration of 6 weeks.

Correct — pulled directly from the Guidelines PDF.

### ✅ Out-of-scope questions (should NOT be answerable)

**Q4: What is the capital of France?**
> A: I don't have information about that in the provided documents.

Correctly refused. Notably, the retrieved chunks were completely unrelated (Task 5 tech stack, Ames Housing dataset info) — the model did not fall back on its own general knowledge to answer "Paris," even though it clearly "knows" that fact from training. This is the grounding mechanism working as intended.

**Q5: What is BetaBytez's refund policy for interns?**
> A: I don't have information about that in the provided documents.

Correctly refused, even though the retrieved chunks (certification requirements, professionalism notes) were topically closer to "program policy" than the France question — making this a harder case where a weaker grounding prompt might have blended those chunks into a fabricated policy answer.

---

## Known Limitations (Honest Findings)

**1. Ambiguous/numeric queries sometimes retrieved the wrong task section.**
Asking *"What tech stack is used for Task 4?"* initially retrieved Task 5's tech stack instead, because the embedding model weighs overall semantic meaning more heavily than exact numbers — "Task 4" and "Task 5" content are otherwise very similar in phrasing. Rephrasing the query to use vocabulary unique to Task 4 (e.g., "RAG chatbot," "LangChain," "FAISS") reliably fixed this. **Takeaway:** basic similarity search does not guarantee exact keyword/number precision.

**2. A numbered list caused a genuine hallucination.**
When asked *"in which tasks have we used FastAPI,"* the chatbot answered *"tasks 1 and 24"* — but Task 24 does not exist (the program only has 5 tasks). The TaskBook uses a continuous numbered list of deliverables spanning all tasks (item "24." was actually just the 24th bullet point in that list, not a task number). The model misread that number as a task reference.

I addressed this by strengthening the grounding prompt with an explicit instruction not to infer numbers or facts not clearly stated in the context. After the fix, the same question no longer invented a fake task number — but it did still misattribute the correct deliverable to the wrong task ("Task 1" instead of the actual source). This shows the prompt fix reduced but did not fully eliminate the underlying issue, which is a genuine limitation of chunk-level context isolation at this chunk size — a single 300-character chunk doesn't always carry enough surrounding context for the model to know which task section it came from.

**3. Chunk boundaries can split a single answer.**
A query about my Task 3 model results initially retrieved a chunk that stopped right before the actual model comparison numbers, because that content had been split into the next chunk over. Rephrasing with more specific vocabulary from my own README recovered the correct chunk.

---

## Tech Stack

- **Python** — core language
- **LangChain** (`langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-groq`) — document loading, chunking, embeddings, LLM integration
- **FAISS** — vector database
- **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** — embeddings (local, free)
- **Groq API (`llama-3.1-8b-instant`)** — LLM
- **Streamlit** — chat UI

---

## Project Structure

```
betabytez-aiml-task4-sabiha/
├── data/                          # Source documents (PDFs + READMEs)
├── faiss_index/                   # Saved vector store (generated, gitignored)
├── app.py                         # Main Streamlit application
├── style.css                      # Custom UI styling
├── .env                           # Groq API key (gitignored, not committed)
├── .gitignore
└── README.md
```

---

## How to Run

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   python -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-groq faiss-cpu pypdf streamlit python-dotenv
   ```
3. Create a `.env` file with your own Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run the app:
   ```
   python -m streamlit run app.py
   ```
5. Open `http://localhost:8501` in your browser.

---

## What I Learned

This task was my first time building a RAG pipeline end-to-end. Beyond the individual concepts (embeddings, vector search, grounding prompts), the most valuable lesson was seeing firsthand how retrieval quality directly determines answer quality — no amount of prompt engineering fully compensates for the wrong chunks being retrieved in the first place. Debugging the "Task 24" hallucination in particular taught me that grounding prompts reduce but don't eliminate errors rooted in how documents are chunked and indexed.