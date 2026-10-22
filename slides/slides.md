---
marp: true
theme: default
paginate: true
style: |
  section {
    background: linear-gradient(135deg, #ea580c 0%, #18181b 55%, #000000 100%);
    color: #fef08a;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    padding: 40px 60px;
  }
  h1, h2, h3 {
    color: #fef9c3;
  }
  h2 {
    font-size: 28px;
    border-bottom: 2px solid #fdba74;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  strong {
    color: #ffffff;
  }
  ul {
    margin-bottom: 20px;
  }
  li {
    margin-bottom: 8px;
  }
  code {
    background-color: rgba(0, 0, 0, 0.4);
    color: #fde047;
    padding: 2px 6px;
    border-radius: 4px;
  }
---

![bg right:40%](./s1.jpg)

## Foundations, Ingestion, and Vectorization

- **Key Phrases:** *"Building the Northstar Knowledge Base Foundation"*, *"Automated Ingestion and 500-Word Chunking"*, *"Local Vector Store via ChromaDB"*

The project established a technical foundation for the Northstar support platform. Markdown files from the knowledge base were systematically scanned, cleaned, and segmented using an optimized chunk size of 500 words and a sliding overlap of 50 words to maintain semantic continuity. 

These processed chunks were persisted into `data/processed_chunks.jsonl`. Subsequently, the local vector store was implemented using ChromaDB and the `all-MiniLM-L6-v2` embedding model, successfully vectorizing all documents for semantic indexing.

---

![bg left:40%](./s2.jpg)

## Semantic Retrieval and Generative Integration

- **Key Phrases:** *"Precision Similarity Search"*, *"Strict Grounding to Prevent Hallucinations"*, *"Mistral LLM Integration"*

Phase 3 introduced the semantic retriever, enabling the system to query the vector database and extract relevant chunks using vector similarity. 

Building upon this, Phase 4 integrated the Mistral generative model (`ministral-14b-latest`) via an OpenAI-compatible interface. A strict system prompt was enforced, ensuring the LLM answered user queries using *only* the retrieved context. If information was missing, the assistant safely declined rather than hallucinating, providing a professional and secure operational workflow.

---

![bg right:40%](./s3.jpg)

## End-to-End Pipeline and Interactive Deployment

- **Key Phrases:** *"Unified Master Script"*, *"Interactive Support Chat Experience"*, *"Scalable Python Architecture"*

The final phase unified all components into a comprehensive end-to-end master script (`rag_pipeline.py`). This script orchestrates the entire workflow: loading embeddings, connecting to ChromaDB, querying Mistral, and running an interactive console session for support operators. 

The assistant successfully handles real-time technical queries regarding account access, policies, and troubleshooting while maintaining strict traceability and robust error handling. The project is fully complete, modular, and structured entirely within the `scripts/` directory.