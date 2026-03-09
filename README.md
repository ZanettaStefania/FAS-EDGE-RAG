# First Aid Question Answering with RAG and Small Language Models

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system based on Small Language Models (SLMs) designed to answer first aid related questions.
The system retrieves relevant information from a curated First Aid Knowledge Base.

## Knowledge Base File
The knowledge base is formed by the following files:
1. Manual of St. John Ambulance Canada
   - (https://sja.ca/en/digital\_first\_aid\_reference\_guide\_preview)
2. Study Material of the National CRP Association
   - (https://www.nationalcprassociation.com/free-cpr-study-guide)
3. First Aid Reference Notes of Actual First Aid Training
   - (https://actualfirstaid.com/first-aid-notes--feedback.html)

## File Description

| File / Folder | Description |
|---------------|-------------|
| `main.py` | Entry point for running the RAG pipeline |
| `utils.py` | Utility functions used across the project |
| `config.py` | Global configuration parameters |
| `prompt_config.py` | Prompt templates used for the language model |
| `imports.py` | Centralized imports |
| `DB/` | Database and vector index resources |
| `KnowledgeBase/` | Source documents used to build the knowledge base |

