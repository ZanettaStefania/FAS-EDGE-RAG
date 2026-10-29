# Point to the existing database
PERSIST_DIRECTORY_DB = 'DB/new1500_db'
questions_csv_file='questions_ground_truths_25.csv'
DB_EMBD_MODEL_NAME="BAAI/bge-small-en-v1.5"
SEARCH_TYPE_RETRIEVER="similarity_score_threshold"
SIMILARITY_THRESHOLD=0.6
RETRIVED_K=5
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
RETRIEVER_TOP_N=2
MAX_NEW_TOKENS=1024
MODEL_REPLY="text-generation"
DEVICE_MAP='cuda'
