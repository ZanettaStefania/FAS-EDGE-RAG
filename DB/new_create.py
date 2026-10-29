import os

# Get process ID
pid = os.getpid()
print(pid)

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        

# Updated imports
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

# Load and process the text files
loader = DirectoryLoader('../PDF_files/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(len(texts))


#instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
instructor_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cuda"})

#instructor_embeddings = HuggingFaceBgeEmbeddings(model_name="/media/jetson/8822e6d5-68f8-44c2-8d88-adde671365d71/[download]Hug_model/Embedding/abhinand/MedEmbed-small-v0.1", model_kwargs={"device": "cuda"})


#/media/jetson/8822e6d5-68f8-44c2-8d88-adde671365d71/[download]Hug_model/Embedding/
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'new500_db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
                                 

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


