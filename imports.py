## ADDED IMPORTS
import torch

## END

import warnings
import argparse
from langchain._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Suppress transformers warnings
from transformers import logging
logging.set_verbosity_error()


import textwrap


import pandas as pd


import os

# Updated imports
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from config import qwen_rag_prompt_template

## RERANK
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker 

from langchain_huggingface import HuggingFacePipeline
## END RERANK
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


