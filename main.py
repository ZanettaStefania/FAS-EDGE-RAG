from imports import * 
from utils import *
from config import *
import gc

# Get process ID
pid = os.getpid()
#print(pid)

# Parse the model name input from command line
args = parse_args()
if args.test is not None and not os.path.exists(args.test):
	args.test = os.path.join("EvalSet", args.test)
#print(args.test)
#print(args.test is None)

qwen_rag_prompt_template = process_prompt(args)

model_id=inizialize(args)
#print(model_id)

# Load the existing vector database instead of creating a new one
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY_DB, 
		  embedding_function=HuggingFaceBgeEmbeddings(model_name=DB_EMBD_MODEL_NAME, model_kwargs={"device": DEVICE_MAP}))

retriever = vectordb.as_retriever(search_type=SEARCH_TYPE_RETRIEVER, search_kwargs={"k": RETRIVED_K, "score_threshold": SIMILARITY_THRESHOLD})

#retriever = vectordb.as_retriever(search_kwargs={"k": RETRIVED_K})

model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=model, top_n=RETRIEVER_TOP_N)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor, base_retriever=retriever
)

#model_id = "/media/jetson/8822e6d5-68f8-44c2-8d88-adde671365d71/[download]Hug_model/Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)



pipe = pipeline(
          MODEL_REPLY,
          model=model,
          tokenizer=tokenizer,
          max_new_tokens=MAX_NEW_TOKENS,
          clean_up_tokenization_spaces=True,
          device_map=DEVICE_MAP
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Create the retrieval chain
combine_docs_chain = create_stuff_documents_chain(local_llm, PromptTemplate(input_variables=["context", "question"], template=qwen_rag_prompt_template))

# Create the full RAG chain with the compression retriever and the combine_docs_chain
rag_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)


if args.test is None:
	while True:
		q=input("How can I help you?\n>>")
		if q=="exit":
			break
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		#context_, answer_ = split_string(llm_response['answer'])
		#process_llm_response(llm_response)
		#dict_keys(['question', 'input', 'context', 'answer'])
		
		#print("Context:\n", type(llm_response['context']))
		#print("Context:\n", len(llm_response['context']))
		#print("Context:\n", llm_response['context'][0])
		#print("Context:\n", llm_response['context'][1])
		#print("Context:\n", llm_response['context'][0].metadata)
		print("question:\n", llm_response['question'])
		print("answer:\n", split_answer(llm_response['answer']))
		#print("input:\n", llm_response['input'])
		#print("Context:\n", context_)
		#print("\nAnswer:\n", answer_)
		print("--------------------------------------------------------------------")


### ------------------------------------------------------
### WHEN GIVEN A CSV FILE WITH QUESTIONS
else:
	data = pd.read_csv(args.test)
	#print(len(data))
	for n in range(0, len(data)):
		q=data['Question'][n]
		g=data['ground_truths'][n]
		print("Question: \n", data['Question'][n])
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		#print("LLM_Response : ", llm_response['answer'])
		#process_llm_response(llm_response)
		#context_, answer_ = split_string(llm_response['answer'])
		#print("Time: ", (time.time() - start))
		#process_question(model=args.model_name, question=n, answer=answer_, context=context_, time=(time.time() - start), data=data)
		print("n==len(data) :", n==len(data))
		print("n :", n)
		print("len(data) :", len(data))
		results_file = store_answer(db=PERSIST_DIRECTORY_DB, model=args.model_name, file_name=args.test, llm_response=llm_response, time=(time.time() - start), ground_truth_data=g, save_file=(True if n==(len(data)-1) else False))
		#print("Context:\n", context_)
		#print("\nAnswer:\n", answer_)
		print("\nAnswer:\n", split_answer(llm_response['answer']))
		print("--------------------------------------------------------------------")
		#break
	# Release the generator before loading the judge.
	del rag_chain, combine_docs_chain, local_llm, pipe, model, tokenizer
	del compressor, compression_retriever, retriever, vectordb
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	from ragas_evaluation import evaluate_ragas
	evaluate_ragas(results_file, DB_EMBD_MODEL_NAME)

