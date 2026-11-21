from imports import * 
from utils_med import *
from config_med import *
from langchain_core.runnables import RunnableLambda
from results_io import FirstTokenTimer, append_timing, mark_ragas_skipped
import gc

PIPELINE_NAME = os.path.splitext(os.path.basename(__file__))[0]

logger = get_logger(__name__)

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

vectordb = None
if args.no_retrieval:
	logger.info("Retrieval disabled: generating answers without context")
else:
	# Load the existing vector database instead of creating a new one
	vectordb = Chroma(persist_directory=PERSIST_DIRECTORY_DB, 
			  embedding_function=HuggingFaceBgeEmbeddings(model_name=DB_EMBD_MODEL_NAME, model_kwargs={"device": DEVICE_MAP}))

compressor = None
if not args.no_rerank and not args.no_retrieval:
	model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
	compressor = CrossEncoderReranker(model=model, top_n=RETRIEVER_TOP_N)

# Per-question step durations (seconds), reset before each question.
step_times = {}

def retrieve_and_log(inputs):
	question = inputs["input"]
	t0 = time.perf_counter()
	scored_documents = vectordb.similarity_search_with_relevance_scores(
		question, k=RETRIVED_K
	)
	step_times["retrieval_s"] = time.perf_counter() - t0
	logger.info(
		f"Retrieval before threshold: {len(scored_documents)} candidates "
		f"(k={RETRIVED_K}, threshold={SIMILARITY_THRESHOLD})"
	)

	selected = []
	for rank, (document, score) in enumerate(scored_documents, start=1):
		keep = score >= SIMILARITY_THRESHOLD
		logger.info(
			f"\n--- Chunk {rank}/{len(scored_documents)} | score={score:.6f} | "
			f"{'kept' if keep else 'below threshold'} | "
			f"source={document.metadata.get('source', 'unknown')} ---\n"
			f"{document.page_content}\n--- End chunk {rank} ---"
		)
		if keep:
			selected.append(document)

	logger.info("Passed threshold: %s/%s", len(selected), len(scored_documents))
	if args.no_rerank:
		kept = [(score, document) for document, score in scored_documents if score >= SIMILARITY_THRESHOLD]
		kept.sort(key=lambda item: item[0], reverse=True)
		kept = kept[:RETRIEVER_TOP_N]
		logger.info("Reranking disabled: using top %s chunks by similarity (scores=%s)",
			len(kept), [round(score, 6) for score, _ in kept])
		return [document for _, document in kept]
	documents = []
	if selected:
		t0 = time.perf_counter()
		documents = compressor.compress_documents(selected, question)
		step_times["reranking_s"] = time.perf_counter() - t0
	logger.info("After reranking: %s chunks", len(documents))
	return documents


compression_retriever = RunnableLambda(retrieve_and_log)

#model_id = "/media/jetson/8822e6d5-68f8-44c2-8d88-adde671365d71/[download]Hug_model/Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)



pipe = pipeline(
          MODEL_REPLY,
          model=model,
          tokenizer=tokenizer,
          streamer=(ttft_timer := FirstTokenTimer()),
          max_new_tokens=MAX_NEW_TOKENS,
          clean_up_tokenization_spaces=True,
          device_map=DEVICE_MAP
)

local_llm = HuggingFacePipeline(pipeline=pipe)

def timed_generation(inputs):
	t0 = time.perf_counter()
	answer = combine_docs_chain.invoke(inputs)
	step_times["generation_s"] = time.perf_counter() - t0
	return answer

if args.no_retrieval:
	combine_docs_chain = PromptTemplate.from_template(qwen_rag_prompt_template) | local_llm
	# Same output shape as create_retrieval_chain, with an empty context.
	rag_chain = RunnableLambda(lambda inputs: {**inputs, "context": [], "answer": timed_generation(inputs)})
else:
	# Create the retrieval chain
	combine_docs_chain = create_stuff_documents_chain(local_llm, PromptTemplate(input_variables=["context", "question"], template=qwen_rag_prompt_template))

	# Create the full RAG chain with the compression retriever and the combine_docs_chain
	rag_chain = create_retrieval_chain(compression_retriever, RunnableLambda(timed_generation))


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
		logger.info("Question:\n%s", llm_response['question'])
		logger.info("Answer:\n%s", split_answer(llm_response['answer']))
		#print("input:\n", llm_response['input'])
		#print("Context:\n", context_)
		#print("\nAnswer:\n", answer_)
		logger.info("--------------------------------------------------------------------")
else:
	data = pd.read_csv(args.test)
	#print(len(data))
	for n in range(0, len(data)):
		q=data['Question'][n]
		g=data['ground_truths'][n]
		log_event(f"Generation {n + 1}/{len(data)} started: {q}")
		step_times.clear()
		ttft_timer.reset()
		start = time.perf_counter()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		elapsed_seconds = time.perf_counter() - start
		#print("LLM_Response : ", llm_response['answer'])
		#process_llm_response(llm_response)
		#context_, answer_ = split_string(llm_response['answer'])
		#print("Time: ", (time.time() - start))
		#process_question(model=args.model_name, question=n, answer=answer_, context=context_, time=(time.time() - start), data=data)
		logger.info("n==len(data): %s", n == len(data))
		logger.info("n: %s", n)
		logger.info("len(data): %s", len(data))
		result_model = args.model_name + "_med" + ("_no_rerank" if args.no_rerank else "") + ("_no_retrieval" if args.no_retrieval else "")
		append_timing({"model": result_model, "pipeline": PIPELINE_NAME,
			"dataset": os.path.basename(args.test), "question": q, "total_s": elapsed_seconds,
			"ttft_s": None if ttft_timer.first_token_at is None else ttft_timer.first_token_at - start,
			**{key: step_times.get(key) for key in ("retrieval_s", "reranking_s", "generation_s")}})
		results_file = store_answer(db=PERSIST_DIRECTORY_DB, model=result_model, file_name=args.test, llm_response=llm_response, time=elapsed_seconds, ground_truth_data=g, save_file=(True if n==(len(data)-1) else False))
		log_event(
			f"Generation {n + 1}/{len(data)} completed in {elapsed_seconds:.3f}s"
		)
		#print("Context:\n", context_)
		#print("\nAnswer:\n", answer_)
		logger.info("Answer:\n%s", split_answer(llm_response['answer']))
		logger.info("--------------------------------------------------------------------")
		#break
	# Release generator resources before the optional evaluation.
	del rag_chain, combine_docs_chain, local_llm, pipe, model, tokenizer
	del compressor, compression_retriever, vectordb
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if args.skip_ragas:
		mark_ragas_skipped(results_file)
		logger.info("Ragas evaluation skipped: %s", results_file)
	else:
		from ragas_evaluation import evaluate_ragas
		evaluate_ragas(results_file, DB_EMBD_MODEL_NAME, use_context=not args.no_retrieval)
