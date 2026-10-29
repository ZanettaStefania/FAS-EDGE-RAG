import argparse
intro_marker_phi = "You are an assistant for question-answering tasks about medical domain. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
context_marker_phi = "Context: "
end_context_marker_phi = f"""<|end|>
<
"""
answer_marker_phi = "|assistant|>"

intro_marker_qwen = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
context_marker_qwen = "Context: "
end_context_marker_qwen = f"""
<|im_end|>
<|im_start|>assistant<|im_end|>
"""
answer_marker_qwen = "<|im_start|>assistant<|im_end|>"

intro_marker = "Use a maximum of three sentences, and keep your answer concise, accurate, and where applicable, use medical terminology."
context_marker = "Context: "
end_context_marker = f"""
<|im_end|>
<|im_start|>assistant<|im_end|>
"""
answer_marker = "<|im_start|>assistant<|im_end|>"


qwen_rag_prompt_template = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are a knowledgeable assistant specializing in answering questions first aid procedures.Use the most relevant pieces of the retrieved context below to support your answer. If you don't know the answer or the context is insufficient, state that you are unsure. <|im_end|>""" + """
<|im_start|>user
Question: {question} 
Context: {context} 
<|im_end|>
<|im_start|>assistant<|im_end|>
"""

phi_rag_prompt_template = f"""<|system|> 
You are an assistant for question-answering tasks about medical domain. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<|end|>""" + """
<|user|>
Question: {input} 
Context: {context} <|end|>
<|assistant|>"""

model_path = ""
