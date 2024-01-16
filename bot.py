import os
import streamlit as st

from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings
)
from langchain.chat_models import ChatOllama
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

ollama_base_url = "http://host.docker.internal:11434"
embedding_model_name = "ollama"
llm_name = "llama2"

if embedding_model_name == "ollama":
    embeddings = OllamaEmbeddings(
        base_url="http://host.docker.internal:11434", model="llama2"
    )
    dimension = 4096
else:
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
    )
    dimension = 384

llm = ChatOllama(
        temperature=0,
        base_url="http://host.docker.internal:11434",
        model=llm_name,
        streaming=True,
        # seed=2,
        top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=3072,  # Sets the size of the context window used to generate the next token.
    )

template = """
You are a helpful assistant that helps to answer human questions.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

chain = chat_prompt | llm

# Streamlit chat prompt
if prompt := st.chat_input(placeholder="Escribe aqui tu pregunta"):

    # Show the user prompt in the chat
    st.chat_message("user").write(prompt)
    
    # Run the chain
    response = chain.invoke({"question": prompt})
    
    st.chat_message("assistant").write(response)
