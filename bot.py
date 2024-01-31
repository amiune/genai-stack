import os
import streamlit as st
import weaviate
import weaviate.classes as wvc

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

#client = weaviate.connect_to_local()

client = weaviate.connect_to_custom(
    http_host="weaviate",
    http_port=8080,
    http_secure=False,
    grpc_host="weaviate",
    grpc_port=50051,
    grpc_secure=False,
)

if client.collections.exists("Question"):
    client.collections.delete(name="Question")
    #questions = client.collections.get("Question")

#https://weaviate.io/developers/weaviate/starter-guides/custom-vectors
questions = client.collections.create(
    "Question",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        distance_metric=wvc.config.VectorDistance.COSINE
    ),
)


ollama_base_url = "http://host.docker.internal:11434"
embedding_model_name = "ollama"
llm_name = "llama2"

if embedding_model_name == "ollama":
    #https://python.langchain.com/docs/integrations/text_embedding/ollama
    embeddings = OllamaEmbeddings(
        base_url="http://host.docker.internal:11434", model="llama2"
    )
    dimension = 4096
else:
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
    )
    dimension = 384


question1_uuid = questions.data.insert(
            properties={
                "question": "Como esta el clima hoy?",
                "answer": "Muy lindo",
            },
            vector=embeddings.embed_query("Como esta el clima hoy?")
        )

question2_uuid = questions.data.insert(
            properties={
                "question": "De que color es el pasto?",
                "answer": "verde",
            },
            vector=embeddings.embed_query("De que color es el pasto?")
        )



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
You can only speak only in spanish!!
Don't answer any question in other language different from Spanish!!!
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

chain = chat_prompt | llm

rag_enabled = "No"
rag_enabled = st.radio("Usar RAG para contestar", ["No", "Si"], horizontal=True)

# Streamlit chat prompt
prompt = st.chat_input(placeholder="Escribe aqui tu pregunta")

if prompt:
    # Show the user prompt in the chat
    st.chat_message("user").write(prompt)
    
    # Run the chain
    if rag_enabled == "No":
        response = chain.invoke({"question": prompt})
        st.chat_message("assistant").write(response.content)
    else:
        response = questions.query.near_vector(
            near_vector=embeddings.embed_query(prompt),
            limit=2
        )
        r1 = f"La pregunta mas parecida en la BD es {response.objects[0].properties}"
        st.chat_message("assistant").write(r1)
        r2 = f"La segunda pregunta mas parecida en la BD es {response.objects[1].properties}"
        st.chat_message("assistant").write(r2)
    


#client.close()