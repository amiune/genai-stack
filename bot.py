import os
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

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
from langchain.schema.messages import AIMessage

ollama_base_url = "http://127.0.0.1:11434"
embedding_model_name = "ollama"
llm_name = "phi"

if embedding_model_name == "ollama":
    embeddings = OllamaEmbeddings(
        base_url=ollama_base_url, model=llm_name
    )
    dimension = 4096
else:
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
    )
    dimension = 384

llm = ChatOllama(
        temperature=0,
        base_url=ollama_base_url,
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

# we need some other things to be global, though 
# this one must go first
st.set_page_config(page_title="Searchable")

# Create containers for chat history and user input
response_container = st.container()
container = st.container()


def init_all_things():
    
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm the all knowing useles bot 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = []


def show_all_things():
    # sidebar
    with st.sidebar:
        st.title('Searchable')
        # some other things here that we might want to add
        st.markdown('''
        ## About
        LLM + RAG powered search engine
        ''')
        add_vertical_space(5)
        st.write('Made with ❤️ by [searchable](https://github.com/leomrocha/searchable)')

    # User input form
    with container:
        with st.form(key='chat-input', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="... (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # Run the chain
            response = chain.invoke({"question": user_input})
            print(type(response), response )
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)
    # `st.chat_input()` can't be used inside an `st.expander`, `st.form`, `st.tabs`, `st.columns`, or `st.sidebar`.
    # if user_input := st.chat_input(placeholder="Escribe aqui tu pregunta"):
    #     # Run the chain
    #     # so here goes all the logic that we should have to make the agent work nicely
    #     response = chain.invoke({"question": user_input})
    #     st.session_state['past'].append(user_input)
    #     st.session_state['generated'].append(response)

    # Display chat history
    with response_container:
        for i in range(len(st.session_state['generated'])):
            ai_msg = st.session_state['generated'][i]
            if isinstance(ai_msg, AIMessage):
                ai_msg = ai_msg.content
            # print(type(ai_msg), ai_msg)
            message(ai_msg, key=str(i), avatar_style="thumbs")
            if i < len(st.session_state['past']):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")


if __name__ == "__main__":
    init_all_things()
    show_all_things()