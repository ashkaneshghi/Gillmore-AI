from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from llama_index import GPTListIndex, LLMPredictor, GPTSimpleVectorIndex, download_loader, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
import streamlit as st
from streamlit_chat import message
import os
from llama_index import GPTSimpleVectorIndex, download_loader
from pathlib import Path
import openai

st.set_page_config(page_title='Gillmore AI', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# def get_text():
#     input_text = st.text_input("You: ", st.session_state["input"], key="input",
#                             placeholder="Please type your question here ...", 
#                             label_visibility='hidden')
#     return input_text

def get_text():
    input_text = st.text_input("You: ", "", key="input",
                            placeholder="Please type your question here ...", 
                            label_visibility='hidden')
    return input_text

def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
#     st.session_state["generated"] = []
#     st.session_state["past"] = []
    st.session_state["input"] = ""
    # st.session_state.entity_memory.store = {}
    # st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
# with st.sidebar.expander("🛠️ ", expanded=False):
#     # Option to preview memory store
#     if st.checkbox("Preview memory store"):
#         with st.expander("Memory-Store", expanded=False):
#             st.session_state.entity_memory.store
#     # Option to preview memory buffer
#     if st.checkbox("Preview memory buffer"):
#         with st.expander("Bufffer-Store", expanded=False):
#             st.session_state.entity_memory.buffer
#     #MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
#     K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)


st.title(" Gillmore AI ")
# st.subheader(" Presented by Gillmore Centre for Financial Technology")
# st.subheader(" Powered by Llama_Index + LangChain + OpenAI + Streamlit")

API_O = st.sidebar.text_input("API-KEY", type="password")

# Session state storage would be ideal
#if API_O:
    # Create an OpenAI instance
    # llm = OpenAI(temperature=0,
    #             openai_api_key=API_O, 
    #             model_name=MODEL, 
    #             verbose=False) 


    # Create a ConversationEntityMemory object if not already created
    # if 'entity_memory' not in st.session_state:
    #         st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    # Conversation = ConversationChain(
    #         llm=llm, 
    #         prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    #         memory=st.session_state.entity_memory
    #     )  
    
if API_O:
#     st.session_state.generated.append("Hello!")
    
#     %env OPENAI_API_KEY=API_O
#     open_api_key = os.getenv("OPENAI_API_KEY")
#     openai.api_key = API_O
    os.environ['OPENAI_API_KEY'] = API_O
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    st.stop() 

st.sidebar.button("New Chat", on_click = new_chat, type='primary')

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index_summaries = ['Fintech related papers' for i in range(0,11)]
index_summaries[0] = 'WITS-WBS Gillmore Centre Fintech Workshop & Industry Conference'
ind = {}
ind[0] = GPTSimpleVectorIndex.load_from_disk('WITS_Gillmore.json', service_context=service_context)    
for i in range(1,11):
    ind[i] = GPTSimpleVectorIndex.load_from_disk(f'IndPapersB{i}.json', service_context=service_context)    

graph = ComposableGraph.from_indices(
    GPTListIndex, 
    [ind[i] for i in range(0,11)], 
    index_summaries=index_summaries,
    service_context=service_context)

decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True)

query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 1,
                # "include_summary": True
            },
            "query_transform": decompose_transform
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
                "verbose": True
            }
        },
    ]

index_configs = [IndexToolConfig(
    index=ind[0], 
    name="Vector Index 1",
    description="useful for when you want to answer queries about Gillmore Centre Fintech Workshop & Industry Conference",
    index_query_kwargs={"similarity_top_k": 3},
    tool_kwargs={"return_direct": True})]
for i in range(1, 11):
    tool_config = IndexToolConfig(
        index=ind[i], 
        name=f"Vector Index {i+1}",
        description="useful for when you want to answer queries about fintech literature",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True})
    index_configs.append(tool_config)
    
graph_config = GraphToolConfig(
    graph=graph,
    name="Graph Index",
    description="useful for when you want to answer queries about anything.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True})

toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config])

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True)

def showres(prompt):
    output = agent_chain.run(input=prompt) 
    st.session_state.past.append(prompt)  
    st.session_state.generated.append(output)

user_input = get_text()

if user_input:
    showres(user_input)
    st.button("Send", on_click = showres(user_input), type='primary')
#     output = agent_chain.run(input=user_input) 
#     st.session_state.past.append(user_input)  
#     st.session_state.generated.append(output)  
#     message(st.session_state["generated"])
#     message(st.session_state['past'])
# if st.session_state['generated']:   
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
# def send(input):
#     output = res(input,API_O) 
#     st.session_state.past.append(input)  
#     st.session_state.generated.append(output)  

    
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

  

# st.button("New Chat", on_click = new_chat, type='primary')

# for i, sublist in enumerate(st.session_state.stored_session):
#         with st.sidebar.expander(label= f"Conversation-Session:{i}"):
#             st.write(sublist)

# if st.session_state.stored_session:   
#     if st.sidebar.checkbox("Clear-all"):
#         del st.session_state.stored_session
