import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
from langchain_openai import ChatOpenAI
import uuid
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory,GetSessionHistoryCallable
llm = ChatOpenAI(model="gpt-4o-mini")


if "store" not in st.session_state:
    st.session_state["store"] = {}
store = st.session_state["store"]

if "id" not in st.session_state:
        st.session_state["id"]= []
        st.session_state["id"].append(str(uuid.uuid4()))
        st.session_state["session_id"] = st.session_state['id'][-1]
session_id = st.session_state["session_id"]

if "sumarisation" not in st.session_state:
    st.session_state["sumarisation" ] = {session_id:None}


def get_session_history(session_id:str)-> BaseChatMessageHistory:
    if session_id not in store.keys():
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# if "result" not in st.session_state:
#     st.session_state["result"]= llm.invoke([SystemMessage(content="you are a help full agent")])
# result = st.session_state['result']

# if session_id not in store.keys():
#     store[session_id] = ChatMessageHistory(messages=[SystemMessage(content="you are a help full agent"),result])

# with_history = RunnableWithMessageHistory(llm,get_session_history) 

input = st.chat_input("enter something")

# for message in store[session_id].messages:
#     type = message.type
#     if type == 'system':
#         with st.chat_message(f"human"):
#             st.write(message.content)
#     else:
#         with st.chat_message(f"{type}"):
#             st.write(message.content)

# # input = st.chat_input("enter something")
# if input != None:
#     output = with_history.invoke(input = [HumanMessage(content = input )],
#         config = {'configurable': {'session_id': session_id}})
#     # store[session_id].add_messages([HumanMessage(content = input ),output])
#     with st.chat_message("user"):
#         st.write(input)

#     with st.chat_message("ai"):
#         st.write(output.content)
if "test" not in st.session_state:
    st.session_state["test"] = "ok"
    if "result" not in st.session_state:
        st.session_state["result"]= llm.invoke([SystemMessage(content="you are a help full agent")])
    result = st.session_state['result']

    if session_id not in store.keys():
        store[session_id] = ChatMessageHistory(messages=[SystemMessage(content="you are a help full agent"),result])

    with_history = RunnableWithMessageHistory(llm,get_session_history) 




with st.sidebar:
    if st.button("new chat",help="click me for new chat"):
        if st.session_state["sumarisation"][session_id] == None :
            del st.session_state["sumarisation"][session_id]
        # if "session_id" not in st.session_state:
        st.session_state["id"].append(str(uuid.uuid4()))
        st.session_state["session_id"] = st.session_state["id"][-1]
        session_id = st.session_state["session_id"]

        st.session_state["sumarisation"][session_id] = None
    # session_id = st.selectbox("select your chat",st.session_state['id'][::-1])
    summarisation = st.selectbox("select your chat",list(st.session_state["sumarisation"].values())[::-1])
    # summarisation = st.selectbox("select your chat",list(st.session_state["sumarisation"].values()))
    st.write(list(st.session_state["sumarisation"].values()))
    if summarisation != None:
        for summari in st.session_state["sumarisation"].keys():
            if st.session_state["sumarisation"][summari] == summarisation:
                session_id = summari
        # session_id = st.session_state["sumarisation"][summarisation]
#____
if "result" not in st.session_state:
    st.session_state["result"]= llm.invoke([SystemMessage(content="you are a help full agent")])
result = st.session_state['result']

if session_id not in store.keys():
    store[session_id] = ChatMessageHistory(messages=[SystemMessage(content="you are a help full agent"),result])

with_history = RunnableWithMessageHistory(llm,get_session_history) 

for message in store[session_id].messages:
    type = message.type
    if type == 'system':
        with st.chat_message(f"human"):
            st.write(message.content)
    else:
        with st.chat_message(f"{type}"):
            st.write(message.content)

# input = st.chat_input("enter something")
if input != None:
    output = with_history.invoke(input = [HumanMessage(content = input )],
        config = {'configurable': {'session_id': session_id}})
    # store[session_id].add_messages([HumanMessage(content = input ),output])
    with st.chat_message("user"):
        st.write(input)

    with st.chat_message("ai"):
        st.write(output.content)
    if None in st.session_state["sumarisation"].values():
        prompt = ChatPromptTemplate.from_messages([
            ("system","you are a helpful assistant who makes a very short and presise of human message for a streamlit select bar"),
            ("user","here is the chat \n<chat>\n\n----\n{chat}\n\n----\n")
        ])
        chain = prompt | llm 
        result = chain.invoke({"chat":store[st.session_state["session_id"]].messages[-2]})
        st.session_state["sumarisation"].update({st.session_state["id"][-1] : result.content})
        st.write(st.session_state["sumarisation"])


st.write(st.session_state["id"])
st.write(session_id)
