DB_USERNAME = "postgres"
DB_PASSWORD = "12345678"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "testdb"

from sqlalchemy import create_engine
base_url = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(base_url)
from langchain_community.utilities.sql_database import SQLDatabase

db = SQLDatabase(engine=engine)
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini',api_key=os.environ["OPENAI_API_KEY"])
llm.invoke("what is capital of india")
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db = db , llm = llm)
tools = toolkit.get_tools()
llm_with_tools= llm.bind_tools(tools)
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

class State(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]

def llm_tool (state:State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages":[result]}

from pydantic import BaseModel,Field
from typing import Literal

    
class Checker(BaseModel):
    result : Literal["yes","no"] = Field(description=("Return 'yes' if the user intends to end the conversation. "
            "Examples include messages like 'Exit', 'Bye', 'Quit', 'Stop', 'exit is working', etc. "
            "Return 'no' otherwise."))

# def checker(state:State):
#     llm_withstructure = llm.with_structured_output(Checker)
#     result = llm_withstructure.invoke(state["messages"])
#     if result.result == 'yes':
#         return "__end__"
#     return "llm_tool"

def checker(state: State):
    # Get the last user message content
    last_message = state["messages"][-1].content.strip().lower()

    # List of common exit phrases
    exit_phrases = ["exit", "quit", "bye", "stop", "end", "close", "terminate", "goodbye"]

    # If user's message clearly indicates they want to exit
    if any(phrase in last_message for phrase in exit_phrases):
        return "__end__"

    # Else let LLM handle nuanced understanding
    llm_withstructure = llm.with_structured_output(Checker)
    result = llm_withstructure.invoke(state["messages"])
    return "__end__" if result.result == "yes" else "llm_tool"


def tool_condition(state:State,messages_key = "messages"):
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "node_3"
    

def node_3(state:State):
    return {"messages":state["messages"]}

from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition

graph_builder = StateGraph(State)

graph_builder.add_node("llm_tool",llm_tool)
graph_builder.add_node("node_3",node_3)
graph_builder.add_node(ToolNode(toolkit.get_tools()))

graph_builder.add_edge(START,"llm_tool")
graph_builder.add_conditional_edges("llm_tool",tool_condition,{"tools":"tools","node_3":"node_3"})
graph_builder.add_edge("tools","llm_tool")
graph_builder.add_conditional_edges("node_3",checker,{"llm_tool":"llm_tool","__end__":END})


from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()

graph = graph_builder.compile(checkpointer=memory,interrupt_after=["node_3"])

from langchain_core.messages import SystemMessage,HumanMessage
config = {"configurable":{"thread_id":"test_03"}}
f = open("database schema.txt",'r')
schema = f.read()
from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_prompt = prompt_template.invoke({"dialect":"postgresql","top_k":5})
for event in graph.stream(State(messages=[SystemMessage(content=
    "You are a restaurant management assistant. "
    "Your role is to help users interact with the restaurant's PostgreSQL database. "
    "You can perform tasks like checking reservations, adding or updating bookings, managing guests, "
    "and answering questions using SQL tools. "
    "Always prefer querying the database to find information. "
    "If the database does not contain enough information to answer a question, ask the user for more details. "
    "Do not make assumptions. Always ensure the data is correct before responding. "
    "If the user says 'Exit', end the conversation."
)]+system_prompt.messages + [HumanMessage(content=input("eneter something:"))]),config,stream_mode="value"):
    print(event)

state=graph.get_state(config)
print(state.next)
print(state.values["messages"])

while(state.next != ()):
    state=graph.get_state(config)
    print(state.next)
    for message in state.values["messages"]:
        message.pretty_print()
    graph.update_state(config,MessagesState(messages=[HumanMessage(content = input("enter something:"))]))
    for event in graph.stream(None,config,stream_mode="value"):
        print(event)

# state=graph.get_state(config)
# print(state)
    # SELECT table_name
    # FROM information_schema.tables
    # WHERE table_type = 'BASE TABLE'
    # AND table_schema NOT IN ('pg_catalog', 'information_schema');