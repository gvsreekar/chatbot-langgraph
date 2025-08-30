from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sqlite3

class chatbot_state(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(chatbot_state)

def chatbot_func(state: chatbot_state):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages':[response]}

graph.add_node('chatbot_func',chatbot_func)
graph.add_edge(START,'chatbot_func')
graph.add_edge('chatbot_func',END)

chatbot = graph.compile(checkpointer=checkpointer)

# # testing this backend code
# config = {'configurable':{'thread_id':'thread-1'}}
# response = chatbot.invoke({'messages':[HumanMessage(content='What is my name?')]},
#                           config = config)
# print(response)
