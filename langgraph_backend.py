from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

class chatbot_state(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')
checkpointer = InMemorySaver()
graph = StateGraph(chatbot_state)

def chatbot_func(state: chatbot_state):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages':[response]}

graph.add_node('chatbot_func',chatbot_func)
graph.add_edge(START,'chatbot_func')
graph.add_edge('chatbot_func',END)

chatbot = graph.compile(checkpointer=checkpointer)

