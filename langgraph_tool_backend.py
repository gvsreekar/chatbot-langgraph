from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
import requests
import os 

load_dotenv()
stockprice_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
model = ChatOpenAI(model='gpt-4o-mini')

# Tools 
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str)->dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation=='add':
            result = first_num + second_num
        elif operation=='sub':
            result = first_num - second_num 
        elif operation=='mul':
            result = first_num*second_num
        elif operation=='div':
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {'error':str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stockprice_api_key}"
    r = requests.get(url)
    return r.json()

tools = [search_tool,calculator,get_stock_price]
model_with_tools = model.bind_tools(tools)

class chatbot_state(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chatbot_func(state: chatbot_state):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(chatbot_state)
graph.add_node('chatbot_func',chatbot_func)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chatbot_func')
graph.add_conditional_edges('chatbot_func',tools_condition)
graph.add_edge('tools','chatbot_func')
graph.add_edge('chatbot_func',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

# # testing this backend code
# config = {'configurable':{'thread_id':'thread-1'}}
# response = chatbot.invoke({'messages':[HumanMessage(content='What is my name?')]},
#                           config = config)
# print(response)
