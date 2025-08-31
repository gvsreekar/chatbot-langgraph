# langgraph_tool_thread_id_words.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
import requests
import os
import re

load_dotenv()
stockprice_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
model = ChatOpenAI(model='gpt-4o-mini')

# -------------------
# Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers (add, sub, mul, div).
    Returns a dict with operands, operation and result or an error message.
    """
    try:
        if operation == 'add':
            result = first_num + second_num
        elif operation == 'sub':
            result = first_num - second_num
        elif operation == 'mul':
            result = first_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {'error': str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a symbol using Alpha Vantage.
    Returns the JSON response (or an error dict if the API key is missing).
    """
    if not stockprice_api_key:
        return {"error": "Alpha Vantage API key not configured (ALPHAVANTAGE_API_KEY)"}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stockprice_api_key}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
model_with_tools = model.bind_tools(tools)

# -------------------
# State / graph
# -------------------
class chatbot_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_func(state: chatbot_state):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(chatbot_state)
graph.add_node('chatbot_func', chatbot_func)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chatbot_func')
graph.add_conditional_edges('chatbot_func', tools_condition)
graph.add_edge('tools', 'chatbot_func')
graph.add_edge('chatbot_func', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

# -------------------
# Thread title generation helper (NOT a tool)
# -------------------
def _sanitize_title(raw: str) -> str:
    title = re.sub(r'[^A-Za-z0-9\s-]', '', raw)
    title = re.sub(r'\s+', ' ', title).strip()
    words = title.split()
    if len(words) > 4:
        title = " ".join(words[:4])
    return title.lower()

def generate_thread_title(user_message: str) -> str:
    """
    Generate a concise 3-4 word title for a new conversation based on the first user message.
    Uses the same ChatOpenAI model; returns a sanitized lowercase title.
    """
    prompt = (
        "Produce a concise 3 to 4 word title describing a conversation whose first user message is:\n\n"
        f"\"{user_message}\"\n\n"
        "Respond with ONLY the title. No punctuation, no quotes, and no extra text."
    )
    try:
        resp = model.invoke([HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()
    except Exception:
        raw = " ".join(user_message.split()[:4])
    title = _sanitize_title(raw)
    if not title:
        import uuid
        title = str(uuid.uuid4())[:8]
    return title
