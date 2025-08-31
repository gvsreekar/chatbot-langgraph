# streamlit_tool_frontend.py
import streamlit as st
from langgraph_tool_thread_id_words import chatbot, retrieve_all_threads, generate_thread_title
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id(user_message=None):
    """
    If user_message is provided, call backend.generate_thread_title to get a human-readable id.
    Otherwise, fallback to a random uuid string.
    """
    if user_message:
        title = generate_thread_title(user_message)
        return title
    return str(uuid.uuid4())

def reset_chat():
    # Start a fresh session without creating a thread id yet.
    st.session_state["thread_id"] = None
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if not thread_id:
        return
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # state.values is expected to be a dict-like; messages may not exist yet
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    # None means no title yet — will generate on first user message
    st.session_state["thread_id"] = None

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads() or []

# If there's a current thread_id (e.g., from previous session), ensure it's listed
if st.session_state["thread_id"]:
    add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
# show most recent first
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================
# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # If this is the first user message in this session (no thread_id assigned),
    # generate a descriptive thread title and ensure uniqueness.
    if not st.session_state.get("thread_id"):
        base_title = generate_thread_id(user_input)
        # make unique if same title already exists
        candidate = base_title
        i = 1
        while candidate in st.session_state["chat_threads"]:
            candidate = f"{base_title} ({i})"
            i += 1
        st.session_state["thread_id"] = candidate
        add_thread(candidate)

    # Show and store user's message locally
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Mutable holder so generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # When a tool runs, show a status container (create lazily)
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status if used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message locally to show in UI
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
