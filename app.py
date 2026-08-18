from agentic_chatbot_backend import chatbot , get_all_threads , ingest_rag_document
from langchain_core.messages import BaseMessage , HumanMessage , AIMessage , ToolMessage 
import streamlit as st
import uuid
import os
import tempfile
from langgraph.types import Command

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversations(thread_id):
    state = chatbot.get_state(
        config = {
            "configurable" : {
                "thread_id" : thread_id
            }
        }
    )
    return state.values.get("messages" , [])


def get_pending_interrupt(thread_id):
    config = {
        "configurable" : {
            "thread_id" : thread_id
        }
    }

    try:
        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:
            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id , interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id" : thread_id,
        "prompt" : str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:
        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if(
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pedning_hitl"] = None

def resume_hitl_execution(decision):

    pending_hitl  =st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or rejects."
        )

        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {
            "thread_id" : interrupted_thread_id
        },
        "run_name" : "hitl_resume_trace"
    }

    try:

        with st.chat_message("assistant"):

            status_holder = {
                "box" : st.status(
                    "Resuming the requested action...",
                    expanded = True
                )
            }

            def resumed_ai_only_stream():
                for message_chunk , metadata in chatbot.stream(
                    Command(resume = decision),
                    config = resume_config,
                    stream_mode = "messages",
                ):

                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name= getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label = f"Using '{tool_name}'...",
                            state = "running",
                            expanded = True,
                        )

                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label = "Another approval is required",
                    state = "complete",
                    expanded = False
                )

            else:
                st.session_state["pemding_hitl"] = None

                status_holder["box"].update(
                    label = "Action completed",
                    state = "complete",
                    expanded = False
                )

        if resumed_ai_message:
            st.session_state["message_history"].append({
                "role" : "assistant",
                "content" : resumed_ai_message
            })

        st.rerun()
    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )



st.set_page_config(
    page_title = "Agentic Chatbot",
    page_icon = ""
)

st.title("Agentic ChatBot with Langgraph")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()



if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])

st.sidebar.title("My Conversations")



if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()



for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(
        str(thread_id),
        key = thread_id
    ):
        st.session_state["thread_id"] = thread_id
        messages = load_conversations(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message , HumanMessage):
                role = "user"
            elif isinstance(message , AIMessage):
                role = "asistant"
            else:
                continue

            temp_messages.append({
                "role" : role,
                "content" : message.content
            })

        st.session_state["message_history"] = temp_messages
        st.rerun()




for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


submission = st.chat_input(
    "Type Here",
    accept_file = True,
    file_type = ["pdf"]
)


user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        temporary_file_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ".pdf"
            ) as temporary_file:
                temporary_file.write(
                    uploaded_pdf.get_value()
                )

                temporary_file_path = temporary_file.name

            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )

            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon = ""
            )

        except Exception as error:

            st.error(
                f"PDF processing failed: {error}"
            )
        finally:

            if(
                temporary_file_path and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)



if user_input:
    st.session_state["message_history"].append({"role" : "user" , "content" : user_input})
    with st.chat_message("user"):
        st.text(user_input)


    CONFIG = {
        "configurable" : {
            "thread_id" : st.session_state["thread_id"]
        }
    }


    
    
    with st.chat_message("assistant"):
        status_holder = {"box" : None}

        def ai_only_stream():
            for message_chunk , metadata in chatbot.stream(
                {"messages" : [HumanMessage(content = user_input)]},
                config = CONFIG,
                stream_mode = "messages",
            ):
                if isinstance(message_chunk , ToolMessage):
                    tool_name = getattr(message_chunk , "name" , "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using {tool_name}" , expanded = True
                        )
                    else:
                        status_holder["box"].update(
                            label = f" Using{tool_name}",
                            state = "running",
                            expanded = True,
                        )

                if isinstance(message_chunk , AIMessage):
                    yield message_chunk

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label = "Tool finished" , state = "complete" , expanded = False
            )


    st.session_state["message_history"].append({
        "role" : "assistant",
        "content" : "ai_message"
    })

    