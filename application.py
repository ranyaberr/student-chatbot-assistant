from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st
from config import MODEL_NAME, SYSTEM_PROMPT
from google_classroom import get_google_classroom_service, get_classroom_context
from conversations import (
    load_conversations,
    save_conversations,
    make_title,
    create_new_conversation,
    ensure_conversation_fields
)

from memory import (
    load_memories,
    extract_memory_from_exchange,
    store_new_memories,
    retrieve_relevant_memories
)

from pending_tasks import( 
    get_pending_tasks_safe, 
    get_pending_tasks_context,
    refresh_pending_tasks,
    render_pending_tasks_view
    )



st.set_page_config(page_title="Student Chatbot Assistant", page_icon="🎓")
# ---------------------------------------------------------
#          Load the API key from the .env file
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "conversations.json")


# ---------------------------------------------------------
# CONVERSATION SUMMARY: generate/update per-conversation summaries
# ---------------------------------------------------------
def generate_conversation_summary(conversation):
    """
    Create or update a short summary of the conversation using the LLM.
    Called after each exchange so the summary always reflects the
    latest state of the conversation.
    """
    real_messages = conversation["messages"][1:]  # skip system prompt
    if not real_messages:
        return ""

    transcript = "\n".join(
        [f"{m['role']}: {m['content']}" for m in real_messages]
    )

    previous_summary = conversation.get("summary", "")

    summary_prompt = f"""Summarize the following conversation between a student and an assistant.

The summary must capture:
- the main topic
- important things discussed
- important questions or problems
- important conclusions or decisions
- context needed to understand what happened

Do NOT include greetings, small talk, or a message-by-message account.
Keep it concise (1-3 sentences).

Previous summary (if any): "{previous_summary}"

Full conversation so far:
{transcript}

Respond ONLY with the updated summary text. No labels, no quotes, no extra formatting.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}]
    )

    return response.choices[0].message.content.strip()

def get_previous_conversation_summary(conversations, active_id):
    """
    Return the summary of the conversation that comes right before the
    active one, based on creation order (dict insertion order).
    Excludes the active conversation itself. Returns None if there is
    no previous conversation or it has no summary.
    """
    conversation_ids = list(conversations.keys())

    if active_id not in conversation_ids:
        return None

    active_index = conversation_ids.index(active_id)

    # Look backwards from the active conversation for the nearest
    # conversation that has a non-empty summary
    for i in range(active_index - 1, -1, -1):
        prev_id = conversation_ids[i]
        prev_conv = conversations[prev_id]
        if prev_conv.get("summary"):
            return prev_conv["summary"], prev_conv.get("title", "")

    return None

def question_needs_previous_conversation(current_question):
    """
    Ask the LLM if the question is referring to a previous/past
    conversation (e.g. "what did we talk about last time?").
    Returns True/False.
    """
    check_prompt = f"""Does this student message refer to a PREVIOUS conversation
(e.g. asking what was discussed before, asking to continue an earlier topic,
referring to "last time", "yesterday", "before", "previously")?

Message: "{current_question}"

Respond ONLY with "yes" or "no".
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": check_prompt}]
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")

def build_system_prompt_with_context(current_question, conversations, active_id):
    """
    Build the system prompt, injecting:
    - relevant student memories (memories.json)
    - the previous conversation's summary, ONLY if the question
      seems to refer to a past conversation
    """
    prompt = SYSTEM_PROMPT

    # ---- Student memory (durable facts) ----
    all_memories = load_memories()
    relevant_memories = retrieve_relevant_memories(
        current_question,
        all_memories,
        client,
        MODEL_NAME
    )
    if relevant_memories:
        memory_block = "\n".join([f"- {fact}" for fact in relevant_memories])
        prompt += f"""

You have the following relevant background information about this student
from previous conversations. Use it naturally when helpful, without
explicitly mentioning that you "remembered" it unless it fits naturally:
{memory_block}"""

    # ---- Previous conversation summary (only if relevant) ----
    if question_needs_previous_conversation(current_question):
        result = get_previous_conversation_summary(conversations, active_id)
        if result:
            prev_summary, prev_title = result
            prompt += f"""

The student is asking about a previous conversation. Here is a summary of
the most recent earlier conversation (titled "{prev_title}"):
"{prev_summary}"

Use this to answer naturally. Do NOT say you are stateless or that you
don't remember previous conversations — you have this summary available."""
        else:
            prompt += """

The student is asking about a previous conversation, but no earlier
conversation with a summary exists. Honestly tell them there is no
previous conversation on record."""

    return prompt



def question_is_about_assignments(question):
    keywords = [
        "assignment",
        "assignments",
        "homework",
        "task",
        "tasks",
        "pending",
        "deadline",
        "deadlines",
        "due",
        "overdue",
        "devoir",
        "devoirs",
        "travail",
        "travaux",
        "tâche",
        "tâches",
        "date limite"
    ]

    question = question.lower()

    return any(keyword in question for keyword in keywords)

# ---------------------------------------------------------
# STEP 4: Load conversations from disk ONCE per session
# ---------------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = ensure_conversation_fields(load_conversations())

if len(st.session_state.conversations) == 0:
    new_id = create_new_conversation(st.session_state.conversations)
    st.session_state.active_conversation_id = new_id
    save_conversations(st.session_state.conversations)

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = list(st.session_state.conversations.keys())[-1]

# ---------------------------------------------------------
# STEP 5: Sidebar — New Chat button + list of conversations
# ---------------------------------------------------------

with st.sidebar:

    st.title("🎓 Student Assistant")

# ---------------------------------------------------------
# CSS for Pending Tasks bar
# ---------------------------------------------------------
    st.markdown("""
    <style>
    .st-key-pending_tasks_bar_container {
        position: relative;
        margin-bottom: 10px;
    }

    .st-key-pending_tasks_bar_container button {
        width: 100% !important;
        background-color: rgba(128,128,128,0.12) !important;
        border: 1px solid rgba(128,128,128,0.3) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        text-align: left !important;
        font-size: 14px !important;
    }

    .pending-dot {
        position: absolute;
        top: -4px;
        right: -4px;
        width: 10px;
        height: 10px;
        background-color: #ff4b4b;
        border-radius: 50%;
       
        z-index: 5;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    #                      pending tasks 
    #----------------------------------------------------------
   

    if "pending_count" not in st.session_state:
        pending_tasks, pending_error = get_pending_tasks_safe()

        st.session_state.pending_count = (
            len(pending_tasks) if pending_tasks else 0
        )

    pending_count = st.session_state.pending_count

    if "show_pending_tasks" not in st.session_state:
        st.session_state.show_pending_tasks = False

    # The container itself comes BEFORE New Chat
    with st.container(key="pending_tasks_bar_container"):

        # Red notification dot
        if pending_count > 0:
            st.markdown(
                '<div class="pending-dot"></div>',
                unsafe_allow_html=True
            )

        # REAL clickable button
        if st.button(
           f"📋 Pending Tasks        {pending_count}",
            key=   "open_pending_tasks_btn",
            use_container_width=True
):

            pending_tasks, pending_error = refresh_pending_tasks()

            st.session_state.pending_tasks = pending_tasks
            st.session_state.pending_tasks_error = pending_error

            st.session_state.pending_count = (
                len(pending_tasks) if pending_tasks else 0
    )

            st.session_state.show_pending_tasks = True
            st.rerun()

    # ---------------------------------------------------------
    # New Chat
    # ---------------------------------------------------------

    if st.button("New Chat", use_container_width=True):
        new_id = create_new_conversation(
            st.session_state.conversations
        )

        st.session_state.active_conversation_id = new_id

        save_conversations(
            st.session_state.conversations
        )

        st.rerun()


    # ---------------------------------------------------------
    # Recent Conversations
    # ---------------------------------------------------------

    st.markdown("### Recent Conversations")

    conversation_ids = list(
        st.session_state.conversations.keys()
    )

    for conversation_id in reversed(conversation_ids):

        conversation = st.session_state.conversations[
            conversation_id
        ]

        title = conversation["title"]

        is_active = (
            conversation_id
            == st.session_state.active_conversation_id
        )

        renaming_key = f"renaming_{conversation_id}"

        if renaming_key not in st.session_state:
            st.session_state[renaming_key] = False


        col1, col2 = st.columns([6, 1])


        # -----------------------------
        # delete - rename button
        # -----------------------------

        with col1:

            if st.session_state[renaming_key]:

                new_title = st.text_input(
                    "Rename",
                    value=title,
                    key=f"rename_input_{conversation_id}",
                    label_visibility="collapsed"
                )

            else:

                if st.button(
                    title,
                    key=f"conv_{conversation_id}"
                ):
                    st.session_state.active_conversation_id = (
                        conversation_id
                    )

                    st.rerun()


        # -----------------------------
        # Three-dot menu
        # -----------------------------

        with col2:

            with st.popover("⋮"):

                if st.session_state[renaming_key]:

                    if st.button(
                        "Save",
                        key=f"confirm_rename_{conversation_id}"
                    ):

                        new_value = st.session_state[
                            f"rename_input_{conversation_id}"
                        ].strip()

                        if new_value:
                            conversation["title"] = new_value

                            save_conversations(
                                st.session_state.conversations
                            )

                        st.session_state[renaming_key] = False

                        st.rerun()


                    if st.button(
                        "Cancel",
                        key=f"cancel_{conversation_id}"
                    ):

                        st.session_state[renaming_key] = False

                        st.rerun()

                else:

                    if st.button(
                        "Rename",
                        key=f"edit_{conversation_id}"
                    ):

                        st.session_state[renaming_key] = True

                        st.rerun()


                    if st.button(
                        "Delete",
                        key=f"delete_{conversation_id}"
                    ):

                        del st.session_state.conversations[
                            conversation_id
                        ]


                        if (
                            conversation_id
                            == st.session_state.active_conversation_id
                        ):

                            remaining_ids = list(
                                st.session_state.conversations.keys()
                            )

                            if remaining_ids:

                                st.session_state.active_conversation_id = (
                                    remaining_ids[-1]
                                )

                            else:

                                new_id = create_new_conversation(
                                    st.session_state.conversations
                                )

                                st.session_state.active_conversation_id = (
                                    new_id
                                )


                        save_conversations(
                            st.session_state.conversations
                        )

                        st.rerun()


    # ---------------------------------------------------------
    # Google Classroom
    # ---------------------------------------------------------

    if st.button("Connect to Google Classroom"):

        service = get_google_classroom_service()

        st.success("Google Classroom connected!")


# ---------------------------------------------------------
# Display Pending Tasks OR Chat
# ---------------------------------------------------------

if st.session_state.show_pending_tasks:

    render_pending_tasks_view()

else:

    # ---------------------------------------------------------
    # Main chat area
    # ---------------------------------------------------------

    st.title("Student Chatbot Assistant")

    st.write(
        "Ask me anything about programming, math, CS, "
        "engineering, or your studies!"
    )

    active_id = st.session_state.active_conversation_id

    active_conversation = (
        st.session_state.conversations[active_id]
    )

    for message in active_conversation["messages"][1:]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])
# ---------------------------------------------------------
# STEP 7: Handle new user input
# ---------------------------------------------------------
user_question = st.chat_input("Type your question here...")

if user_question:
    active_conversation["messages"].append({"role": "user", "content": user_question})

    if active_conversation["title"] == "New conversation":
        active_conversation["title"] = make_title(user_question)

    with st.chat_message("user"):
        st.markdown(user_question)

    # Build system prompt with student memory + previous-conversation summary if relevant
    dynamic_system_prompt = build_system_prompt_with_context(
    user_question,
    st.session_state.conversations,
    active_id
)

    if question_is_about_assignments(user_question):
        pending_tasks_context, pending_error = get_pending_tasks_context()

        if pending_error:
            pending_tasks_context = (
                f"Unable to retrieve pending tasks: {pending_error}"
            )

        # Add Classroom information to the LLM context
        dynamic_system_prompt += f"""

    Here is live information from the student's Google Classroom:

    {pending_tasks_context}

    Use this information when the student's question is related
    to their courses or university activities.

    This information is live external data. Do not treat it as
    permanent student memory.
    """

    messages_for_llm = [{"role": "system", "content": dynamic_system_prompt}] + \
    active_conversation["messages"][1:]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_llm
            )
            bot_reply = response.choices[0].message.content
            st.markdown(bot_reply)

    active_conversation["messages"].append({"role": "assistant", "content": bot_reply})

    # Update this conversation's summary to reflect the latest state
    active_conversation["summary"] = generate_conversation_summary(active_conversation)

    save_conversations(st.session_state.conversations)

    # Extract and store durable student facts (unchanged system)
    new_facts = extract_memory_from_exchange(user_question, bot_reply)
    store_new_memories(new_facts, active_id)

