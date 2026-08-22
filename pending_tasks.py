import datetime
import streamlit as st
from google_classroom import get_pending_tasks

@st.cache_data(ttl=30, show_spinner= False)
def get_pending_tasks_safe():
    try:
        tasks = get_pending_tasks()

        print("DEBUG - Google Classroom returned:", len(tasks), "pending tasks")

        return tasks, None
    except Exception as e :
        print("DEBUG - Google Classroom error:", e)
        return None, str(e)
def refresh_pending_tasks():
    """
    fetch fresh pending tasks from google classroom 
    """

    
    get_pending_tasks_safe.clear()

    return get_pending_tasks_safe()

# ---------------------------------------------------------
# FORMAT PENDING TASKS FOR THE LLM
# ---------------------------------------------------------

def get_pending_tasks_context():
    """
    fetch pending tasks (cached up to 30s) and format them for the LLM. 
    """

    tasks, error = get_pending_tasks_safe()

    if error:
        return None, error

    if not tasks:
        return "The student currently has no pending tasks.", None

    context = []
    for task in tasks:
        course = task.get("course_name", "Unknown course")
        title = task.get("title", "Untitled assignment")
        due_date = format_due_date(task)
        category = categorize_task(task)

        context.append(
            f"- Course: {course} | "
            f"Assignment: {title} | "
            f"Due: {due_date} | "
            f"Status: {category}"
        )

    return "\n".join(context), None




def parse_due_date(task):
    due = task.get("due_date")

    if not due:
        return None

    try:
        return datetime.date(
            due["year"],
            due["month"],
            due["day"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def categorize_task(task):
    due_date = parse_due_date(task)

    if due_date is None:
        return "upcoming"

    today = datetime.date.today()

    if due_date < today:
        return "overdue"
    elif due_date == today:
        return "due_today"
    else:
        return "upcoming"


def format_due_date(task):
    due_date = parse_due_date(task)

    if due_date is None:
        return "No due date"

    return due_date.strftime("%B %d, %Y")


def render_pending_tasks_view():
    st.title("📋 Pending Tasks")

    if st.button("←"):
        st.session_state.show_pending_tasks = False
        st.rerun()

    tasks, error = get_pending_tasks_safe()
    if not error:
        st.session_state.pending_count = len(tasks)

    if error:
        st.error(error)
        return

    if not tasks:
        st.write("No pending tasks. You're all caught up! 🎉")
        return

    overdue = [
        t for t in tasks
        if categorize_task(t) == "overdue"
    ]

    due_today = [
        t for t in tasks
        if categorize_task(t) == "due_today"
    ]

    upcoming = [
        t for t in tasks
        if categorize_task(t) == "upcoming"
    ]

    st.write(
        f"**{len(tasks)} unfinished assignment"
        f"{'s' if len(tasks) != 1 else ''}**"
    )

    st.markdown(
        f"🔴 {len(overdue)} overdue &nbsp;&nbsp; "
        f"🟠 {len(due_today)} due today &nbsp;&nbsp; "
        f"📅 {len(upcoming)} upcoming",
        unsafe_allow_html=True
    )

    icons = {
        "overdue": "🔴",
        "due_today": "🟠",
        "upcoming": "📅"
    }

    labels = {
        "overdue": "Overdue",
        "due_today": "Due today",
        "upcoming": "Upcoming"
    }

    for task in overdue + due_today + upcoming:
        category = categorize_task(task)

        with st.container(border=True):
            st.markdown(
                f"**{icons[category]} "
                f"{task.get('course_name', 'Unknown course')}**"
            )

            st.write(
                task.get("title", "Untitled assignment")
            )

            st.caption(
                f"{labels[category]} • "
                f"{format_due_date(task)}"
            )

            url = task.get("classroom_url")

            if url:
                st.markdown(
                    f"[Open in Classroom ↗]({url})"
                )