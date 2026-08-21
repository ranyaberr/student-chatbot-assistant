import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly"
]

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def get_google_classroom_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    service = build(
        "classroom",
        "v1",
        credentials=creds
    )

    return service

#----------------------------
#          courses 
#----------------------------
def get_courses():
    service = get_google_classroom_service()

    results = service.courses().list(
        pageSize=10
    ).execute()

    return results.get("courses", [])

def get_classroom_context():
    courses = get_courses()

    if not courses:
        return "No Google Classroom courses were found."

    course_lines = []

    for course in courses:
        name = course.get("name", "Unnamed course")
        course_lines.append(f"- {name}")

    return "Current Google Classroom courses:\n" + "\n".join(course_lines)

#----------------------------
#       course work 
#----------------------------

def get_coursework():
    """
    Retrieve coursework (assignments) for all of the student's courses,
    along with submission status for each one.
    """
    service = get_google_classroom_service()
    courses = get_courses()

    all_coursework = []

    for course in courses:
        course_id = course.get("id")
        course_name = course.get("name", "Unnamed course")

        coursework_results = service.courses().courseWork().list(
            courseId=course_id
        ).execute()

        coursework_items = coursework_results.get("courseWork", [])

        for item in coursework_items:
            coursework_id = item.get("id")

            entry = {
                "course_id": course_id,
                "course_name": course_name,
                "coursework_id": coursework_id,
                "title": item.get("title", "Untitled assignment"),
            }

            if "description" in item:
                entry["description"] = item["description"]

            if "dueDate" in item:
                entry["due_date"] = item["dueDate"]

            if "dueTime" in item:
                entry["due_time"] = item["dueTime"]

            if "alternateLink" in item:
                entry["classroom_url"] = item["alternateLink"]

            # Retrieve the student's own submission for this coursework
            submissions_results = service.courses().courseWork().studentSubmissions().list(
                courseId=course_id,
                courseWorkId=coursework_id,
                userId="me"
            ).execute()

            submissions = submissions_results.get("studentSubmissions", [])

            if submissions:
                entry["submission_state"] = submissions[0].get("state")

            all_coursework.append(entry)

    return all_coursework


#----------------------------
#        pending tasks
#----------------------------
def get_pending_tasks():
    """
    Return only the coursework items that are NOT completed/submitted,
    based on Google Classroom's own submission state field.
    """
    coursework = get_coursework()

    # States that mean the assignment is NOT pending
    completed_states = {"TURNED_IN", "RETURNED"}

    pending = []

    for item in coursework:
        state = item.get("submission_state")

        # If we have no submission info at all, treat it as pending
        # (nothing was ever turned in)
        if state is None or state not in completed_states:
            pending.append(item)

    return pending 
