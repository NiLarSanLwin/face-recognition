import io
import time
from datetime import date, timedelta
from config import WORK_START_TIME
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from config import (FACES_DIR, MATCH_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS,
                    WORK_START_TIME, WORK_END_TIME, LATE_GRACE_MINUTES, MIN_HOURS_BEFORE_AUTO_CHECKOUT)
from database import (
    init_db, add_employee, get_employees, add_embedding, get_all_embeddings,
    record_check_in, record_face_attendance, record_check_out, get_attendance,
    get_missing_checkouts, attendance_summary, set_employee_active,
    log_recognition_attempt, recognition_statistics, recent_attendance_activity,
    weekly_attendance_counts, attendance_distribution,
    ensure_default_admin, authenticate_user, get_employee_today_status,
    update_employee, delete_employee
)
from face_engine import FaceEngine, BlinkLivenessDetector

st.set_page_config(page_title="Face Recognition Attendance", page_icon="🧑‍💼", layout="wide")
init_db()
ensure_default_admin()
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Traditional Myanmar-inspired burgundy and royal-gold visual theme.
def apply_theme():
    st.markdown(
        """
       
<style>
    :root {
        --royal-burgundy: #D63384;
        --deep-maroon: #D63384;
        --royal-gold: #D9A02E;
        --warm-ivory: #FFF3D8;
        --soft-gold: #F3D28A;
        --background-blue: #DCEEFF;
    }

    .stApp {
        background: #DCEEFF !important;
        color: var(--deep-maroon);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF4FF 0%, #DCEEFF 62%, #C9E2FF 100%);
        border-right: 2px solid var(--royal-gold);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--royal-burgundy) !important;
        text-shadow: 0 1px 8px rgba(217,160,46,.22);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--deep-maroon);
    }

    h1, h2, h3, h4 {
        color: var(--royal-burgundy) !important;
    }

    p, label, [data-testid="stCaption"] {
        color: var(--deep-maroon);
    }

    [data-testid="stMetric"],
    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.75);
        border: 1px solid rgba(217,160,46,.75);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--royal-burgundy) !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #E8B94E, #C58A1D) !important;
        color: #2B0000 !important;
        border: 1px solid #F5D98E !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #F5D98E !important;
        border-color: #FFF3D8 !important;
    }

    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input {
        background: #FFFFFF !important;
        color: #3B1700 !important;
        border-color: var(--royal-gold) !important;
    }

    [data-testid="stRadio"] label {
        color: var(--deep-maroon) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(217,160,46,.7);
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(87,0,0,.25);
    }
    
/* Streamlit Top Toolbar Background */
.stAppHeader {
    background: #DCEEFF !important;
}

/* Toolbar Container */
[data-testid="stToolbar"],
[data-testid="stToolbarActions"] {
    background: #DCEEFF !important;
}

/* Header Action Buttons */
[data-testid="stBaseButton-header"],
[data-testid="stMainMenuButton"] {
    background: #DCEEFF !important;
    color: #570000 !important;
    border: none !important;
}

/* Toolbar Button Hover */
[data-testid="stBaseButton-header"]:hover,
[data-testid="stMainMenuButton"]:hover {
    background: #C9E2FF !important;
}

/* Toolbar Icons */
[data-testid="stToolbar"] svg {
    color: #570000 !important;
    fill: #570000 !important;
}

</style>

        """,
        unsafe_allow_html=True,
    )

apply_theme()

@st.cache_resource
def load_engine():
    return FaceEngine()

def image_to_bgr(uploaded):
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def save_face_sample(employee_id, face_rgb, embedding):
    employee_dir = FACES_DIR / employee_id
    employee_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(employee_dir.glob("*.jpg"))) + 1
    Image.fromarray(face_rgb).save(employee_dir / f"face_{count:03d}.jpg", quality=95)
    add_embedding(employee_id, embedding)

def page_dashboard():
    st.title("📊 Attendance Dashboard")
    summary = attendance_summary()
    today_rows = get_attendance(date.today(), date.today())
    late_count = sum(1 for row in today_rows if int(row.get("late_minutes") or 0) > 0)

    st.subheader("Summary Cards")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", summary["total"])
    c2.metric("Present", summary["present"])
    c3.metric("Absent", summary["absent"])
    c4.metric("Late", late_count)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Weekly Attendance Analytics")
        weekly_df = pd.DataFrame(weekly_attendance_counts(7))
        if not weekly_df.empty:
            st.bar_chart(weekly_df.set_index("Date"), y="Present", use_container_width=True)
        else:
            st.info("No weekly attendance data yet.")
    with right:
        st.subheader("Attendance Distribution")
        distribution_df = pd.DataFrame(attendance_distribution())
        if int(distribution_df["Employees"].sum()) > 0:
            st.vega_lite_chart(
                distribution_df,
                {
                    "mark": {"type": "arc", "innerRadius": 55},
                    "encoding": {
                        "theta": {"field": "Employees", "type": "quantitative"},
                        "color": {"field": "Status", "type": "nominal",
                                  "scale": {"domain": ["Present", "Absent"], "range": ["#D9A02E", "#8B2929"]}},
                        "tooltip": [{"field": "Status"}, {"field": "Employees"}]
                    }
                },
                use_container_width=True
            )
            st.dataframe(distribution_df, hide_index=True, use_container_width=True)
        else:
            st.info("No employee attendance distribution to display.")

    st.divider()
    st.subheader("Recent Attendance Activity")
    activity = recent_attendance_activity(10)
    if activity:
        activity_df = pd.DataFrame(activity).rename(columns={
            "event_time": "Time", "event_type": "Activity", "employee_id": "Employee ID",
            "full_name": "Employee Name", "department": "Department", "method": "Method", "note": "Details"
        })
        st.dataframe(activity_df[["Time", "Employee ID", "Employee Name", "Department", "Activity", "Method", "Details"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No attendance activity recorded yet.")

    st.divider()
    st.subheader("AI Recognition Statistics — Today")
    ai = recognition_statistics()
    a1, a2, a3 = st.columns(3)
    a1.metric("Recognition Success", ai["success"])
    a2.metric("Unknown Face", ai["unknown"])
    a3.metric("Failed Attempts", ai["failed"])
    st.caption("Recognition statistics count throttled camera recognition outcomes recorded by the Live Recognition page.")

    missing = get_missing_checkouts(date.today())
    if missing:
        st.error(f"Missing check-out alert: {len(missing)} employee(s) have not checked out.")
        st.dataframe(pd.DataFrame(missing), use_container_width=True, hide_index=True)
    st.caption(f"Shift: {WORK_START_TIME}–{WORK_END_TIME} | Late grace: {LATE_GRACE_MINUTES} minutes | Face auto check-out requires a second recognition at least {MIN_HOURS_BEFORE_AUTO_CHECKOUT} hours after check-in.")

def page_employees(engine):
    st.title("👥 Employee Registration")
    with st.form("employee_form", clear_on_submit=True):
        employee_id = st.text_input("Employee ID *")
        full_name = st.text_input("Full name *")
        department = st.text_input("Department")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Create Employee", type="primary")
    if submitted:
        if not employee_id.strip() or not full_name.strip():
            st.error("Employee ID and full name are required.")
        else:
            try:
                add_employee(employee_id, full_name, department, email)
                st.success("Employee created. Capture face samples below.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create employee: {exc}")

    employees = get_employees(active_only=False)
    if employees:
        st.subheader("Employee Directory")
        df = pd.DataFrame(employees)
        st.dataframe(df[["employee_id", "full_name", "department", "email", "active", "created_at"]],
                     use_container_width=True, hide_index=True)

        st.markdown("#### Update Employee")
        selected_edit = st.selectbox(
            "Choose employee to update", employees,
            format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}', key="edit_employee_select"
        )
        with st.form("update_employee_form"):
            edit_name = st.text_input("Full name", value=selected_edit["full_name"])
            edit_department = st.text_input("Department", value=selected_edit["department"] or "")
            edit_email = st.text_input("Email", value=selected_edit["email"] or "")
            update_submitted = st.form_submit_button("Update Employee", type="primary")
        if update_submitted:
            if not edit_name.strip():
                st.error("Full name is required.")
            elif update_employee(selected_edit["employee_id"], edit_name, edit_department, edit_email):
                st.success("Employee information updated.")
                st.rerun()

        st.markdown("#### Delete Employee")
        delete_target = st.selectbox(
            "Choose employee to delete", employees,
            format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}', key="delete_employee_select"
        )
        confirm_delete = st.checkbox(
            f'I understand that deleting {delete_target["full_name"]} permanently removes their attendance history and face samples.',
            key="confirm_delete_employee"
        )
        if st.button("Delete Employee", type="secondary", disabled=not confirm_delete):
            if delete_employee(delete_target["employee_id"]):
                import shutil
                shutil.rmtree(FACES_DIR / delete_target["employee_id"], ignore_errors=True)
                st.success("Employee and related records deleted.")
                st.rerun()

        st.divider()
        st.subheader("Capture Face Samples")
        active = [e for e in employees if e["active"]]
        if active:
            selected = st.selectbox("Select employee for face registration", active,
                                    format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}')
            capture = st.camera_input("Take a clear, front-facing face photo", key="registration_camera")
            if capture:
                bgr = image_to_bgr(capture)
                boxes = engine.detect(bgr)
                if len(boxes) != 1:
                    st.warning(f"Please capture exactly one face. Detected: {len(boxes)}")
                else:
                    face_rgb = engine.crop_face_rgb(bgr, boxes[0])
                    st.image(face_rgb, caption="Detected face crop", width=220)
                    if st.button("Save this face sample", type="primary"):
                        embedding = engine.embedding_from_rgb(face_rgb)
                        save_face_sample(selected["employee_id"], face_rgb, embedding)
                        st.success("Face sample and FaceNet embedding saved. Add 3–5 samples with different lighting/angles.")

class AttendanceVideoProcessor(VideoProcessorBase):
    def __init__(self, allowed_employee_id=None):
        self.allowed_employee_id = allowed_employee_id
        self.engine = load_engine()
        self.known = get_all_embeddings()
        self.liveness = BlinkLivenessDetector()
        self.last_seen = {}
        self.last_refresh = 0
        self.last_message = "Waiting for camera…"
        self.last_stats_log = {}

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        if now - self.last_refresh > 10:
            self.known = get_all_embeddings()
            self.last_refresh = now
        boxes = self.engine.detect(image)
        liveness_results = self.liveness.update(image, boxes, now)
        # Haar Cascade returns all detected face boxes; process each face independently.
        cv2.putText(image, f"Faces detected: {len(boxes)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for box, (is_live, liveness_message) in zip(boxes, liveness_results):
            x, y, w, h = box
            try:
                if not is_live:
                    label = f"Liveness: {liveness_message}"
                    color = (0, 165, 255)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(image, label[:55], (x, max(25, y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    continue

                face_rgb = self.engine.crop_face_rgb(image, box)
                match, distance = self.engine.recognize(face_rgb, self.known, MATCH_THRESHOLD)
                if match:
                    employee_id = match["employee_id"]
                    if self.allowed_employee_id and employee_id != self.allowed_employee_id:
                        label, color = "Different user", (0, 0, 220)
                        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(image, label, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        continue
                    stats_key = f"success:{employee_id}"
                    if now - self.last_stats_log.get(stats_key, 0) >= 5:
                        log_recognition_attempt("success", employee_id, float(distance))
                        self.last_stats_log[stats_key] = now
                    label = f'{match["full_name"]} ({distance:.2f})'
                    if now - self.last_seen.get(employee_id, 0) > ATTENDANCE_COOLDOWN_SECONDS:
                        state, msg = record_face_attendance(employee_id, method="FaceNet+BlinkLiveness")
                        self.last_seen[employee_id] = now
                        self.last_message = f'{employee_id}: {msg}'
                    color = (0, 180, 0)
                else:
                    stats_key = "unknown"
                    if now - self.last_stats_log.get(stats_key, 0) >= 5:
                        log_recognition_attempt("unknown", distance=float(distance))
                        self.last_stats_log[stats_key] = now
                    label = "Unknown"
                    color = (0, 0, 220)
            except Exception as exc:
                stats_key = "failed"
                if now - self.last_stats_log.get(stats_key, 0) >= 5:
                    log_recognition_attempt("failed", details=str(exc)[:200])
                    self.last_stats_log[stats_key] = now
                label, color = f"Error: {str(exc)[:30]}", (0, 165, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, self.last_message[:90], (10, image.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def page_live_recognition(allowed_employee_id=None):
    st.title("📷 Live Face Recognition & Check-in")
    st.warning("Blink challenge enabled: keep your eyes open, blink once, then open them again. This is a basic demo and does not reliably prevent photo/video replay attacks. Obtain informed employee consent.")
    known = get_all_embeddings()
    if not known:
        st.info("No face embeddings are registered yet. Register employees and capture face samples first.")
        return
    st.caption(f"Recognition threshold: Euclidean distance ≤ {MATCH_THRESHOLD:.2f}. Liveness requires open eyes → blink → open eyes. Glasses, poor lighting, or camera angle may affect detection.")
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="attendance-camera",
        video_processor_factory=(lambda: AttendanceVideoProcessor(allowed_employee_id)),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption(f"First recognized appearance records check-in. A later recognition records check-out only after {MIN_HOURS_BEFORE_AUTO_CHECKOUT} hours, preventing repeated camera frames from immediately closing attendance. Daily duplicate rows are blocked by the database unique constraint.")

def page_records():
    st.title("🗓️ Attendance Records")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date.today() - timedelta(days=30))
    end = c2.date_input("To", value=date.today())
    if start > end:
        st.error("From date must be before or equal to To date.")
        return
    rows = get_attendance(start, end)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"attendance_{start}_{end}.csv", mime="text/csv")
    st.subheader("Attendance Status & Alerts")
    missing = get_missing_checkouts(date.today())
    if missing:
        st.error(f"Missing check-out: {len(missing)} employee(s) have an open attendance record.")
        st.dataframe(pd.DataFrame(missing), use_container_width=True, hide_index=True)
    else:
        st.success("No missing check-out alert for today (or the scheduled shift has not ended yet).")

    st.subheader("Manual Check-out (Fallback)")
    employees = get_employees()
    if employees:
        selected = st.selectbox("Employee", employees,
                                format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}')
        if st.button("Record Check-out", type="primary"):
            if record_check_out(selected["employee_id"]):
                st.success("Check-out recorded and working hours calculated.")
            else:
                st.warning("No open check-in found for this employee today.")
    else:
        st.info("No active employees.")

def page_settings():
    st.title("⚙️ Employee Status & Settings")
    st.write(f"**Face match threshold:** {MATCH_THRESHOLD}")
    st.write("Lower threshold = stricter matching. Tune with genuine and impostor test samples.")
    st.subheader("Activate / Deactivate Employees")
    for emp in get_employees(active_only=False):
        c1, c2, c3 = st.columns([2, 4, 2])
        c1.write(emp["employee_id"])
        c2.write(emp["full_name"])
        new_state = c3.toggle("Active", value=bool(emp["active"]), key=f'active_{emp["employee_id"]}')
        if new_state != bool(emp["active"]):
            set_employee_active(emp["employee_id"], new_state)
            st.rerun()

def page_user_panel(employee_id):
    employees = {e["employee_id"]: e for e in get_employees(active_only=False)}
    emp = employees.get(employee_id)
    if not emp:
        st.error("Employee account is not linked to an employee record. Contact the administrator.")
        return
    st.title("👤 User Panel")
    st.subheader(f'{emp["full_name"]} ({employee_id})')
    today = get_employee_today_status(employee_id)
    c1, c2 = st.columns(2)
    c1.metric("Today's Attendance", today["status"])
    c2.metric("Check-in Time", today["check_in"].replace("T", " ") if today["check_in"] else "—")
    st.divider()
    st.subheader("📷 My Face Recognition Attendance")
    st.info(
        "Camera ကိုဖွင့်ပြီး ဝန်ထမ်းအများအပြားကို တစ်ပြိုင်နက် Scan လုပ်နိုင်ပါသည်။ "
        "Blink challenge အောင်မြင်ပြီး Registered Face ID ကိုက်ညီသူတစ်ဦးချင်းစီအတွက် Attendance မှတ်တမ်းတင်ပါမည်။"
    )
    # Multi-face mode: recognize every registered employee visible to the camera.
    page_live_recognition()
    st.caption(
        "Multi Detection ဖွင့်ထားသောကြောင့် Camera မြင်ကွင်းအတွင်းရှိ Registered ဝန်ထမ်းများကို တစ်ပြိုင်နက် အသိအမှတ်ပြုနိုင်ပါသည်။ "
        "အပေါ်တွင် ရွေးထားသော ဝန်ထမ်း၏ Present / Absent အခြေအနေကို ပြသထားပါသည်။"
    )
    if st.button("🔄 Refresh My Attendance Status", key="refresh_my_attendance"):
        st.rerun()

def login_page():
    st.title("🔐 Attendance System")
    st.write("Admin Panel သို့ဝင်ရန်သာ Admin User ID နှင့် Password လိုအပ်ပါသည်။")
    st.subheader("Admin Login")
    with st.form("admin_login_form"):
        user_id = st.text_input("Admin User ID")
        password = st.text_input("Admin Password", type="password")
        submitted = st.form_submit_button("Admin Login", type="primary")
    if submitted:
        user = authenticate_user(user_id, password)
        if user and user["role"] == "admin":
            st.session_state["auth_user"] = user
            st.rerun()
        else:
            st.error("Admin User ID သို့မဟုတ် Password မှားယွင်းနေပါသည်။")

    st.divider()
    st.subheader("User Panel")
    st.info("User Panel ကို Password မလိုဘဲ ဝင်ရောက်နိုင်ပါသည်။ မိမိ Employee ID ကို ရွေးချယ်ပါ။")
    active_employees = get_employees(active_only=True)
    if active_employees:
        selected = st.selectbox(
            "Select your Employee ID", active_employees,
            format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}',
            key="public_user_employee"
        )
        if st.button("Enter User Panel", type="primary"):
            st.session_state["public_employee_id"] = selected["employee_id"]
            st.rerun()
    else:
        st.warning("Active employee မရှိသေးပါ။ Admin မှ Employee Registration ပြုလုပ်ပါ။")

def main():
    user = st.session_state.get("auth_user")
    if user:
        st.sidebar.title("Face Recognition Attendance")
        st.sidebar.write(f'Logged in: **{user["user_id"]}**')
        if st.sidebar.button("Logout"):
            st.session_state.pop("auth_user", None)
            st.rerun()
        st.sidebar.subheader("Admin Panel")
        page = st.sidebar.radio("Navigation", ["Dashboard", "Employee Registration", "Live Recognition", "Attendance Records", "Settings"])
        try:
            engine = load_engine() if page in ("Employee Registration", "Live Recognition") else None
            if page == "Dashboard": page_dashboard()
            elif page == "Employee Registration": page_employees(engine)
            elif page == "Live Recognition": page_live_recognition()
            elif page == "Attendance Records": page_records()
            else: page_settings()
        except Exception as exc:
            st.error(f"Application error: {exc}")
            st.exception(exc)
            st.info("Check installation steps and terminal output. First FaceNet model initialization may take time.")
        return

    public_employee_id = st.session_state.get("public_employee_id")
    if public_employee_id:
        c1, c2 = st.columns([5, 1])
        with c2:
            if st.button("Change Employee"):
                st.session_state.pop("public_employee_id", None)
                st.rerun()
        page_user_panel(public_employee_id)
        return

    login_page()


if __name__ == "__main__":
    main()
