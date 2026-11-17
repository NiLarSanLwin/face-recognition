import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, time, timedelta

from config import DB_PATH, DATA_DIR, WORK_START_TIME, WORK_END_TIME, LATE_GRACE_MINUTES, MIN_HOURS_BEFORE_AUTO_CHECKOUT


@contextmanager
def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(conn, table, column, definition):
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            department TEXT DEFAULT '',
            email TEXT DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_employee ON face_embeddings(employee_id);
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT,
            recognition_method TEXT NOT NULL DEFAULT 'FaceNet',
            late_minutes INTEGER NOT NULL DEFAULT 0,
            early_departure_minutes INTEGER NOT NULL DEFAULT 0,
            working_minutes INTEGER,
            attendance_status TEXT NOT NULL DEFAULT 'On Time',
            check_out_method TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id),
            UNIQUE(employee_id, attendance_date)
        );
        CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date);
        CREATE TABLE IF NOT EXISTS attendance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            method TEXT NOT NULL DEFAULT 'FaceNet',
            note TEXT DEFAULT '',
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
        );
        CREATE INDEX IF NOT EXISTS idx_events_employee_time ON attendance_events(employee_id, event_time);
        CREATE TABLE IF NOT EXISTS recognition_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_time TEXT NOT NULL,
            result TEXT NOT NULL CHECK(result IN ('success','unknown','failed')),
            employee_id TEXT,
            distance REAL,
            details TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_recognition_attempts_time ON recognition_attempts(attempt_time);
        """)
        # Safe in-place migration for databases created by earlier versions.
        _add_column_if_missing(conn, "attendance", "late_minutes", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "attendance", "early_departure_minutes", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "attendance", "working_minutes", "INTEGER")
        _add_column_if_missing(conn, "attendance", "attendance_status", "TEXT NOT NULL DEFAULT 'On Time'")
        _add_column_if_missing(conn, "attendance", "check_out_method", "TEXT")


def _password_hash(password, salt=None):
    import os, hashlib
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000)
    return salt.hex() + "$" + digest.hex()

def verify_password(password, stored):
    import hashlib
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200000)
        return digest.hex() == digest_hex
    except Exception:
        return False

def ensure_default_admin():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','user')),
            employee_id TEXT UNIQUE, active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id))""")
        if not conn.execute("SELECT 1 FROM users WHERE user_id='admin'").fetchone():
            conn.execute("INSERT INTO users(user_id,password_hash,role,created_at) VALUES(?,?,?,?)",
                ("admin", _password_hash("admin123"), "admin", datetime.now().isoformat(timespec='seconds')))

def create_employee_user(user_id, password, employee_id):
    with get_connection() as conn:
        conn.execute("INSERT INTO users(user_id,password_hash,role,employee_id,created_at) VALUES(?,?,?,?,?)",
            (user_id.strip(), _password_hash(password), "user", employee_id,
             datetime.now().isoformat(timespec='seconds')))

def authenticate_user(user_id, password):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=? AND active=1", (user_id.strip(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return {"user_id": row["user_id"], "role": row["role"], "employee_id": row["employee_id"]}
    return None

def get_employee_today_status(employee_id):
    with get_connection() as conn:
        row = conn.execute("SELECT check_in FROM attendance WHERE employee_id=? AND attendance_date=?",
                           (employee_id, date.today().isoformat())).fetchone()
    return {"status": "Present" if row else "Absent", "check_in": row["check_in"] if row else None}


def add_employee(employee_id, full_name, department='', email=''):
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO employees(employee_id, full_name, department, email, created_at) VALUES (?, ?, ?, ?, ?)",
            (employee_id.strip(), full_name.strip(), department.strip(), email.strip(), now)
        )


def get_employees(active_only=True):
    query = "SELECT * FROM employees"
    if active_only:
        query += " WHERE active=1"
    query += " ORDER BY full_name"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(query).fetchall()]


def set_employee_active(employee_id, active):
    with get_connection() as conn:
        conn.execute("UPDATE employees SET active=? WHERE employee_id=?", (int(active), employee_id))


def add_embedding(employee_id, embedding):
    import numpy as np
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO face_embeddings(employee_id, embedding, created_at) VALUES (?, ?, ?)",
            (employee_id, arr.tobytes(), datetime.now().isoformat(timespec='seconds'))
        )


def get_all_embeddings():
    import numpy as np
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT e.employee_id, e.full_name, f.embedding
            FROM face_embeddings f JOIN employees e ON e.employee_id=f.employee_id
            WHERE e.active=1 ORDER BY e.employee_id
        """).fetchall()
    return [{"employee_id": r["employee_id"], "full_name": r["full_name"],
             "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy()} for r in rows]


def _shift_datetime(day, clock_text):
    hh, mm = map(int, clock_text.split(":"))
    return datetime.combine(day, time(hh, mm))


def _late_minutes(check_in_dt):
    late_after = _shift_datetime(check_in_dt.date(), WORK_START_TIME) + timedelta(minutes=LATE_GRACE_MINUTES)
    return max(0, int((check_in_dt - late_after).total_seconds() // 60))


def _checkout_metrics(check_in_text, check_out_dt):
    check_in_dt = datetime.fromisoformat(check_in_text)
    elapsed = max(0, int((check_out_dt - check_in_dt).total_seconds() // 60))
    scheduled_end = _shift_datetime(check_out_dt.date(), WORK_END_TIME)
    early = max(0, int((scheduled_end - check_out_dt).total_seconds() // 60))
    return elapsed, early


def record_face_attendance(employee_id, method='FaceNet'):
    """First daily recognition checks in; a later recognition after the safety gap checks out."""
    now = datetime.now().replace(microsecond=0)
    today = now.date().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM attendance WHERE employee_id=? AND attendance_date=?",
            (employee_id, today)
        ).fetchone()
        if row is None:
            late = _late_minutes(now)
            status = "Late" if late > 0 else "On Time"
            conn.execute("""
                INSERT INTO attendance(employee_id, attendance_date, check_in, recognition_method, late_minutes, attendance_status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (employee_id, today, now.isoformat(timespec='seconds'), method, late, status))
            conn.execute("INSERT INTO attendance_events(employee_id,event_time,event_type,method,note) VALUES (?,?,?,?,?)",
                         (employee_id, now.isoformat(timespec='seconds'), "CHECK_IN", method, "Automatic face recognition"))
            return "check_in", f"Automatic check-in recorded ({status})"

        if row["check_out"]:
            return "complete", "Check-in and check-out already recorded today"

        check_in_dt = datetime.fromisoformat(row["check_in"])
        minimum_gap = timedelta(hours=MIN_HOURS_BEFORE_AUTO_CHECKOUT)
        if now - check_in_dt < minimum_gap:
            remaining = minimum_gap - (now - check_in_dt)
            return "waiting", f"Already checked in; auto check-out available after {int(remaining.total_seconds() // 60)} more minute(s)"

        working, early = _checkout_metrics(row["check_in"], now)
        conn.execute("""
            UPDATE attendance SET check_out=?, working_minutes=?, early_departure_minutes=?, check_out_method=?
            WHERE employee_id=? AND attendance_date=? AND check_out IS NULL
        """, (now.isoformat(timespec='seconds'), working, early, method, employee_id, today))
        conn.execute("INSERT INTO attendance_events(employee_id,event_time,event_type,method,note) VALUES (?,?,?,?,?)",
                     (employee_id, now.isoformat(timespec='seconds'), "CHECK_OUT", method, "Automatic face recognition"))
        return "check_out", "Automatic check-out recorded"


def record_check_in(employee_id, method='Manual'):
    state, message = record_face_attendance(employee_id, method)
    return state == "check_in", message


def record_check_out(employee_id, method='Manual'):
    now = datetime.now().replace(microsecond=0)
    today = now.date().isoformat()
    with get_connection() as conn:
        row = conn.execute("SELECT check_in, check_out FROM attendance WHERE employee_id=? AND attendance_date=?",
                           (employee_id, today)).fetchone()
        if not row or row["check_out"]:
            return False
        working, early = _checkout_metrics(row["check_in"], now)
        cur = conn.execute("""UPDATE attendance SET check_out=?, working_minutes=?, early_departure_minutes=?, check_out_method=?
                              WHERE employee_id=? AND attendance_date=? AND check_out IS NULL""",
                           (now.isoformat(timespec='seconds'), working, early, method, employee_id, today))
        if cur.rowcount:
            conn.execute("INSERT INTO attendance_events(employee_id,event_time,event_type,method,note) VALUES (?,?,?,?,?)",
                         (employee_id, now.isoformat(timespec='seconds'), "CHECK_OUT", method, "Manual check-out"))
        return cur.rowcount > 0


def get_attendance(start_date=None, end_date=None):
    query = """SELECT a.id, a.employee_id, e.full_name, e.department, a.attendance_date,
                      a.check_in, a.check_out, a.late_minutes, a.early_departure_minutes,
                      a.working_minutes, a.attendance_status, a.recognition_method, a.check_out_method
               FROM attendance a JOIN employees e ON e.employee_id=a.employee_id WHERE 1=1"""
    params = []
    if start_date:
        query += " AND a.attendance_date >= ?"; params.append(str(start_date))
    if end_date:
        query += " AND a.attendance_date <= ?"; params.append(str(end_date))
    query += " ORDER BY a.attendance_date DESC, a.check_in DESC"
    with get_connection() as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    for row in rows:
        mins = row.get("working_minutes")
        row["working_hours"] = f"{mins // 60}h {mins % 60}m" if mins is not None else "Pending check-out"
    return rows


def get_missing_checkouts(day=None):
    day = day or date.today()
    now = datetime.now()
    scheduled_end = _shift_datetime(day, WORK_END_TIME)
    if day == date.today() and now <= scheduled_end:
        return []
    with get_connection() as conn:
        rows = conn.execute("""SELECT a.employee_id, e.full_name, e.department, a.attendance_date, a.check_in
                              FROM attendance a JOIN employees e ON e.employee_id=a.employee_id
                              WHERE a.attendance_date=? AND a.check_out IS NULL ORDER BY e.full_name""",
                           (day.isoformat(),)).fetchall()
    return [dict(r) for r in rows]


def attendance_summary():
    today = date.today().isoformat()
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM employees WHERE active=1").fetchone()[0]
        present = conn.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date=?", (today,)).fetchone()[0]
    return {"total": total, "present": present, "absent": max(total-present, 0)}


def log_recognition_attempt(result, employee_id=None, distance=None, details=''):
    """Store a throttled camera recognition outcome for dashboard statistics."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO recognition_attempts(attempt_time,result,employee_id,distance,details) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec='seconds'), result, employee_id, distance, str(details)[:250])
        )

def recognition_statistics():
    today = date.today().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT result, COUNT(*) AS total FROM recognition_attempts WHERE date(attempt_time)=? GROUP BY result",
            (today,)
        ).fetchall()
    counts = {r['result']: r['total'] for r in rows}
    return {"success": counts.get('success', 0), "unknown": counts.get('unknown', 0), "failed": counts.get('failed', 0)}

def recent_attendance_activity(limit=10):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT ev.event_time, ev.event_type, ev.method, ev.note, ev.employee_id,
                   e.full_name, e.department
            FROM attendance_events ev JOIN employees e ON e.employee_id=ev.employee_id
            ORDER BY ev.event_time DESC, ev.id DESC LIMIT ?
        """, (int(limit),)).fetchall()
    return [dict(r) for r in rows]

def weekly_attendance_counts(days=7):
    start = date.today() - timedelta(days=days-1)
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT attendance_date, COUNT(*) AS present
            FROM attendance WHERE attendance_date >= ? AND attendance_date <= ?
            GROUP BY attendance_date ORDER BY attendance_date
        """, (start.isoformat(), date.today().isoformat())).fetchall()
    counts = {r['attendance_date']: r['present'] for r in rows}
    return [{"Date": (start + timedelta(days=i)).strftime('%a %d'), "Present": counts.get((start + timedelta(days=i)).isoformat(), 0)} for i in range(days)]

def attendance_distribution():
    summary = attendance_summary()
    return [{"Status": "Present", "Employees": summary['present']},
            {"Status": "Absent", "Employees": summary['absent']}]
