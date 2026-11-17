from pathlib import Path
from datetime import time

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "employee_faces"
DB_PATH = DATA_DIR / "attendance.db"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODEL_CACHE_DIR = BASE_DIR / "models"

FACE_SIZE = (160, 160)
DETECTION_SCALE = 1.1
DETECTION_NEIGHBORS = 5
MATCH_THRESHOLD = 0.75
MIN_FACE_SIZE = 60
ATTENDANCE_COOLDOWN_SECONDS = 60

# Default shift policy. Change these values to match the organization's work schedule.
WORK_START_TIME = time(9, 0)
WORK_END_TIME = "17:00"
LATE_GRACE_MINUTES = 15
# A second face recognition can become check-out only after this minimum elapsed time.
MIN_HOURS_BEFORE_AUTO_CHECKOUT = 4

# Liveness settings: blink-based demonstration using OpenCV Haar eye detection.
# These values may need tuning for the camera, lighting, and users.
LIVENESS_ENABLED = True
LIVENESS_TRACK_MAX_DISTANCE = 100
LIVENESS_TRACK_TIMEOUT_SECONDS = 3.0
LIVENESS_CHALLENGE_TIMEOUT_SECONDS = 12.0
LIVENESS_CONSECUTIVE_FRAMES = 2
