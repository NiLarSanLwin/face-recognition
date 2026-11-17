import threading
import cv2
import numpy as np
from keras_facenet import FaceNet
from config import FACE_SIZE, DETECTION_SCALE, DETECTION_NEIGHBORS, MIN_FACE_SIZE

class FaceEngine:
    """OpenCV Haar face detector + pretrained TensorFlow/Keras FaceNet embeddings."""
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
        self.embedder = FaceNet()
        self._model_lock = threading.Lock()

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray, scaleFactor=DETECTION_SCALE,
            minNeighbors=DETECTION_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]

    def crop_face_rgb(self, frame_bgr, box, margin=15):
        x, y, w, h = box
        height, width = frame_bgr.shape[:2]
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(width, x+w+margin), min(height, y+h+margin)
        face = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        if face.size == 0:
            raise ValueError("Empty face crop")
        face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
        return face.astype(np.uint8)

    def embedding_from_rgb(self, face_rgb):
        batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
        with self._model_lock:
            embeddings = self.embedder.embeddings(batch)
        vector = np.asarray(embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Face embedding has zero norm")
        return vector / norm

    def recognize(self, face_rgb, known_embeddings, threshold=0.75):
        if not known_embeddings:
            return None, None
        query = self.embedding_from_rgb(face_rgb)
        best_id, best_name, best_distance = None, None, float('inf')
        for item in known_embeddings:
            ref = item['embedding']
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                continue
            distance = float(np.linalg.norm(query - (ref / ref_norm)))
            if distance < best_distance:
                best_id, best_name, best_distance = item['employee_id'], item['full_name'], distance
        if best_distance <= threshold:
            return {"employee_id": best_id, "full_name": best_name}, best_distance
        return None, best_distance


class FaceLivenessDetector:
    """
    Basic blink-challenge liveness check using OpenCV's Haar eye cascade.

    This is a lightweight educational/demo check, not production-grade
    anti-spoofing. Eye-detection failures can be caused by lighting, glasses,
    pose, or camera quality, and replay attacks may still bypass it.
    """
    def __init__(self, max_track_distance=100, track_timeout=3.0,
                 challenge_timeout=12.0, consecutive_frames=2):
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_detector = cv2.CascadeClassifier(eye_path)
        if self.eye_detector.empty():
            raise RuntimeError(f"Could not load eye cascade: {eye_path}")
        self.max_track_distance = max_track_distance
        self.track_timeout = track_timeout
        self.challenge_timeout = challenge_timeout
        self.consecutive_frames = consecutive_frames
        self.tracks = {}
        self.next_track_id = 1

    def _eyes_open(self, frame_bgr, box):
        x, y, w, h = box
        height, width = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + int(h * 0.62))
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        eyes = self.eye_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10)
        )
        # Two detected eyes are stronger evidence than one, but Haar may
        # detect only one eye when the face is slightly turned.
        return len(eyes) >= 1

    def update(self, frame_bgr, boxes):
        """Return {box_index: (is_live, instruction)} for the current frame."""
        now = time.time()
        for track_id in list(self.tracks):
            if now - self.tracks[track_id]["last_seen"] > self.track_timeout:
                del self.tracks[track_id]

        results = {}
        assigned = set()
        for index, box in enumerate(boxes):
            x, y, w, h = box
            center = (x + w / 2.0, y + h / 2.0)
            best_id, best_distance = None, float("inf")
            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue
                tx, ty = track["center"]
                distance = ((center[0] - tx) ** 2 + (center[1] - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None or best_distance > self.max_track_distance:
                best_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[best_id] = {
                    "center": center, "last_seen": now, "started": now,
                    "phase": "WAIT_OPEN", "open_count": 0, "closed_count": 0
                }

            assigned.add(best_id)
            track = self.tracks[best_id]
            track["center"] = center
            track["last_seen"] = now

            if track["phase"] != "LIVE" and now - track["started"] > self.challenge_timeout:
                track.update(phase="WAIT_OPEN", open_count=0, closed_count=0, started=now)

            eyes_open = self._eyes_open(frame_bgr, box)
            if eyes_open is True:
                track["open_count"] += 1
                track["closed_count"] = 0
                if track["phase"] == "WAIT_OPEN" and track["open_count"] >= self.consecutive_frames:
                    track["phase"] = "OPEN"
                elif track["phase"] == "CLOSED" and track["open_count"] >= self.consecutive_frames:
                    track["phase"] = "LIVE"
            elif eyes_open is False:
                track["closed_count"] += 1
                track["open_count"] = 0
                if track["phase"] == "OPEN" and track["closed_count"] >= self.consecutive_frames:
                    track["phase"] = "CLOSED"
            else:
                track["open_count"] = 0
                track["closed_count"] = 0

            if track["phase"] == "LIVE":
                results[index] = (True, "Liveness verified")
            elif track["phase"] == "WAIT_OPEN":
                results[index] = (False, "Look at camera / keep eyes open")
            elif track["phase"] == "OPEN":
                results[index] = (False, "Blink your eyes")
            else:
                results[index] = (False, "Open your eyes")
        return results
