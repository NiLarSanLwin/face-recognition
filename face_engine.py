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

class BlinkLivenessDetector:
    """
    Basic blink challenge for a classroom/demo prototype.
    Requires: eyes open -> eyes not detected (blink) -> eyes open.
    This is not a production anti-spoofing system.
    """
    def __init__(self, max_track_distance=100, track_timeout=3.0, challenge_timeout=10.0):
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_detector = cv2.CascadeClassifier(eye_path)
        if self.eye_detector.empty():
            raise RuntimeError(f"Could not load eye cascade: {eye_path}")
        self.max_track_distance = max_track_distance
        self.track_timeout = track_timeout
        self.challenge_timeout = challenge_timeout
        self.tracks = {}
        self.next_id = 0

    def _eye_state(self, frame_bgr, box):
        x, y, w, h = box
        roi = frame_bgr[y:y + int(h * 0.62), x:x + w]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(8, w // 12), max(8, h // 12))
        )
        return len(eyes) > 0

    def update(self, frame_bgr, boxes, now):
        """Return one (verified, message) tuple for each face box."""
        results = []
        used_tracks = set()

        for box in boxes:
            x, y, w, h = box
            center = (x + w / 2, y + h / 2)
            best_id, best_dist = None, float("inf")

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                tx, ty = track["center"]
                dist = ((center[0] - tx) ** 2 + (center[1] - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = track_id, dist

            if best_id is None or best_dist > self.max_track_distance:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "center": center, "state": "WAIT_OPEN",
                    "last_seen": now, "started": now
                }

            used_tracks.add(best_id)
            track = self.tracks[best_id]
            track["center"] = center
            track["last_seen"] = now

            if now - track["started"] > self.challenge_timeout and track["state"] != "LIVE":
                track["state"] = "WAIT_OPEN"
                track["started"] = now

            eyes_open = self._eye_state(frame_bgr, box)
            if track["state"] == "WAIT_OPEN":
                if eyes_open:
                    track["state"] = "WAIT_BLINK"
                    message = "Blink your eyes"
                else:
                    message = "Look at camera / keep eyes open"
            elif track["state"] == "WAIT_BLINK":
                if not eyes_open:
                    track["state"] = "WAIT_REOPEN"
                    message = "Open your eyes"
                else:
                    message = "Blink your eyes"
            elif track["state"] == "WAIT_REOPEN":
                if eyes_open:
                    track["state"] = "LIVE"
                    message = "Liveness verified"
                else:
                    message = "Open your eyes"
            else:
                message = "Liveness verified"

            results.append((track["state"] == "LIVE", message))

        stale = [
            track_id for track_id, track in self.tracks.items()
            if now - track["last_seen"] > self.track_timeout
        ]
        for track_id in stale:
            del self.tracks[track_id]

        return results

