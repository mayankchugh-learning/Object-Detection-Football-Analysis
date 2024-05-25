from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20
        dectections = []
        for i in range(0, len(frames), batch_size):
            # batch = frames[i:i+batch_size]
            detection_batch = self.model.predict(frames[i:i+batch_size], config=0.1) 
            # dectections.extend(self.model.predict(batch))
            dectections += detection_batch
            # dectections = self.model.predict(frames)

        return dectections

    def get_obects_tracker(self, frames):
        
        detections = self.detect_frames(frames)