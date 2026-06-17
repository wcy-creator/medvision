"""
ByteTrack-style Multi-Object Tracker
Based on: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2022)
Simplified version for surgical instrument tracking.
"""
import time
import numpy as np
from collections import defaultdict


class Track:
    """Single tracked object."""
    _next_id = 0

    def __init__(self, bbox, score, class_id):
        Track._next_id += 1
        self.id = Track._next_id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.history = [bbox]
        self.velocity = [0, 0]

    def predict(self):
        """Predict next position based on velocity."""
        self.age += 1
        self.time_since_update += 1
        if len(self.history) >= 2:
            prev = np.array(self.history[-1])
            prev2 = np.array(self.history[-2])
            self.velocity = (prev - prev2).tolist()

    def update(self, bbox, score):
        """Update track with new detection."""
        if len(self.history) >= 2:
            prev = np.array(self.history[-1])
            curr = np.array(bbox)
            self.velocity = (curr - prev).tolist()
        self.history.append(bbox)
        if len(self.history) > 20:
            self.history.pop(0)
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)

    def get_area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2-x1) * (y2-y1)


class ByteTracker:
    """
    Multi-object tracker using IoU association.
    Based on ByteTrack algorithm (simplified).
    """

    def __init__(self, track_thresh=0.3, match_thresh=0.5, max_age=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks = []
        self.frame_id = 0

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of {'bbox': (x1,y1,x2,y2), 'score': float, 'class_id': int}
        Returns: list of active tracks
        """
        self.frame_id += 1

        # Predict all existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to existing tracks using IoU
        matched = set()
        matched_tracks = set()

        if detections:
            # Sort detections by score (high confidence first)
            det_sorted = sorted(enumerate(detections), key=lambda x: x[1]['score'], reverse=True)

            for det_idx, det in det_sorted:
                best_iou = 0
                best_track = None
                for track in self.tracks:
                    if track.id in matched_tracks:
                        continue
                    iou = self._iou(det['bbox'], track.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track

                if best_track and best_iou > self.match_thresh:
                    best_track.update(det['bbox'], det['score'])
                    best_track.class_id = det['class_id']
                    matched_tracks.add(best_track.id)
                    matched.add(det_idx)
                elif best_track and best_iou > 0.1:
                    # Low confidence match
                    best_track.update(det['bbox'], det['score'])
                    matched_tracks.add(best_track.id)
                    matched.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched and det['score'] > self.track_thresh:
                self.tracks.append(Track(det['bbox'], det['score'], det['class_id']))

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        return self.tracks

    def get_tracks(self, class_filter=None):
        """Get active tracks, optionally filtered by class."""
        active = [t for t in self.tracks if t.time_since_update < 3]
        if class_filter:
            active = [t for t in active if t.class_id in class_filter]
        return active

    def _iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        union = area1 + area2 - inter

        return inter / max(union, 1e-6)

    def reset(self):
        self.tracks = []
        self.frame_id = 0
