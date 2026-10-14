import numpy as np
import math

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {} # ID: centroid
        self.disappeared = {} # ID: count of frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box, label):
        self.objects[self.next_object_id] = {'centroid': centroid, 'box': box, 'label': label, 'idle_frames': 0}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, labels):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i], labels[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[id]['centroid'] for id in object_ids]
            
            # Simple distance calculation
            distances = []
            for oc in object_centroids:
                row = []
                for ic in input_centroids:
                    dist = math.hypot(oc[0] - ic[0], oc[1] - ic[1])
                    row.append(dist)
                distances.append(row)
            D = np.array(distances)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                old_centroid = self.objects[object_id]['centroid']
                movement = math.hypot(old_centroid[0] - input_centroids[col][0], old_centroid[1] - input_centroids[col][1])
                if movement < 5:
                    self.objects[object_id]['idle_frames'] += 1
                else:
                    self.objects[object_id]['idle_frames'] = 0
                
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['box'] = rects[col]
                self.objects[object_id]['label'] = labels[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col], labels[col])

        return self.objects
