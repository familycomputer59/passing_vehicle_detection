from collections import defaultdict
import cv2
import sys
import numpy as np
import pandas as pd

# Yolo v8　使用
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# 対象車両のリスト
vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}

# エリア選択
def select_roi(frame):
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return roi 

# 引数から動画ファイルのパスを取得
if len(sys.argv) < 2:
    print("解析する動画パスを入力してください。")
    sys.exit(1)
    
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# 初期化
track_history = defaultdict(lambda: [])
processed_track_ids = set()  

# エリア選択処理
success, first_frame = cap.read()
if not success:
    raise Exception("Failed to read video file")

roi_x, roi_y, roi_w, roi_h = select_roi(first_frame)

# 分析開始
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)
    
    # 検知した車両をプロット
    annotated_frame = results[0].plot()

    cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_ids = results[0].boxes.cls.int().cpu().tolist()

    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        x, y, w, h = box
        class_name = results[0].names[class_id]

        # 1回目にエリアに入った対象車両
        if (roi_x < float(x) < roi_x + roi_w and roi_y < float(y) < roi_y + roi_h and
                class_name in vehicle_counts and track_id not in processed_track_ids):
            vehicle_counts[class_name] += 1
            processed_track_ids.add(track_id)
            cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

    # 動画再生
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # q を押したら停止するように対応
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# results/vehicle_counts.csv　に保存
df = pd.DataFrame(list(vehicle_counts.items()), columns=['Vehicle Type', 'Count'])
df.to_csv('results/vehicle_count.csv', index=False)

cap.release()
cv2.destroyAllWindows()
