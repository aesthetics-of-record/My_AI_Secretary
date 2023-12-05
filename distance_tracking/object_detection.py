import cv2
from ultralytics import YOLO
import math


model = YOLO('yolov8n.pt')  # 사전학습된 모델 불러오기

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 사람 탐지
    detections = model.predict(source=frame, save=False)

    for i in detections:
        boxes = i.boxes

        for box in boxes:
            print(box)

            # 감지된 사람에 대한 박스 그리기
            x1, y1, x2, y2 = box.xyxy[0]

            # 거리 추정 (단순화된 예시)
            distance = 1000 / (y2 - y1)  # 예시적인 공식, 실제 사용시 보정 필요

            """
            문제점은.. 객체 인식 모델이 크기가 계속 달라지기 때문에 정확한 거리추정이 불가능하다.
            KITTI / 깊이추정 ?
            아니면, 각도만 가운데로 맞추고 나머지는 하드웨어(라이더 센서 등)로 판단.*
            """

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 각도 조정을 위한 x 좌표의 차이
            height, width, channels = frame.shape
            center_distance = width / 2 - (x2 - (x2 - x1) / 2)

            """
            이게 0이 될 때 까지 (- 면 우로 회전 + 면 좌로 회전)
            그리고 -10 과 +10 사이일 때는 전진.
            """

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class Name
            cls = int(box.cls[0])

            if cls == 0 and conf >= 0.75:
                print(center_distance)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.putText(frame, f"cls: {cls} / conf: {conf} / Distance: {distance:.2f}",
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
