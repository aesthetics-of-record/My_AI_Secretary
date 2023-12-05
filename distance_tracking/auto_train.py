import cv2
from ultralytics import YOLO
import math
import os

model = YOLO('yolov8n.pt')  # 사전학습된 모델 불러오기

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 프레임 개수
count = 0


# 출력 폴더가 존재하지 않으면 생성
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


make_dir("./data/train/images")
make_dir("./data/train/labels")


while True:
    if count == 100:
        print("프레임 100개 라벨링 완료.")
        break

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

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class Name
            cls = int(box.cls[0])

            if cls == 0 and conf >= 0.75:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.putText(frame, f"cls: {cls} / conf: {conf} / Distance: {distance:.2f}",
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1
    print(count)

cap.release()
cv2.destroyAllWindows()
