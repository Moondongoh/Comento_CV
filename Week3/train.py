# Week 3: YOLOv8 모델 학습
# 이 코드는 YOLOv8 모델을 학습시키는 코드입니다.

from ultralytics import YOLO
import torch


def main():
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolov8n.pt")

    # 학습 실행
    model.train(
        data="C:/Users/as/Desktop/dong/cat-dog-person.v4i.yolov8/data.yaml",  # roboflow 데이터셋 다운로드
        epochs=20,
        imgsz=640,
        batch=8,
        device="0",
        project="runs",
        name="cat-dog-person",
    )


# 이걸로 실행
if __name__ == "__main__":
    main()
