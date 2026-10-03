from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

results = model.val(
    data="dataset/data.yaml",
    split="val",
    imgsz=640,
    device="mps"
)

print(results)