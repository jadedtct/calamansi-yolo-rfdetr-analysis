from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("YOLO26 test completed.")