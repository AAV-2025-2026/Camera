from ultralytics import YOLO
model = YOLO('/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/traffic_lights.pt')
model.export(format='onnx', imgsz=640, simplify=True)