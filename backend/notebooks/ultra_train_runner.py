
import glob, os, sys
from ultralytics import YOLO

DATA_YAML = '/home/myid/bp67339/plant_disease/notebooks/Leaf-spot-disease-9/data.yaml'
MODEL_PREF = 'yolo11n-seg.pt'
FALLBACK_MODEL = 'yolov8n-seg.pt'
EPOCHS = 100
IMGSZ = 1024
BATCH = 8

def try_train(model_name):
    print(f"=== Training with {model_name} ===")
    m = YOLO(model_name)
    res = m.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=20,
        # disable runtime augs (already baked in Roboflow)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0,
        workers=8
    )
    runs = sorted(glob.glob("runs/segment/train*"), key=os.path.getmtime)
    latest = runs[-1]
    print("LATEST_RUN", latest)   # marker line for parent
    print("BEST_PT", f"{latest}/weights/best.pt")
    return latest

try:
    latest = try_train(MODEL_PREF)
    used = MODEL_PREF
except Exception as e:
    print("YOLOv11 failed, falling back to YOLOv8:", e)
    latest = try:
        try_train(FALLBACK_MODEL)
    except Exception as e2:
        print("Fallback also failed:", e2)
        raise

