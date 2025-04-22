import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('SFFNet-n.yaml')
    model.train(data='VisDrone.yaml',
                imgsz=640,
                epochs=1000,
                batch=1,
                workers=8,
                device='0',
                optimizer='SGD',
                patience=50,
                amp=False,
                project='/runs/train',
                name='exp',
                )