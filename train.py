import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

if __name__ == '__main__':

    model = YOLOv10('SFFNet-S.yaml')  # 选择配置路径

    model.train(data='VisDrone.yaml',  # 选择数据集配置路径
                imgsz=640,
                epochs=1000,
                batch=8,
                device='0',
                optimizer='SGD',
                patience=50, 
                amp=False,
                project='runs/train',
                name='exp',
                )
