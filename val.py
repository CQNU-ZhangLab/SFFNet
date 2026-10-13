import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10


if __name__ == '__main__':
    model = YOLOv10('best.pt') # 选择配置路径

    model.val(data='VisDrone.yaml', # 选择数据集配置路径
              split='val',
              imgsz=1024,
              batch=8,
              save_json=True,
              project='runs/val',
              name='exp',
              )



