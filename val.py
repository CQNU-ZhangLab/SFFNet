import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('SFFNet-n.pt')

    metrics= model.val(
              data='VisDrone.yaml',
              split='val',
              imgsz=1024,
              batch=1,
              save_json=True,
              project='runs/val',
              name='exp',
              )




    
    
