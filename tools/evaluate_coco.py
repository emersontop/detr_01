import os
import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor
import supervision as sv
import cv2

# Configurações
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'detr_model')
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
VAL_DIRECTORY = os.path.join(DATASET_DIR, 'valid')
ANNOTATION_FILE = os.path.join(VAL_DIRECTORY, '_annotations.coco.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega modelo e processador
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(DEVICE)
image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

# Carrega COCO
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()

results = []
print('Gerando predições...')
for image_id in tqdm(image_ids):
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(VAL_DIRECTORY, image_info['file_name'])
    image = cv2.imread(image_path)
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    pred = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        result = {
            'image_id': image_id,
            'category_id': int(label),
            'bbox': [float(x_min), float(y_min), float(width), float(height)],
            'score': float(score)
        }
        results.append(result)

# Salva predições em formato COCO
results_path = os.path.join(os.path.dirname(__file__), 'coco_predictions.json')
with open(results_path, 'w') as f:
    json.dump(results, f)
print(f'Predições salvas em {results_path}')

# Avaliação COCO
coco_dt = coco.loadRes(results_path)
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
