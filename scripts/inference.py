import random
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection

from src.dataset import CocoDetection

# variaveis globais de configuracao

# Initialize the image processor
IMAGE_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Dataset directory path
DATASET_DIR = '/home/live-segcom/Documents/detr_01/data'

# Dataset directories and annotation file name
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "train")
VAL_DIRECTORY = os.path.join(DATASET_DIR, "valid")
TEST_DIRECTORY = os.path.join(DATASET_DIR, "test")

# Create datasets using the custom CocoDetection class
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=False)

# Threshold for filtering weak detections
CONFIDENCE_TRESHOLD = 0.1

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_PATH = '/home/live-segcom/Documents/detr_01/models/detr_model'

# Load the pre-trained model
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

# select random image
image_ids = TEST_DATASET.coco.getImgIds()
idx = 0

while True:
    image_id = image_ids[idx]
    print('Image #{}'.format(image_id))

    # load image and annotations
    image_info = TEST_DATASET.coco.loadImgs(image_id)[0]
    annotations = TEST_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TEST_DATASET.root, image_info['file_name'])
    image = cv2.imread(image_path)

    # Ground truth
    detections_gt = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels_gt = [f"{id2label[class_id]}" for _, _, class_id, _ in detections_gt]
    frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections_gt, labels=labels_gt)

    # Inference
    with torch.no_grad():
        inputs = IMAGE_PROCESSOR(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = IMAGE_PROCESSOR.post_process_object_detection(
            outputs=outputs, 
            threshold=CONFIDENCE_TRESHOLD, 
            target_sizes=target_sizes
        )[0]
        detections_pred = sv.Detections.from_transformers(transformers_results=results)
        labels_pred = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_pred]
        frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections_pred, labels=labels_pred)

    # Combine images side by side
    combined = np.concatenate([frame_ground_truth, frame_detections], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title('Ground Truth (left) | Detections (right)')
    pressed_key = {'value': None}

    def on_key(event):
        if event.key == 'n':
            pressed_key['value'] = 'n'
            plt.close(fig)
        elif event.key == 'q':
            pressed_key['value'] = 'q'
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if pressed_key['value'] == 'n':
        idx = (idx + 1) % len(image_ids)
    elif pressed_key['value'] == 'q':
        break