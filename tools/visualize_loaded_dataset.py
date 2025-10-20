import random
import cv2
import numpy as np
import os
from transformers import DetrImageProcessor
import supervision as sv
import matplotlib.pyplot as plt

# Dataset class
from src.dataset import CocoDetection

# variaveis globais de configuracao

# Initialize the image processor
IMAGE_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Dataset directory path
DATASET_DIR = '/home/live/Documents/pessoal/detr_01/data'

# Dataset directories and annotation file name
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "train")
VAL_DIRECTORY = os.path.join(DATASET_DIR, "valid")
TEST_DIRECTORY = os.path.join(DATASET_DIR, "test")

# Create datasets using the custom CocoDetection class
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=IMAGE_PROCESSOR, train=False)

def visualize_random_sample(dataset):

	# select a random image id
	image_ids = dataset.coco.getImgIds()
	image_id = random.choice(image_ids)
	print('Image #{}'.format(image_id))

	# load image and annotations
	image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
	annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
	image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
	image = cv2.imread(image_path)

	# annotate
	detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

	# we will use id2label function for training
	categories = TRAIN_DATASET.coco.cats
	id2label = {k: v['name'] for k,v in categories.items()}

	labels = [
		f"{id2label[class_id]}" 
		for _, _, class_id, _ 
		in detections
	]

	box_annotator = sv.BoxAnnotator()
	frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

	# %matplotlib inline
	sv.show_frame_in_notebook(image, (8, 8))


def main():
	visualize_random_sample(TRAIN_DATASET)

if __name__ == "__main__":
	main()