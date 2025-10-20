import os
import torchvision
# from transformers import DetrImageProcessor

# Initialize the image processor
# IMAGE_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Dataset directory path
# DATASET_DIR = '/home/live/Documents/pessoal/detr_01/data'

# Dataset directories and annotation file name
# ANNOTATION_FILE_NAME = "_annotations.coco.json"
# TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "train")
# VAL_DIRECTORY = os.path.join(DATASET_DIR, "valid")
# TEST_DIRECTORY = os.path.join(DATASET_DIR, "test")

# Custom CocoDetection dataset class that integrates the image processor from Hugging Face
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor,
        annotation_file_name: str = "_annotations.coco.json", 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, annotation_file_name)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target