import os
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from pytorch_lightning import Trainer

from src.model import DetrModel
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

# Custom collate function for DataLoader
def collate_fn(batch, image_processor=IMAGE_PROCESSOR):
    """
    Custom collate function for object detection batches.

    Pads images in the batch to the same size using the image processor,
    and collects corresponding target labels.

    Args:
        batch (list): List of tuples (pixel_values, target) from the dataset.

    Returns:
        dict: Dictionary containing:
            - 'pixel_values': Tensor of padded images [batch_size, C, H, W]
            - 'pixel_mask': Tensor mask indicating valid pixels [batch_size, H, W]
            - 'labels': List of target dictionaries for each image
    """
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# Create DataLoaders for training, validation, and testing
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=1, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=1)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=1)

MAX_EPOCHS = 200
MODEL_PATH = '/home/live-segcom/Documents/detr_01/models/detr_model'

def main():
	print("Number of training examples:", len(TRAIN_DATASET))
	print("Number of validation examples:", len(VAL_DATASET))
	print("Number of test examples:", len(TEST_DATASET))
	
    # Model
	model = DetrModel(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
	batch = next(iter(TRAIN_DATALOADER))
	outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
	
    # Trainer
	trainer = Trainer(max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
	trainer.fit(model, TRAIN_DATALOADER, VAL_DATALOADER)
	
    # Save the trained model
	model.model.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
	main()
