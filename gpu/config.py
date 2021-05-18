import glob
import os
from torchvision import transforms

train_files_224 = glob.glob("../input/tfrecords-jpeg-224x224/train/*.tfrec")
val_files_224 = glob.glob("../input/tfrecords-jpeg-224x224/val/*.tfrec")

# tfrecords to pickle converted files
train_image_224_pkl = "../encoded_data/train_images_224.pkl"
train_ids_224_pkl = "../encoded_data/train_ids_224.pkl"
train_class_224_pkl = "../encoded_data/train_class_224.pkl"

val_image_224_pkl = "../encoded_data/val_images_224.pkl"
val_ids_224_pkl = "../encoded_data/val_ids_224.pkl"
val_class_224_pkl = "../encoded_data/val_class_224.pkl"


# jpeg converted files
DATASET_DIR = "../input/jpeg/jpeg-512x512"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

epochs = 25
train_bs = 512
valid_bs = 512
lr = 1e-4

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize,
    ]
)

valid_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
)
