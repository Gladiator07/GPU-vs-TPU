from numpy.lib.type_check import imag
import torch
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
import pickle
import config as cfg
import albumentations
import matplotlib.pyplot as plt


try:
    import torch_xla.core.xla_model as xm
    
    _xla_available = True
except:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:

        return pickle.load(f)
    

class ClassificationDataset:
    def __init__(self, id, classes, images, is_valid=False):
        self.id = id
        self.classes = classes
        self.images = images
        self.is_valid = is_valid

        if self.is_valid == True:
            self.aug = albumentations.Compose([
                albumentations.Normalize(cfg.mean, cfg.std, always_apply=True)
            ])
        
        else:
            # for training data (add more augmentations after setup)
            self.aug = albumentations.Compose([
                albumentations.Normalize(cfg.mean, cfg.std, always_apply=True)
            ])

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        id = self.id[idx]
        img = self.images[idx]
        img = np.array(Image.open(io.BytesIO(img)))
        img = self.aug(image = img)['image']
        classes = int(self.classes[idx])
        # id = int(self.id[idx])

        return {
                "targets": torch.tensor(classes),
                "image": torch.tensor(img, dtype=torch.float)

        }


class ClassificationDataLoader:
    def __init__(self, id, classes, images, is_valid=False):
        self.id = id
        self.classes = classes
        self.images = images
        self.is_valid = is_valid

        self.dataset = ClassificationDataset(
            id = self.id,
            classes = self.classes,
            images = self.images,
            is_valid = self.is_valid
        )
    
    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True, tpu=False):
        sampler = None
        if tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank = xm.get_ordinal(),
                shuffle=shuffle
            )
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = batch_size,
            sampler = sampler,
            drop_last=drop_last,
            num_workers = num_workers
        )

        return data_loader

def display_train_image(idx, train_dataset):
    """
    Displays/saves the image
    """
    print(train_dataset)
    target = train_dataset[idx]['targets']
    print(target)
    img = train_dataset[idx]['image']
    npimg = img.numpy()
    # image = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg)
    plt.savefig(f'../plots/train_image_{target}.png')

def display_valid_image(idx, valid_dataset):
    """
    Displays/saves the image
    """

    target = valid_dataset[idx]['targets']
    print(target)
    img = valid_dataset[idx]['image']
    npimg = img.numpy()
    # image = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg)
    plt.savefig(f'../plots/valid_image_{target}.png')


def check_dataset():
    train_ids = load_pickle_file(cfg.train_ids_224_pkl)
    train_class = load_pickle_file(cfg.train_class_224_pkl)
    train_images = load_pickle_file(cfg.train_image_224_pkl)

    val_ids = load_pickle_file(cfg.val_ids_224_pkl)
    val_class = load_pickle_file(cfg.val_class_224_pkl)
    val_images = load_pickle_file(cfg.val_image_224_pkl)

    train_dataset = ClassificationDataset(id=train_ids, classes = train_class, images = train_images)
    val_dataset = ClassificationDataset(id=val_ids, classes=val_class, images = val_images, is_valid=True)

    display_train_image(219, train_dataset)
    display_train_image(100, train_dataset)
    display_train_image(123, train_dataset)
    display_valid_image(25, val_dataset)
    display_valid_image(34, val_dataset)
    display_valid_image(2, val_dataset)


if __name__ == "__main__":
    check_dataset()
