import glob

train_files_224 = glob.glob('../input/tfrecords-jpeg-224x224/train/*.tfrec')
val_files_224 = glob.glob('../input/tfrecords-jpeg-224x224/val/*.tfrec')
# val_files_224 = glob.glob('../input/*val/*.tfrec')

# print(train_files_224[:5])

train_image_224_pkl = "../encoded_data/train_images_224.pkl"
train_ids_224_pkl = "../encoded_data/train_ids_224.pkl"
train_class_224_pkl = "../encoded_data/train_class_224.pkl"

val_image_224_pkl = "../encoded_data/val_images_224.pkl"
val_ids_224_pkl = "../encoded_data/val_ids_224.pkl"
val_class_224_pkl = "../encoded_data/val_class_224.pkl"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

batch_size = 16
epochs = 25
train_bs = 16 # per core
val_bs = 16
lr = 1e-4