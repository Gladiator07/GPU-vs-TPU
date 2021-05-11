import glob

train_files_224 = glob.glob('../input/tfrecords-jpeg-224x224/train/*.tfrec')
train_files_224 = glob.glob('../input/tfrecords-jpeg-224x224/val/*.tfrec')
val_files_224 = glob.glob('../input/*val/*.tfrec')

# print(train_files_224[:5])

train_image_224_pkl = "../encoded_data/train_images_224.pkl"
train_ids_224_pkl = "../encoded_data/train_ids_224.pkl"
train_class_224_pkl = "../encoded_data/train_class_224.pkl"

val_image_224_pkl = "../encoded_data/val_images_224.pkl"
val_ids_224_pkl = "../encoded_data/val_ids_224.pkl"
val_class_224_pkl = "../encoded_data/val_class_224.pkl"