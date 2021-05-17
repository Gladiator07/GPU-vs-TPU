import tensorflow as tf
import config as cfg
import pickle

# Create a dictionary describing the features.
train_feature_description = {
    'class': tf.io.FixedLenFeature([], tf.int64),
    'id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
 return tf.io.parse_single_example(example_proto, train_feature_description)


def tfrec_to_bytes():

    train_ids = []
    train_class = []
    train_images = []

    val_ids = []
    val_class = []
    val_images = []
    for i in cfg.train_files_224:
        train_image_dataset = tf.data.TFRecordDataset(i)
        train_image_dataset = train_image_dataset.map(_parse_image_function)
        ids = [str(id_features['id'].numpy())[2:-1] for id_features in train_image_dataset] # [2:-1] is done to remove b' from 1st and 'from last in train id names
        train_ids = train_ids + ids
        classes = [int(class_features['class'].numpy()) for class_features in train_image_dataset]
        train_class = train_class + classes
        images = [image_features['image'].numpy() for image_features in train_image_dataset]
        train_images = train_images + images
    
    print("Saving encoded train data")
    with open(cfg.train_ids_224_pkl, 'wb') as f:
        pickle.dump(train_ids, f)
    with open(cfg.train_class_224_pkl, 'wb') as f:
        pickle.dump(train_class, f)
    with open(cfg.train_image_224_pkl, 'wb') as f:
        pickle.dump(train_images, f)
    
    for i in cfg.val_files_224:
        val_image_dataset = tf.data.TFRecordDataset(i)
        val_image_dataset = val_image_dataset.map(_parse_image_function)
        ids = [str(id_features['id'].numpy())[2:-1] for id_features in val_image_dataset] # [2:-1] is done to remove b' from 1st and 'from last in train id names
        val_ids = val_ids + ids
        classes = [int(class_features['class'].numpy()) for class_features in val_image_dataset]
        val_class = val_class + classes
        images = [image_features['image'].numpy() for image_features in val_image_dataset]
        val_images = val_images + images

    print("Saving encoded validation data")
    with open(cfg.val_ids_224_pkl, 'wb') as f:
        pickle.dump(val_ids, f)
    with open(cfg.val_class_224_pkl, 'wb') as f:
        pickle.dump(val_class, f)
    with open(cfg.val_image_224_pkl, 'wb') as f:
        pickle.dump(val_images, f)
    
    print(len(val_ids))
    print(len(train_ids))

if __name__ == "__main__":
    tfrec_to_bytes()