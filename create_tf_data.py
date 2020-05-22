import tensorflow as tf
import sys
sys.path.append("../ALET")

from object_detection.utils import dataset_util

import cv2
import base64
from alet import ALET



def create_tf_example(img, ann, path, categories):
    height, width = img.shape[:2]
    filename = ann['image_name']
    image_format = b'jpg'

    with open(path, "rb") as image:
        encoded_image_data = image.read()    #with tf.compat.v1.Session() as sess:
    #    img = img.eval()

    #_, buffer = cv2.imencode('.jpg', img)
    #encoded_image_data = base64.b64encode(buffer)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for bbox in ann['bbox']:
        xmin = bbox['left']/width
        xmax = (bbox['left']+bbox['width'])/width
        ymin = bbox['top']/height
        ymax = (bbox['top']+bbox['height'])/height
        if xmin>0 and ymin>0 and xmax<1 and ymax<1 and bbox['height']>0 and bbox['width']>0:
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

    for a in xmins+xmaxs+ymins+ymaxs:
        if a>=1 or a<=0:
            print(path, a, ann, len(ann['bbox']))

    classes_text = [categories[bbox['class']-1]['name'].encode() for bbox in ann['bbox']]
    classes = [bbox['class'] for bbox in ann['bbox']]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def create_tf_dataset(outname, dataset):
    writer = tf.io.TFRecordWriter(outname)

    cnt = 0
    for i in [ann['image_id'] for ann in dataset.annotations]:

        res = dataset.get_sample(i)
        if res==None:
            #print("Skip, {}".format(i))
            pass
        else:

            
            #print(res[0], res[1].shape)
            ann, img, path = res    
            tf_example = create_tf_example(img, ann, path, dataset.categories)
            writer.write(tf_example.SerializeToString())
            cnt += 1

    print(cnt)
    writer.close()



train_dataset = ALET("../ALET/new_train","../ALET/new_train.json")      
val_dataset = ALET("../ALET/new_val","../ALET/new_val.json")      
test_dataset = ALET("../ALET/new_test","../ALET/new_test.json")      



create_tf_dataset("alet_train.records", train_dataset)
create_tf_dataset("alet_val.records", val_dataset)
create_tf_dataset("alet_test.records", test_dataset)