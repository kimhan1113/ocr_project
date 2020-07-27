import numpy as np
import os
import glob
from xml.etree import ElementTree
import cv2

from thirdparty.get_image_size import get_image_size

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for Ky_synth (International Conference on Document Analysis
    and Recognition) Focused Scene Text dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        test: Boolean for using training or test set.
        polygon: Return oriented boxes defined by their four corner points.
            Required by SegLink...
    """

    def __init__(self, data_path, polygon=False):
        self.data_path = data_path
        # self.gt_path = gt_path

        self.classes = ['Background', 'Text']

        image_path = os.path.join(data_path, 'img')

        # data_path = 'data/ky_synth'

        self.image_path = image_path
        self.image_names = []
        self.data = []
        self.text = []

        self.gt_path = gt_path = os.path.join(image_path, 'Annotations')

        self.classes = ['Background', 'Text']
        classes_lower = [s.lower() for s in self.classes]
        # print(len(glob.glob(os.path.join(gt_path, '*'))))
        if len(glob.glob(os.path.join(gt_path, '*'))) == 0:
            print('there is no annotations')

        # for filename in os.listdir(gt_path):


        for filename in (glob.glob(os.path.join(gt_path, '*'))):
            try:
                tree = ElementTree.parse(filename)
                root = tree.getroot()
            except:
                print(filename + " is corrupted" )
            boxes = []
            text = []

            size_tree = root.find('size')
            img_width = float(size_tree.find('width').text)
            img_height = float(size_tree.find('height').text)
            image_name = root.find('filename').text
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                # class_idx = classes_lower.index(class_name)
                for box in object_tree.iter('bndbox'):
                    xmin = float(box.find('xmin').text) / img_width
                    ymin = float(box.find('ymin').text) / img_height
                    xmax = float(box.find('xmax').text) / img_width
                    ymax = float(box.find('ymax').text) / img_height
                    box = [xmin, ymin, xmax, ymax, 1]
                    boxes.append(box)
                    text.append(class_name)
            boxes = np.asarray(boxes)

            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)


        self.init()

if __name__ == '__main__':
    gt_util = GTUtility('data/ky_synth/')
    print(gt_util)

    # idx = gt_util.image_names.index('00_4218.jpg')
    print(gt_util.image_names)
    # img_name = gt_util.image_names[idx]
    # img_path = os.path.join(gt_util.image_path, img_name)
    # img = cv2.imread(img_path)
    #
    # print('이미지의 shape', img.shape)
    #
    # boxes = np.copy(gt_util.data[idx][:, :-1])

    # print(gt_util.image_names.index('00_4218.jpg'))
