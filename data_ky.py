import numpy as np
import re
import os
from xml.etree import ElementTree
import cv2
import glob
import pickle

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for PASCAL VOC (Visual Object Classes) dataset.

    # Arguments
        data_path: Path to ground truth and image data.
    """

    def __init__(self, data_path='data/ky_dataset', polygon=True, phase='train'):
        self.data_path = data_path
        self.name = 'chip_dataset'
        self.image_path = data_path
        # TEST
        if phase == 'test':
            # self.db_names= ['samsung_mobile_vietnam_renamed']
            self.db_names = ['AGM_GOOD',
                            # 'Autolive_board',
                             # 'Flex_board',
                             # 'Mobis_board',
                             # 'samsung_mobile_vietnam_renamed',
                             # 'samsung_onyang_renamed'
                             ]
            # self.db_names= ['samsung_onyang_renamed']
        # TRAINING
        elif phase == 'train':
            self.db_names = ['QsA0_renamed',
                             'top_images_renamed',
                             '2017.03.02.Hitech_Wuxi_renamed',
                             'BigComp_renamed'
                             ]
        elif phase == 'train_synth':
            self.db_names = ['ky_synth']

        elif phase == 'train_extend':
            self.db_names = [
                '##49671',
                '##LongRunissue',
                '#54621',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6', '7', '8', '9', '14', '15', '16', '17', '18', '20',
                '23', '24', '26', '28', '29', '30', '31',  '32',
                '775-HAE7_ACU_MD_TOP', '1038A-0034REVGTOP_1',
                '2017.03.02.Hitech_Wuxi_renamed',
                '79541', '46518901REVETOP_1', 'AGM_KY', 'BigComp_renamed',
                'chardataset', 'DADO_2ARRAY',  'H3CPU_T', 'Jabil_F2A68-60004-T-NEW',
                'normal_machine_5335A',
                'QsA0_renamed', 'ReE0_renamed', 'S&T_Motive', 'samsung_mobile_vietnam_renamed',
                'samsung_onyang_renamed', 'ssdp_grab', 'top_images_renamed', 'Wistron_CASE3',
                'qboard_renamed'
            ]
        elif phase == 'qboard':
            self.db_names = ['qboard_renamed']

        self.classes = ['Background', 'Text']
        self.image_names = []
        self.data = []
        self.body = []
        self.text = []
        self.db_paths = []
        self.num_boxes = 0
        self.alpha_dict = {}
        self.anno_image = []
        self.img_name = []
        self.exist_anno = []


        for dir_ in self.db_names:
            self.db_paths.append(os.path.join(data_path, dir_))



        # img = [f for f in  if f.endswith('.bmp')]



        for i, db_path in enumerate(self.db_paths):
            img_ = os.listdir(self.db_paths[i])
            img_ = [os.path.splitext(f)[0] for f in img_ if f.endswith('.bmp')]

            self.gt_path = gt_path = os.path.join(db_path, 'Annotations')
            anno_file = glob.glob(os.path.join(gt_path, '*'))
            for anno in anno_file:
                self.anno_image.append(os.path.splitext(os.path.basename(anno))[0])
            # if len(glob.glob(os.path.join(gt_path, '*'))) != len(self.image_names):
            self.anno_image.sort()


            for f in self.anno_image:
                if f in img_:
                    self.exist_anno.append(f + str('.xml'))

            self.classes = ['Background', 'Text']
            classes_lower = [s.lower() for s in self.classes]


            # print(len(glob.glob(os.path.join(gt_path, '*'))))
            if len(glob.glob(os.path.join(gt_path, '*'))) == 0:
                print(db_path)
                assert 1 == 0



            # for name in self.anno_image:
            #     if db_path + '\'' + name + '.bmp' in self.image_names:
            #         self.exist_anno.append(name + str('.xml'))

            # annotations = [f for f in np.sort(glob.glob(os.path.join(gt_path, '*'))) if os.]

            for filename in np.sort(glob.glob(os.path.join(gt_path, '*'))):
                try:
                    tree = ElementTree.parse(filename)
                    root = tree.getroot()
                except:
                    continue
                boxes = []
                text = []
                body = []

                size_tree = root.find('size')
                img_width = float(size_tree.find('width').text)
                img_height = float(size_tree.find('height').text)

                image_name_ = os.path.basename(root.find('filename').text)

                image_name = os.path.join(self.db_names[i], image_name_)

                if os.path.exists(os.path.join(self.image_path, image_name)) == False:
                    continue
                try:
                    cv2.imread(os.path.join(self.image_path, image_name)).shape
                except:
                    continue

                for object_tree in root.findall('object'):
                    class_name = object_tree.find('name').text
                    for box in object_tree.iter('bndbox'):
                        xmin = float(box.find('xmin').text) / img_width
                        ymin = float(box.find('ymin').text) / img_height
                        xmax = float(box.find('xmax').text) / img_width
                        ymax = float(box.find('ymax').text) / img_height
                        box = [xmin, ymin, xmax, ymax, 1]
                        if polygon:
                            xmin = box[0]
                            ymin = box[1]
                            xmax = box[2]
                            ymax = box[3]
                            box = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                        box = box + [1]
                        if class_name != 'body':
                            boxes.append(box)
                            text.append(class_name)
                        else:
                            body.append(box)

                boxes = np.asarray(boxes)

                if len(boxes) == 0:
                    continue

                self.image_names.append(image_name)
                self.data.append(boxes)
                self.text.append(text)
                self.body.append(body)



                for string in text:
                    if string != None:
                        for char in string:
                            if char in list(self.alpha_dict.keys()):
                                self.alpha_dict[char] += 1
                            else:
                                self.alpha_dict[char] = 1

                self.num_boxes += boxes.shape[0]

        self.init()


if __name__ == '__main__':

    gt_util_ky = GTUtility('data/ky_dataset/', phase='train_extend')

    file_name_ky = 'gt_util_ky_dataset.pkl'

    pickle.dump(gt_util_ky, open(file_name_ky, 'wb'))

    # print(gt_util.image_names)
    # idx = gt_util.image_names.index('IC1_4_38_Top.bmp')
    # img_name = gt_util.image_names[idx]
    # print(gt_util.image_path)
    # img_path = os.path.join(gt_util.image_path, gt_util.image_names[0])

    # img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    #
    # print('이미지의 shape', img.shape[:2])
    # print(gt_util.image_names.index('IC1_4_38_Top.bmp'))
    # print(gt_util)



    # gt = gt_util.data
    # print(gt)

