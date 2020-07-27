import cv2
import numpy as np
import os
from pascal_voc_writer import Writer
import fastss
# tinyfastss install
from lib.sl_utils import rbox3_to_polygon, polygon_to_rbox, rbox_to_polygon
from lib.annotation import resize_and_pad, transform_box, to_array

from crnn_data import crop_words
from crnn_utils import alphabet87 as alphabet
from ssd_data import preprocess
from crnn_utils import decode


class Annotater:
    def __init__(self, tbpp, crnn, prior_util, image_names, annotation_path, input_size=(512, 512)):
        self.tbpp = tbpp
        self.crnn = crnn
        self.prior_util = prior_util
        self.image_names = image_names
        self.num_images = len(self.image_names)
        self.annotation_path = annotation_path
        self.input_size = input_size
        self.input_width = 256
        self.input_height = 32

        self.confidence_threshold = 0.35

    def transform_box(self, bbox):
        bbox = bbox[:, (0, 1, 4, 5)]
        bbox_arr = to_array(bbox, self.padding, self.im_shape)

        return bbox_arr

    def predict_tbpp(self):
        self.resized_image, self.padding, self.im_shape = resize_and_pad(self.image_name)
        x = np.array([preprocess(self.resized_image, (512, 512))])
        preds = self.tbpp.predict(x, batch_size=1, verbose=1)
        res = self.prior_util.decode(preds[0], self.confidence_threshold, fast_nms=False)

        return res

    def crop_boxes(self, result, grayscale=True):
        img = self.resized_image

        if len(result) > 0:

            rboxes = result[:, 12:17]

            boxes = np.asarray([rbox3_to_polygon(r) for r in rboxes])
            rboxes = np.array([polygon_to_rbox(b) for b in np.reshape(boxes, (-1, 4, 2))])

            bh = rboxes[:, 3]
            # NOTE : without margin
            # rboxes[:,2] += bh * 0.1
            # rboxes[:,3] += bh * 0.2

            boxes = np.array([rbox_to_polygon(f) for f in rboxes])

            boxes = np.flip(boxes, axis=1)
            boxes = np.reshape(boxes, (-1, 8))

            # boxes_mask_a = np.array([b[2] > b[3] for b in rboxes]) # width > height, in square world
            boxes_mask_b = 1 - np.array([(np.any(b < 0) or np.any(b > 1)) for b in boxes])  # box inside image

            # added filter to filter out detections near edges
            # NOTE : RoI is located on the centre of image
            boxes_mask_c = 1 - np.array([(np.any(b[:2] < 0.2) or np.any(b[:2] > 0.8)) for b in boxes])
            boxes_mask_c = np.array(boxes_mask_c)

            # NOTE : fancier methods using numpy slicing didnt work
            # had to do it old school way
            # boxes_mask = np.array(boxes_mask_a* boxes_mask_b*boxes_mask_c)
            boxes_mask = np.array(boxes_mask_b * boxes_mask_c)
            boxes_ = []

            for i, m in enumerate(boxes_mask):
                if m == 1:
                    boxes_.append(boxes[i])
            boxes = np.array(boxes_)

            if len(boxes) == 0:
                return -1, -1, -1

            words = crop_words(img, boxes, self.input_height, width=self.input_width, grayscale=True)
            words = np.asarray([w.transpose(1, 0, 2) for w in words])

            return words, boxes, 1
        else:
            return -1, -1, -1

    def to_string(self, res_crnn):
        strings = []
        for i in range(len(res_crnn)):
            chars = [alphabet[c] for c in np.argmax(res_crnn[i], axis=1)]
            res_str = decode(chars)
            strings.append(res_str)

        return strings

    def update_dictionary(self, filename):
        print(filename)
        f = open(filename, 'r')
        dict_name = os.path.join(os.path.dirname(filename), 'dict.dat')
        # with fastss.open(dict_name) as index:
        with fastss.open(dict_name) as index:
            #     line = f.readline()
            # with fastss.open(dict_name) as index:
            # with open(dict_name) as index:
            line = f.readline()
            while line:
                index.add(line.rstrip())
                line = f.readline()

    def annotate(self):
        # writer.addObject(query_dict[ed_ind][0],box[0],box[1],box[2],box[3])

        for i in range(self.num_images):
            # determine annotation paths
            self.image_name = self.image_names[i]
            annotation_name = os.path.splitext(os.path.basename(self.image_name))[0] + '.xml'
            annotation_name_full = os.path.join(self.annotation_path, annotation_name)
            dicttext_path = os.path.join(os.path.dirname(self.image_name), 'dict.txt')
            dict_path = os.path.join(os.path.dirname(self.image_name), 'dict.dat')

            for ext in ['.dat', '.dir', '.bak']:
                if os.path.exists(dict_path + ext):
                    os.remove(dict_path + ext)
            self.update_dictionary(dicttext_path)

            index = fastss.open(dict_path)
            print(index)

            # tbpp prediction and decoding ssd results
            res = self.predict_tbpp()

            # crop patches using results of tbpp
            words, boxes, flag = self.crop_boxes(res, grayscale=True)

            # NOTE : crnn model does not tolerate empty input
            if flag == -1:
                continue

            # run crnn with cropped patches (words)
            res_crnn = self.crnn.predict(words)

            # convert vectors into string 
            words_in_str = self.to_string(res_crnn)
            # convert boxes back to coordinates in image 
            box_arr = self.transform_box(boxes)

            # write xml file in voc format
            self.im_shape = cv2.imread(self.image_name).shape
            writer = Writer(self.image_name, self.im_shape[1], self.im_shape[0])

            # sort words such that each word appear at constant level
            # easier to spot anomaly
            args = np.argsort(words_in_str)
            # 수가 작은 값부터 그 수가 있는 인덱스를 반환 np.argsort

            # write words as bounding box format
            for j in args:
                box = box_arr[j]
                print(words_in_str[j])
                query_dict = index.query(words_in_str[j])

                ed_ind = 0
                while query_dict[ed_ind] == []:

                    ed_ind += 1
                    if ed_ind == 3:
                        break
                if ed_ind == 3:
                    continue

                print(query_dict[ed_ind][0])
                writer.addObject(query_dict[ed_ind][0], box[0], box[1], box[2], box[3])

            # close xml file
            writer.save(annotation_name_full)
