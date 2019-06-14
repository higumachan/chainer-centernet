import os
import pickle

import chainer
from chainer.dataset import concat_examples
from chainer.functions.activation.sigmoid import sigmoid
from chainercv.datasets import voc_bbox_label_names, VOCBboxDataset
from chainercv.visualizations import vis_bbox

from centernet.datasets.transforms import CenterDetectionTransform
from centernet.models.center_detector import CenterDetector
from centernet.models.networks.hourglass import HourglassNet

import matplotlib.pyplot as plt
import cv2
import numpy as np


if __name__ == '__main__':
    image = cv2.imread("./data/demo/dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    size = 256
    transform = CenterDetectionTransform(size, 100, 4)
    with chainer.using_config('train', False):
        num_class = len(voc_bbox_label_names)
        detector = CenterDetector(HourglassNet, size, num_class)
        chainer.serializers.load_npz('models/hg_256_pascalvoc.npz', detector)
        predicted = detector.predict([image], detail=True)
        thresh_idx = predicted[2][0] > 0.3
        ax = vis_bbox(
            image,
            predicted[0][0][thresh_idx],
            predicted[1][0][thresh_idx],
            predicted[2][0][thresh_idx],
            label_names=voc_bbox_label_names,
        )
        plt.show()

        output = predicted[3]
        resized_image = cv2.resize(image.transpose((1, 2, 0)), (size, size))
        for cls in range(num_class):
            print(cls)
            plt.imshow(resized_image)
            hm = output['hm'].data[0, cls]
            if hm.max() > 0.3:
                hm_img = cv2.resize(hm, (size, size))
                plt.title(voc_bbox_label_names[cls])
                plt.imshow(hm_img, alpha=0.8, cmap=plt.cm.jet)
                plt.colorbar()
                plt.show()
