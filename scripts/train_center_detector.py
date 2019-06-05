import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam
from chainer.training import StandardUpdater, Trainer
from chainercv.datasets import COCOBboxDataset, VOCBboxDataset, voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator

from centernet.datasets.transforms import CenterDetectionTransform
from centernet.models.center_detector import CenterDetector, CenterDetectorTrain
from centernet.models.networks.hourglass import HourglassNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=2)
    args = parser.parse_args()

    num_class = len(voc_bbox_label_names)

    dataset = VOCBboxDataset()
    dataset = TransformDataset(dataset, CenterDetectionTransform(512, num_class, 4))

    detector = CenterDetector(HourglassNet, 512, num_class)
    train_chain = CenterDetectorTrain(detector, 1, 0.1, 1)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu(args.gpu)

    optimizer = Adam()
    optimizer.setup(train_chain)

    train_iter = MultiprocessIterator(dataset, args.batchsize)
    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = Trainer(updater, (3, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
