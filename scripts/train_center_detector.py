import argparse

import chainer
from chainer import datasets
from chainer.datasets import TransformDataset, ConcatenatedDataset
from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam, SGD
from chainer.training import StandardUpdater, Trainer, extensions, triggers
from chainercv.datasets import COCOBboxDataset, VOCBboxDataset, voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator

from centernet.datasets.transforms import CenterDetectionTransform, DataAugmentationTransform
from centernet.models.center_detector import CenterDetector, CenterDetectorTrain
from centernet.models.networks.simple_cnn import SimpleCNN
from centernet.models.networks.hourglass import HourglassNet
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--mini', action="store_true")
    parser.add_argument('--input_size', type=int, default=512)
    args = parser.parse_args()

    dtype = np.float32

    num_class = len(voc_bbox_label_names)

    data_augmentation_transform = DataAugmentationTransform(args.input_size)
    center_detection_transform = CenterDetectionTransform(args.input_size, num_class, 4, dtype=dtype)

    train = TransformDataset(
        ConcatenatedDataset(
            VOCBboxDataset(year='2007', split='trainval'),
            VOCBboxDataset(year='2012', split='trainval')
        ),
        data_augmentation_transform
    )
    train = TransformDataset(train, center_detection_transform)
    if args.mini:
        train = datasets.SubDataset(train, 0, 100)
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    if args.mini:
        test = datasets.SubDataset(test, 0, 20)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    detector = CenterDetector(HourglassNet, args.input_size, num_class, dtype=dtype)
    #detector = CenterDetector(SimpleCNN, args.input_size, num_class)
    train_chain = CenterDetectorTrain(detector, 1, 0.1, 1)
    #train_chain = CenterDetectorTrain(detector, 1, 0, 0)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu(args.gpu)

    optimizer = Adam(alpha=1.25e-4)
    #optimizer = SGD()
    optimizer.setup(train_chain)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)

    log_interval = 1, 'epoch'
    log_interval_mini = 500, 'iteration'
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=f"result{args.gpu}")
    trainer.extend(extensions.LogReport(trigger=log_interval_mini))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        [
            'epoch', 'iteration', 'lr',
            'main/loss', 'main/hm_loss', 'main/wh_loss', 'main/offset_loss', 'main/hm_mae', 'main/hm_pos_loss', 'main/hm_neg_loss',
            'validation/main/map',
        ]),
        trigger=log_interval_mini)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, detector, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(1, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(detector, 'detector{.updater.epoch:03}.npz'),
        trigger=(1, 'epoch')
    )

    trainer.run()


if __name__ == '__main__':
    main()
