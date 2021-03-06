import argparse

import chainer
from chainer import datasets, training
from chainer.datasets import TransformDataset, ConcatenatedDataset
from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam
from chainer.training import StandardUpdater, Trainer, extensions, triggers
from chainercv.datasets import COCOBboxDataset, VOCBboxDataset, voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator

from centernet.datasets.transforms import CenterDetectionTransform, DataAugmentationTransform
from centernet.models.center_detector import CenterDetector, CenterDetectorTrain
from centernet.models.networks.hourglass import HourglassNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="-1")
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--mini', action="store_true")
    args = parser.parse_args()

    gpus = list(filter(lambda x: x >= 0, map(int, args.gpus.split(","))))

    num_class = len(voc_bbox_label_names)

    data_augmentation_transform = DataAugmentationTransform(512)
    center_detection_transform = CenterDetectionTransform(512, num_class, 4)

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
        test, args.batchsize // len(gpus), repeat=False, shuffle=False)

    detector = CenterDetector(HourglassNet, 512, num_class)
    train_chain = CenterDetectorTrain(detector, 1, 0.1, 1)

    gpus.sort()
    first_gpu = gpus[0]
    remain_gpu = gpus[1:]
    train_chain.to_gpu(first_gpu)

    optimizer = Adam(amsgrad=True)
    optimizer.setup(train_chain)

    devices = {
        "main": first_gpu
    }

    for i, gpu in enumerate(remain_gpu):
        devices[f"{i + 2}"] = gpu

    updater = training.updaters.ParallelUpdater(
        train_iter,
        optimizer,
        devices=devices,
    )

    log_interval = 1, 'epoch'
    trainer = Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        [
            'epoch', 'iteration', 'lr',
            'main/loss', 'main/hm_loss', 'main/wh_loss', 'main/offset_loss',
            'validation/main/map',
        ]),
        trigger=log_interval)
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
