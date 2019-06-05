import argparse

import chainer
from chainer.datasets import TransformDataset, ConcatenatedDataset
from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam
from chainer.training import StandardUpdater, Trainer, extensions, triggers
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

    center_detection_transform = CenterDetectionTransform(512, num_class, 4)

    train = TransformDataset(
        ConcatenatedDataset(
            VOCBboxDataset(year='2007', split='trainval'),
            VOCBboxDataset(year='2012', split='trainval')
        ),
        center_detection_transform
    )
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    detector = CenterDetector(HourglassNet, 512, num_class)
    train_chain = CenterDetectorTrain(detector, 1, 0.1, 1)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu(args.gpu)

    optimizer = Adam()
    optimizer.setup(train_chain)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)

    log_interval = 100, 'iteration'
    trainer = Trainer(updater, (12000, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr', 'main/loss', 'main/hm_loss', 'main/wh_loss', 'main/offset_loss']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, detector, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=triggers.ManualScheduleTrigger(
            100, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
