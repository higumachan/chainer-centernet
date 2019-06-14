import argparse
import multiprocessing

import chainer
import chainermn
import numpy
from chainer import datasets
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
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--mini', action="store_true")
    args = parser.parse_args()

    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator('pure_nccl')
    print(comm.size)

    device = comm.intra_rank

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

    if comm.rank == 0:
        train = TransformDataset(train, center_detection_transform)
        if args.mini:
            train = datasets.SubDataset(train, 0, 100)
    else:
        train = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize // comm.size, n_processes=2)

    if comm.rank == 0:
        test = VOCBboxDataset(
            year='2007', split='test',
            use_difficult=True, return_difficult=True)
        if args.mini:
            test = datasets.SubDataset(test, 0, 20)
        test_iter = chainer.iterators.SerialIterator(
            test, args.batchsize, repeat=False, shuffle=False)

    detector = CenterDetector(HourglassNet, 512, num_class)
    train_chain = CenterDetectorTrain(detector, 1, 0.1, 1, comm=comm)

    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    optimizer = chainermn.create_multi_node_optimizer(
        Adam(amsgrad=True),
        comm
    )
    optimizer.setup(train_chain)

    updater = StandardUpdater(train_iter, optimizer, device=device)

    trainer = Trainer(updater, (args.epoch, 'epoch'))

    if comm.rank == 0:
        log_interval = 1, 'epoch'
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
            extensions.snapshot_object(detector, 'detector{.updator.epoch:03}.npz'),
            trigger=(1, 'epoch')
        )

    trainer.run()


if __name__ == '__main__':
    main()
