import chainer.functions as F
import numpy as np


def focial_loss(pred, gt, alpha=2, beta=4, eps=2e-05):
    pos_indices = gt.data >= 1
    neg_indices = gt.data < 1

    neg_weights = pow(1 - gt, beta)

    loss = 0

    pos_loss = F.log(pred + eps) * pow(1 - pred, alpha) * pos_indices
    neg_loss = F.log(1 - pred + eps) * pow(pred, alpha) * neg_weights * neg_indices

    num_pos = F.sum(F.cast(pos_indices, np.float32))
    pos_loss = F.sum(pos_loss)
    neg_loss = F.sum(neg_loss)

    if num_pos.data == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


if __name__ == '__main__':
    import chainer
    import numpy as np
    pred = np.ones((1, 1, 3, 3)).astype(np.float32)
    gt = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).reshape((1, 1, 3, 3)).astype(np.float32)
    print(focial_loss(chainer.Variable(pred), chainer.Variable(gt)))
    print(focial_loss(chainer.Variable(gt), chainer.Variable(gt)))
