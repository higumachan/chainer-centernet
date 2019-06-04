import chainer.functions as F
import numpy as np


EPS = 2e-05

def focial_loss(pred, gt, alpha=2, beta=4):
    pos_indices = gt.data >= 1
    neg_indices = gt.data < 1

    neg_weights = pow(1 - gt, beta)

    loss = 0

    pos_loss = F.log(pred + EPS) * pow(1 - pred, alpha) * pos_indices
    neg_loss = F.log(1 - pred + EPS) * pow(pred, alpha) * neg_weights * neg_indices

    num_pos = F.sum(F.cast(pos_indices, np.float32))
    pos_loss = F.sum(pos_loss)
    neg_loss = F.sum(neg_loss)

    if num_pos.data == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.shape[2]
    ind = F.repeat(F.expand_dims(ind, 2), dim, axis=2)
    feat = F.select_item(feat, ind)

def _transpose_and_gather_feat(feat, ind):
    feat = F.transpose(feat, (0, 2, 3, 1))
    shape = feat.shape
    feat = F.reshape(feat, (shape[0], -1, shape[3]))
    feat = _gather_feat(feat, ind)

    return feat


def _reg_loss(regr, gt_regr, mask):
    num = F.cast(mask.sum(), np.float32)
    mask = F.cast(F.repeat(F.expand_dims(mask, 2), gt_regr.shape[2], axis=2), np.float32)

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = F.huber_loss(regr, gt_regr, 1.0)
    regr_loss = regr_loss / (num + 1e-4)

    return regr_loss


def reg_loss(output, mask, target):
    """

    :param output: (N, dim, H, W)
    :param mask: (N, dim, H, W)
    :param target: (N, dim, H, W)
    :return:
    """

    ae = F.absolute_error(output, target)

    return F.sum(ae * mask) / (F.sum(F.cast(mask, np.float32)) + EPS)


def center_detection_loss(outputs, gts, hm_weight, wh_weight, offset_weight):
    """

    :param outputs: list of dict of str, np.array(N, dim, H, W)
    :param gts: dict of str, (N, dim, H, W)
    :param mask: (N, dim, H, W)
    :return:
    """

    hm_loss, wh_loss, offset_loss = 0, 0, 0
    for output in outputs:
        output['hm'] = F.sigmoid(output['hm'])

        hm_loss += focial_loss(output['hm'], gts['hm']) / len(outputs)

        if wh_weight > 0.0:
            wh_loss += reg_loss(output['wh'], gts['dense_mask'], gts['dense_wh']) / len(outputs)

        if offset_weight > 0.0:
            offset_loss += reg_loss(output['offset'], gts['dense_mask'], gts['dense_offset']) / len(outputs)

    loss = hm_weight * hm_loss + wh_weight * wh_loss + offset_weight * offset_loss

    return loss


if __name__ == '__main__':
    import chainer
    import numpy as np
    pred = np.ones((1, 1, 3, 3)).astype(np.float32)
    gt = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).reshape((1, 1, 3, 3)).astype(np.float32)
    print(focial_loss(chainer.Variable(pred), chainer.Variable(gt)))
    print(focial_loss(chainer.Variable(gt), chainer.Variable(gt)))

    output = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]]).astype(np.float32)
    mask = np.array([[[[False, False, False], [False, True, False], [False, False, False]], [[False, True, False], [False, False, False], [False, False, False]]]])
    target = np.array([[[[0, 0, 0], [0, 5, 0], [0, 0, 0]], [[0, 12, 0], [0, 0, 0], [0, 0, 0]]]]).astype(np.float32)

    print(reg_loss(
        output, mask, target
    ))