import chainer.functions as F
import numpy as np


EPS = 2e-05

def focial_loss(pred, gt, alpha=2, beta=4, comm=None):
    pos_indices = gt >= 1
    neg_indices = gt < 1

    neg_weights = (1 - gt) ** beta

    pos_loss = F.log(pred + EPS) * (1 - pred) ** alpha * pos_indices
    neg_loss = F.log(1 - pred + EPS) * pred ** alpha * neg_weights * neg_indices

    num_pos = pos_indices.sum()
    pos_loss = F.sum(pos_loss)
    neg_loss = F.sum(neg_loss)

    loss = 0
    if comm is not None:
        num_pos = comm.allreduce_obj(num_pos)

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def reg_loss(output, mask, target, comm=None):
    """

    :param output: (N, dim, H, W)
    :param mask: (N, dim, H, W)
    :param target: (N, dim, H, W)
    :return:
    """



    ae = F.absolute_error(output, target)
    n_pos = mask.sum()

    if comm is not None:
        n_pos = comm.allreduce_obj(n_pos)

    return F.sum(ae * mask) / (n_pos + EPS)


def center_detection_loss(outputs, gts, hm_weight, wh_weight, offset_weight, comm=None):
    """

    :param outputs: list of dict of str, np.array(N, dim, H, W)
    :param gts: dict of str, (N, dim, H, W)
    :param hm_weight:
    :param wh_weight:
    :param offset_weight:
    :return:
    """

    hm_loss, wh_loss, offset_loss = 0, 0, 0
    for output in outputs:
        output['hm'] = F.sigmoid(output['hm'])

        hm_loss += focial_loss(output['hm'], gts['hm'], comm=comm) / len(outputs)

        if wh_weight > 0.0:
            wh_loss += reg_loss(output['wh'], gts['dense_mask'], gts['dense_wh'], comm=comm) / len(outputs)

        if offset_weight > 0.0:
            offset_loss += reg_loss(output['offset'], gts['dense_mask'], gts['dense_offset'], comm=comm) / len(outputs)

    loss = hm_weight * hm_loss + wh_weight * wh_loss + offset_weight * offset_loss

    return loss, hm_loss, wh_loss, offset_loss


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