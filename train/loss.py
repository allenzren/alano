from train import *


def huber_loss(output, target, thres=1):
    """
	# Huber loss, 1D/2D
	Reference: wikipedia
	"""

    diff = torch.abs(output - target)
    b = (diff < thres).float()
    loss = b * 0.5 * (diff**2) + (1 - b) * (thres * diff - 0.5 * (thres**2))

    # Assign more weight to mininum weights
    # for i in [0,2,4,6]:
    #     loss[:,i] *= 10
    # for i in [1,3,5,7]:
    #     loss[:,i] *= 0

    # weight = ((target > 0.5) | (target < -1))*3
    # weight = ((target+0.2)**2)+1
    # loss *= weight

    return torch.mean(loss)


def focal_loss(output, target, alpha=1, gamma=1):
    """
	# Focal loss
	Reference: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
	"""

    BCE_loss = F.binary_cross_entropy_with_logits(output,
                                                  target,
                                                  reduction='none')
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    loss = alpha * (1 - pt)**gamma * BCE_loss
    return loss.mean()
