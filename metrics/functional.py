import torch


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr,dim=(2,3))
    union = torch.sum(gt + pr, dim=(2,3)) - intersection + eps
    #union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return torch.mean((intersection + eps) / union)


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(2,3))
    fp = torch.sum(pr,dim=(2,3)) - tp
    fn = torch.sum(gt,dim=(2,3)) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return torch.mean(score)


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype,dim=(2,3))
    score = tp / gt.view(-1).shape[0]
    return torch.mean(score)


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(2,3))
    fp = torch.sum(pr,dim=(2,3)) - tp

    score = (tp + eps) / (tp + fp + eps)

    return torch.mean(score)


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr,dim=(2,3))
    fn = torch.sum(gt,dim=(2,3)) - tp

    score = (tp + eps) / (tp + fn + eps)

    return torch.mean(score)

def categorical_crossentropy(pr,gt,eps=1e-7,class_weights=1.,ignore_channels=None):

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    #pr /=torch.sum(pr)
    pr_mean = pr /torch.sum(pr,dim=(2,3),keepdim=True)

    # clip to prevent NaN's and Inf's
    pr_new = torch.clamp(pr_mean,eps, 1 - eps)

    # calculate loss
    output = gt *torch.log(pr_new) * class_weights
    return - torch.mean(output)

def categorical_focal_loss(pr,gt,eps=1e-7,gamma=2.0, alpha=0.25,ignore_channels=None):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        gt: ground truth 4D tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        ignore_channels: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    """

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # clip to prevent NaN's and Inf's
    pr_new = torch.clamp(pr,eps, 1 - eps)

    # Calculate focal loss
    loss = - gt * (alpha *torch.pow((1 - pr_new), gamma) *torch.log(pr_new))

    return torch.mean(loss)
