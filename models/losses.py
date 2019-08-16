import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class L2LossWithIgnore(nn.Module):

    def __init__(self, ignore_value=None):
        super(L2LossWithIgnore, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target): # N1HW, N1HW
        if self.ignore_value is not None:
            target_area = target != self.ignore_value
            target = target.float()
            return (input[target_area] - target[target_area]).pow(2).mean()
        else:
            return (input - target.float()).pow(2).mean()


class MaskWeightedCrossEntropyLoss(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super(MaskWeightedCrossEntropyLoss, self).__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.byte()
        target_inmask = target[mask]
        target_outmask = target[~mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()

        predict_inmask = predict[mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        predict_outmask = predict[(~mask).view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss_inmask = nn.functional.cross_entropy(
            predict_inmask, target_inmask, size_average=False)
        loss_outmask = nn.functional.cross_entropy(
            predict_outmask, target_outmask, size_average=False)
        loss = (self.inmask_weight * loss_inmask + self.outmask_weight * loss_outmask) / (n * h * w)
        return loss
