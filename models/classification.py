import torch
import torch.nn as nn

import utils

from . import SingleStageModel


class Classification(SingleStageModel):

    def __init__(self, params, load_path=None, dist_model=False):
        super(Classification, self).__init__(params, load_path, dist_model)
        self.params = params

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, image, target=None):
        self.image = image.cuda()
        if target is not None:
            self.target = target.cuda()

    def inference(self): # for demo, in case inference procedure is different from eval
        raise NotImplemented

    def evaluate(self):
        with torch.no_grad():
            output = self.model(self.image)
            top1, top5 = utils.accuracy(output, self.target, topk=(1, 5))
        return {'top1': top1, 'top5': top5} # consistency with "eval_record" in config

    def extract(self):
        with torch.no_grad():
            output, feat = self.model(self.image, ret_feat=True)
        return feat.detach()

    def forward(self, ret_loss=True):
        with torch.no_grad():
            output = self.model(self.image)
        ret_tensors = {'common_tensors': [self.image]} # if you want to visualize some tensors in tensorboard
        if ret_loss:
            loss = self.criterion(output, self.target)
            return ret_tensors, {'loss': loss} # consistency with "loss_record" in config
        else:
            return ret_tensors

    def step(self):
        output = self.model(self.image)
        loss = self.criterion(output, self.target)
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
