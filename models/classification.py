import torch
import torch.nn as nn

import utils

from . import SingleStageModel


class Classification(SingleStageModel):

    def __init__(self, params, dist_model=False):
        super(Classification, self).__init__(params, dist_model)
        self.params = params

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def inference(self): # for demo, in case inference procedure is different from eval
        raise NotImplemented

    def eval(self, ret_loss=True):
        with torch.no_grad():
            output = self.model(self.image)
        ret_tensors = {'common_tensors': []} # if you want to visualize some tensors in tensorboard
        if ret_loss:
            loss = self.criterion(output, self.target) / self.world_size
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):
        output = self.model(self.mask)
        loss = self.criterion(output, self.target) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
