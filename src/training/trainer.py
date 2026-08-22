import torch
from torch.nn import CrossEntropyLoss
class Trainer:
    def __init__(self,model,optimizer):
        self.model=model
        self.opt=optimizer
        self.loss_fn=CrossEntropyLoss()
    def train_step(self,x,y):
        logits=self.model(x)
        loss=self.loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
