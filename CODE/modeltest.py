import model.model
import torch

m = torch.load('save/mymodel_230120.pt')
m.test()
