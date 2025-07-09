import torch

if torch.cuda.is_available():
    torch.cuda.set_device(torch.device('cuda:0'))
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

num_devices = torch.cuda.device_count()