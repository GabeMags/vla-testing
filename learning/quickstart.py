# I created this file to just get some basic quick start code going because I was having trouble keeping this research going independently. Best to start small and take small steps than grand ones. -GM

# I originally started with imageclassifiertutorial.py but burnt out. Hoping to get some traction with this basic tutorial from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# 6/20/26

import torch
from torch import nn
from torch.utils.data import DataLoader # DataLoader wraps an iterable around the Dataset
from torchvision import datasets # Dataset stores the samples and their corresponding labels
from torchvision.transforms import v2

