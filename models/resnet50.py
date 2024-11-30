import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

from utils.train_test_metrics import DEVICE

def init_model_resnet50(learning_rate=0.001):
    print('Init model...')

    torch.cuda.empty_cache()

    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    model_name = 'ResNet50_CIFAR10'

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Done initializing model.')
    print(f"Model ID: {id(model)}, Optimizer ID: {id(optimizer)}, Criterion ID: {id(criterion)}")
    return model, model_name, criterion, optimizer, transform