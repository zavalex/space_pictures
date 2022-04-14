import torch
import numpy as np
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import sys
import random
from custom_models import CustomEffnet
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()  # Evaluation mode
    accuracy = 0
    correct_samples = 0
    total_samples = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            indices = torch.argmax(pred, 1)
            correct_samples += torch.sum(indices == y)
            total_samples += y.shape[0]
            accuracy = float(correct_samples) / total_samples
    return accuracy


def train_model(model, train_loader, val_loader, loss,
                optimizer, num_epochs, sheduler):
    """
    Train model with sheduler
    Returns: lists of loss, val and train history
    """
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()  # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):

            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        sheduler.step()

        print("Epoch %i. Average loss: %f, Train accuracy: %f, Val accuracy: %f"
              % (epoch+1, ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history


def create_dataloaders(train_folder, batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(
            [0.5362, 0.5362, 0.5362],
            [0.3288, 0.3288, 0.3288])
    ])
    dataset = torchvision.datasets.ImageFolder(train_folder, train_transforms)

    data_size = len(dataset)
    validation_fraction = .2
    val_split = int(np.floor((validation_fraction) * data_size))
    indices = list(range(data_size))
    np.random.seed(127)
    np.random.shuffle(indices)

    val_indices, train_indices = indices[:val_split], indices[val_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=val_sampler)
    return train_loader, val_loader


def test_model(model, batch_size, test_path) -> float:
    test_transforms = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(
             [0.5362, 0.5362, 0.5362],
             [0.3288, 0.3288, 0.3288])])
    test_dataset = torchvision.datasets.ImageFolder(test_path, test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)
    accuracy = compute_accuracy(model, test_loader)
    return accuracy


def main():
    utils.torch_fix_seed(127)
    args = sys.argv[1:]
    TRAIN_FOLDER = args[0]
    MODEL_PATH = args[1]
    MODEL_NAME = args[2]
    train_loader, val_loader = create_dataloaders(TRAIN_FOLDER, 32)
    model = CustomEffnet(pretrained=True, num_classes=4)
    model.cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4)
    sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5,
                                         verbose=True)
    loss_history, train_history, val_history = train_model(
        model, train_loader, val_loader, loss, optimizer, 6,
        sheduler)
    test_result = test_model(model, 32, '/home/alex/space_pictures/images/test')
    print(test_result)
    torch.save(model.state_dict(), MODEL_PATH+'/'+str(test_result)+MODEL_NAME)
    


if __name__ == "__main__":
    main()
