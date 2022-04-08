import torchvision.transforms as transfroms
import torch
import torchvision


def create_dataloaders(train_folder,batch_size):
    dataset = torchvision.datasets.ImageFolder(train_folder, transform=transfroms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True)
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squares_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    #VAR[X] = E[X**2] - E[X]**2
    mean = channels_sum / num_batches
    std = (channels_squares_sum / num_batches - mean**2)**0.5

    return mean, std

if __name__=='__main__':
    mean, std = create_dataloaders('/home/alex/space_pictures/images/train', 16)
    print(str(mean), str(std))