import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


class Params():
    def __init__(self):

        # Number of training epochs for sparse learning
        self.num_sparse_train_epochs = 10

        # Learning rate for optimizer
        self.learning_rate = 0.001

        # Hyperparameters for structured sparsity learning
        self.ssl_hyperparams = {
            "wgt_decay": 5e-4,
            "lambda_n": 5e-2,
            "lambda_c": 5e-2,
            "lambda_s": 5e-2,
        }

        # Threshold below which a weight value should be counted as too low
        self.threshold = 1e-5

def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Resize the image to the expected shape (32x32 pixels)
    image = image.resize((32, 32))

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # set model input
    data = np.expand_dims(normalized_image_array, axis=0)

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score



def count_sparse_wgt_by_layer(model, threshold):
    wgt_cnts = []
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            dims = 1
            for dim in list(param_tensor.size()):
                dims *= dim
            wgt_cnts.append((param_key, dims))
            sparse_wgt_cnt_layer = torch.sum(param_tensor < threshold).item()
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnt_layer))
    return wgt_cnts, sparse_wgt_cnts
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding,
                                           dilation, groups, bias)
        self.masked_channels = []
        self.mask_flag = False
        self.masks = None

    def forward(self, x):
        if self.mask_flag:
            self._expand_masks(x.size())
            weight = self.weight * self.masks
            return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def set_masked_channels(self, masked_channels):
        self.masked_channels = masked_channels
        self.mask_flag = len(masked_channels) > 0

    def get_masked_channels(self):
        return self.masked_channels

    def _expand_masks(self, input_size):
        if not self.masked_channels:
            self.masks = None
        else:
            batch_size, _, height, width = [int(input_size[i].item()) for i in range(4)]
            masks = torch.ones((len(self.masked_channels), batch_size, height, width), device=self.weight.device)
            self.masks = Variable(masks, requires_grad=False)

class CustomNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomNet, self).__init__()
        self.conv1_1 = MaskedConv2d(3, 64, 3, padding=1)
        self.conv2_1 = MaskedConv2d(64, 128, 3, padding=1)
        self.conv3_1 = MaskedConv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_1(out))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_1(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.softmax(out)

def count_sparse_wgt_by_filter(model, threshold):
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            if len(param_tensor.size()) != 4:
                sparse_wgt_cnts.append((param_key, None))
                continue
            num_filters = param_tensor.size()[0]
            sparse_wgt_cnts_by_filter = []
            for filter_idx in range(num_filters):
                cnt = torch.sum(param_tensor[filter_idx, :, :, :] < \
                                threshold).item()
                sparse_wgt_cnts_by_filter.append(cnt)
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnts_by_filter))
    return sparse_wgt_cnts

def count_sparse_wgt(model, threshold):
    weight_cnt = 0
    sparse_weight_cnt = 0
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            dims = 1
            for dim in list(param_tensor.size()):
                dims *= dim
            weight_cnt += dims
            sparse_weight_cnt += torch.sum(param_tensor < threshold).item()
    return weight_cnt, sparse_weight_cnt
def count_sparse_wgt_by_channel(model, threshold):
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            if len(param_tensor.size()) != 4:
                sparse_wgt_cnts.append((param_key, None))
                continue
            num_channels = param_tensor.size()[1]
            sparse_wgt_cnts_by_channel = []
            for channel_idx in range(num_channels):
                cnt = torch.sum(param_tensor[:, channel_idx, :, :] < \
                                threshold).item()
                sparse_wgt_cnts_by_channel.append(cnt)
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnts_by_channel))
    return sparse_wgt_cnts
def print_sparse_weights(model, threshold):
    wgt_cnt, sparse_wgt_cnt = count_sparse_wgt(model, threshold)
    print("\nTotal sparse weights: %.3f (%d/%d)" % (100. * sparse_wgt_cnt / \
          wgt_cnt, sparse_wgt_cnt, wgt_cnt))

    wgt_cnts, sparse_wgt_cnts = count_sparse_wgt_by_layer(model, threshold)
    print("\nSparse weight by layer")

    for idx in range(len(wgt_cnts)):
        layer_name = wgt_cnts[idx][0]
        wgt_cnt = wgt_cnts[idx][1]
        sparse_wgt_cnt = sparse_wgt_cnts[idx][1]
        print("Layer: {}, {} ({}/{})".format(layer_name, sparse_wgt_cnt / \
              wgt_cnt, sparse_wgt_cnt, wgt_cnt))

    sparse_wgt_cnts = count_sparse_wgt_by_filter(model, threshold)
    print("\nSparse weight by filter")
    for idx in range(len(sparse_wgt_cnts)):
        layer_name = sparse_wgt_cnts[idx][0]
        wgts_filters = sparse_wgt_cnts[idx][1]
        print("Layer: {}, {}".format(layer_name, wgts_filters))


def train(model, optimizer, criterion, trainloader):
    """A single training iteration"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch loop
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # A single optimization step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy info
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("Train: Loss: %.3f, Acc: %.3f (%d/%d)" % (train_loss / (batch_idx + 1), correct / total * 100., correct, total))

    return correct / total

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    print("Final Test Accuracy: {:.2%}".format(accuracy))  # Add this print statement for debugging
    return accuracy



def prep_dataloaders():

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    return trainloader, testloader, classes