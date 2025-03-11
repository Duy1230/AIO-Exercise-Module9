import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from settings import IMSIZE, DEVICE


image_transform = transforms.Compose([
    transforms.Resize((IMSIZE, IMSIZE)),
    transforms.ToTensor()
])


def load_image(image_name: str):
    image = Image.open(image_name)
    image = image_transform(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    # Clone the tensor to not do changes on it
    tensor = tensor.clone().detach()
    tensor = tensor.squeeze(0)  # remove batch dimension

    # Denormalize if needed (adjust based on your model's normalization)
    # tensor = tensor * 0.5 + 0.5

    # Convert to numpy and transpose
    tensor = tensor.cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    # Clip values to be between 0 and 1
    tensor = np.clip(tensor, 0, 1)

    # Convert to PIL Image
    return Image.fromarray((tensor * 255).astype(np.uint8))


def normalization(image):
    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width.
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
    mean = normalization_mean.view(-1, 1, 1)
    std = normalization_std.view(-1, 1, 1)
    return (image - mean) / std

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)