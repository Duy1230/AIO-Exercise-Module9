import torch
import torchvision.models as models


IMSIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")


# Model settings
MODEL = models.vgg19(
    weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Training
STEPS = 500
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e5
