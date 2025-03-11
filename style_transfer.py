import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from utils import normalization, gram_matrix
from settings import MODEL, STYLE_LAYERS, DEVICE, STEPS, CONTENT_WEIGHT, STYLE_WEIGHT


ContentLoss = nn.MSELoss()
StyleLoss = nn.MSELoss()

def get_features(image, model=MODEL):
    layers = {
        '0': 'conv_1',
        '5': 'conv_2',
        '10': 'conv_3',
        '19': 'conv_4',
        '28': 'conv_5'
    }
    features = {}
    x = image
    x = normalization(x) #Normalize image before feed to model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def get_dual_style(style_features1, style_features2, style_layers):
    final_style_features = {}
    for layer in style_layers:
        sf1 = style_features1[layer]
        sf2 = style_features2[layer]
        channel_end_index = int(sf1.size(1) / 4)
        sf1 = sf1[:, :channel_end_index, :, :]

        sf2 = sf2[:, channel_end_index:, :, :]

        final_style_features[layer] = torch.cat((sf1, sf2), dim=1)
    return final_style_features


def rot_style_features(style_features, style_layers):
    final_rot_style_features = {}
    for layer in style_layers:
        sf = style_features[layer].clone()
        # 2. Rotate 90 degrees.  torch.rot90 rotates *counter-clockwise*,
        #    and the dimensions are (batch, channel, height, width), so we rotate
        #    dimensions 2 and 3.  k=1 means rotate 90 degrees once.
        rot90 = torch.rot90(sf, k=1, dims=(2, 3))

        # 3. Rotate another 90 degrees (total 180).  We rotate rot90, NOT sf.
        rot180 = torch.rot90(rot90, k=1, dims=(2, 3))

        # 4. Differential evolution combination.
        final_rot = sf + (rot90 - rot180)
        final_rot_style_features[layer] = final_rot
    return final_rot_style_features


def style_tranfer_(optimizer, target_img,
                    content_features, style_features,
                    content_weight, style_weight=STYLE_WEIGHT, model=MODEL, style_layers=STYLE_LAYERS):

    optimizer.zero_grad()
    with torch.no_grad():
        target_img.clamp_(0, 1)
    target_features = get_features(target_img, model)
    content_loss = ContentLoss(content_features["conv_4"], target_features["conv_4"])


    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += StyleLoss(style_gram, target_gram)

    total_loss = content_loss*content_weight + style_loss*style_weight
    total_loss.backward(retain_graph=True)
    optimizer.step()
    return total_loss, content_loss, style_loss


def dual_style_transfer(content_image, style_image, content_weight=CONTENT_WEIGHT):
    content_features = get_features(content_image)
    style_features = get_features(style_image)
    final_style_features = get_dual_style(content_features, style_features, STYLE_LAYERS)
                                          
    # --- Initialize Target Images and Optimizers ---
    target_img = content_image.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.Adam([target_img], lr=0.02, weight_decay=0)

    for step in range(STEPS):
        total_loss, content_loss, style_loss = style_tranfer_(
            optimizer,
            target_img, content_features,
            final_style_features,
            content_weight)

        if step % 100 == 99:
            print(f"Epoch [{step+1}/{STEPS}] Total loss1: {total_loss.item():.6f} - \
                Content loss: {content_loss.item():.6f} - Style loss: {style_loss.item():.6f}")


        with torch.no_grad():
            target_img.clamp_(0, 1)
    
    return target_img


def rot_style_transfer(content_image, style_image, content_weight=CONTENT_WEIGHT):
    content_features = get_features(content_image)
    style_features = get_features(style_image)
    final_rot_style_features = rot_style_features(style_features, STYLE_LAYERS)
                                          
    # --- Initialize Target Images and Optimizers ---
    target_img1 = content_image.clone().requires_grad_(True).to(DEVICE)
    optimizer1 = optim.Adam([target_img1], lr=0.02, weight_decay=0)
    target_img2 = content_image.clone().requires_grad_(True).to(DEVICE)
    optimizer2 = optim.Adam([target_img2], lr=0.02, weight_decay=0)

    for step in range(STEPS):
        total_loss1, content_loss1, style_loss1 = style_tranfer_(
            optimizer1,
            target_img1, content_features,
            style_features,
            content_weight)
        
        total_loss2, content_loss2, style_loss2 = style_tranfer_(
            optimizer2,
            target_img1, content_features,
            final_rot_style_features,
            content_weight)

        if step % 100 == 99:
            print(f"Epoch [{step+1}/{STEPS}] Total loss1: {total_loss1.item():.6f} - \
                Content loss1: {content_loss1.item():.6f} - Style loss1: {style_loss1.item():.6f}")
            print(f"Epoch [{step+1}/{STEPS}] Total loss2: {total_loss2.item():.6f} - \
                Content loss2: {content_loss2.item():.6f} - Style loss2: {style_loss2.item():.6f}")


        with torch.no_grad():
            target_img1.clamp_(0, 1)
            target_img2.clamp_(0, 1)
    
    return target_img1, target_img2