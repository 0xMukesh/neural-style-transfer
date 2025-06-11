from torch import nn
import torch
from torchvision import models, transforms, utils
from PIL import Image

class NST(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.chosen_features = ["0", "5", "10", "19", "28"] # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 
        self.model = models.vgg19(weights="DEFAULT").features[:29] # type: ignore

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.model):
            x = layer(x)

            if (str(i) in self.chosen_features):
                features.append(x)
            
        return features

def load_image(path, device, transform):
    img = Image.open(path)
    img = transform(img).unsqueeze(0) # add batch dimension
    img = img.to(device)
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

content_img_path = "images/content.jpg"
style_img_path = "images/style.jpg"

content_img = load_image(content_img_path, device, transform)
style_img = load_image(style_img_path, device, transform)
generated_img = content_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01

model = NST().to(device).eval()
optimizer = torch.optim.Adam([generated_img], learning_rate)

for step in range(total_steps):
    content_features = model(content_img)
    style_features = model(style_img)
    generated_features = model(generated_img)

    content_loss = torch.Tensor(0)
    style_loss = torch.Tensor(0)

    for content_feature, style_feature, gen_feature in zip(content_features, style_features, generated_features):
        _, channel, height, width = gen_feature.shape

        content_loss += torch.mean((gen_feature - content_feature) ** 2)

        G = gen_feature.view(channel, height * width) @ gen_feature.view(channel, height * width).T
        A = style_feature.view(channel, height * width) @ style_feature.view(channel, height * width).T

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        utils.save_image(generated_img, f"output-{step % 200}.png")
