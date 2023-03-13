from torchvision.models import vgg16, VGG16_Weights

model = vgg16(VGG16_Weights.DEFAULT)

print(model)
