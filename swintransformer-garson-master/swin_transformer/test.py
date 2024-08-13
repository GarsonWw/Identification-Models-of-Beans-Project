import os
import shutil
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# from model import swin_tiny_patch4_window7_224 as SwinTransformer
from model import swin_base_patch4_window7_224 as SwinTransformer

# 定义数据预处理和增强
data_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试集并进行预处理
test_dataset = datasets.ImageFolder(root=r'G:\PLN-DPM\dataset\predict', transform=data_transform)
# test_dataset = datasets.ImageFolder(root=r'G:\resnet-garson-master\data_set\beans_data\val', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载已经训练好的模型
model = SwinTransformer(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), channels=3, num_classes=2)
model.load_state_dict(torch.load(r'G:\swintransformer-garson-master\swin_transformer\weights_base_patch\model-59.pth'))

# 设置模型为评估模式
model.eval()

# 检查是否有可用的 GPU，如果有，使用GPU加速计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
model.to(device)

if not os.path.exists(r'G:\swintransformer-garson-master\data_set\wrong_classified_713'):
    os.makedirs(r'G:\swintransformer-garson-master\data_set\wrong_classified_713')
misclassified_dir=r'G:\swintransformer-garson-master\data_set\wrong_classified_713'

correct = 0
total = 0


with torch.no_grad():
    val_bar = tqdm(test_loader, file=sys.stdout)
    for batch_idx, (images, labels) in enumerate(val_bar):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(f'predict = {predicted}')
        # print(f'labels = {labels}')

        for i, (image, pred, label) in enumerate(zip(images, predicted, labels)):
            a=test_dataset.samples[batch_idx * test_loader.batch_size + i]
            if pred != label:
                image_path = test_dataset.samples[batch_idx * test_loader.batch_size + i][0]
                image_name = os.path.basename(image_path)
                new_path = os.path.join(misclassified_dir, image_name)
                shutil.copy(image_path, new_path)

print('Accuracy: {:.3f}'.format(correct / total))

# with torch.no_grad():
#     val_bar = tqdm(test_loader, file=sys.stdout)
#     for images, labels in val_bar:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#
#         for i, (pred, true) in enumerate(zip(predicted, labels)):
#             if pred != true:
#                 img_name = test_dataset.samples[i][0].split(os.sep)[-1]
#                 image_path= os.path.join(misclassified_dir, img_name)
#                 torchvision.utils.save_image(images[i],image_path)


# print('Accuracy: {:.3f}'.format(correct / total))

