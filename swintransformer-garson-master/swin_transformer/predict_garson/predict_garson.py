import os
import json
import shutil
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from double_swin.train.model import swin_base_patch4_window7_224 as create_model

# 全局变量，用于输出predict模型准确率
correct = 0
wrong = 0


def main(myfile):
    # def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 定义混淆矩阵标签
    actual_labels = []
    predicted_labels = []

    # load image
    root = "G:\\PLN-DPM\\dataset\\predict"
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    val_images_path = []  # 存储验证集的所有图片路径
    # read class_indict
    json_path = 'G:\swintransformer-garson-master\swin_transformer\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create net1
    net1 = create_model(num_classes=2).to(device)
    # load model weights
    model1_weight_path = "G:\swintransformer-garson-master\swin_transformer\weights_base_patch\model-59.pth"
    net1.load_state_dict(torch.load(model1_weight_path, map_location=device))

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        for img_path in images:
            val_images_path.append(img_path)
    for val_path in val_images_path:
        img_path1 = val_path
        assert os.path.exists(img_path1), "file: '{}' dose not exist.".format(img_path1)
        img = Image.open(img_path1).convert('RGB')
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        net1.eval()
        with torch.no_grad():
            # predict class
            output1 = torch.squeeze(net1(img.to(device))).cpu()
            predict = torch.softmax(output1, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print(val_path)
        print_res = "predict class: {}\ttrue class: {}\tprob: {:.3}".format(class_indict[str(predict_cla)],
                                                                            class_indict[val_path.split('\\')[4]],
                                                                            predict[predict_cla].numpy())
        print(print_res)
        # plt.title(print_res)

        p = int(str(predict_cla))
        # p = int(class_indict[str(predict_cla)])
        t = int(val_path.split('\\')[4])
        # 用类别编号作为混淆矩阵的元素
        actual_labels.append(p)
        predicted_labels.append(t)
        ju = 0
        if p == t:
            ju = 1
            global correct
            correct += 1
        else:
            ju = 0
            global wrong
            wrong += 1

        print(print_res + "\t" + str(ju))
        res = val_path + "\t" + print_res + "\t" + str(ju) + "\n"
        myfile.write(res)
        # plt.show()
    cm = confusion_matrix(actual_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    myfile = open(
        './swin_pred_result1.csv',
        'w')
    main(myfile)

    #     最后将结果输出
    print("\nThe accuracy of models is : " + str(correct / (wrong + correct)))
