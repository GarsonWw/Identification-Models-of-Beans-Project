import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from model import swin_base_patch4_window7_224 as create_model
from utilss.utils import *
from my_dataset import MyDataSet
from utils import read_split_data, swin_train_one_epoch, devaluate
from utilss.lr_scheduler import *


save_path1='./weights/best_epoch1.pth'
save_path2='./weights/best_epoch2.pth'
def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model1 = create_model(num_classes=args.num_classes).to(device)
    model2 = create_model(num_classes=args.num_classes).to(device)
    net1 = model1
    net2 = model2

    #------------------------------------------------------------

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(net1.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in net1.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in net1.parameters() if p.requires_grad]
    optimizer1 = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(net2.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in net2.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in net2.parameters() if p.requires_grad]
    optimizer2 = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    epochs = 60
    bestacc = 0.0
    bestacc_com = 0.0
    for epoch in range(epochs):
        # train
        train_loss1, train_acc1 = swin_train_one_epoch(model=net1,
                                                optimizer=optimizer1,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, num_model="net1")
        train_loss2, train_acc2 = swin_train_one_epoch(model=net2,
                                                optimizer=optimizer2,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, num_model="net2")

        # validate
        acc1, acc2, acc_com  = devaluate(model1=net1, model2=net2,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        print("epoch "+str(epoch)+"\t"+"acc1:\t"+str(acc1)+"\t"+"acc2:\t"+str(acc2)+"\tacc_com:\t"+str(acc_com)+"\n")
        if acc1<acc2:
            temp = acc2
        else:
            temp = acc1
        if temp>bestacc:
            bestacc = temp
        if acc_com>bestacc_com:
            bestacc_com = acc_com
            torch.save(net1.state_dict(), save_path1)
            torch.save(net2.state_dict(), save_path2)

    print("The bes acc is " + str(bestacc)+"\t"+"The bes acc_com is "+str(bestacc_com)+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=r"G:\PLN-DPM\dataset\train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='G:\swintransformer-garson-master\swin_base_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)