import os
import pandas
import torch
import torch.optim as optim
import setproctitle
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.transforms import transforms
from tqdm import tqdm
# from testnet.day1221 import testnet2, testnet4, testnet5, testnet6, testnet7
from net.DCCMNet import *
from tools import metrics
from tools.augment import get_train_augmentation, get_val_augmentation
from tools.dataset import Dataset, Datasesloader
from tools.loss import SoftDiceLoss
from tools.makedir import makedir
from tools.sort import *
from tools.seed import set_seed
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


def train(size, name=None, NetName="DCCMNet", dataset="Deepcrack", outnum=1):
    set_seed(727392)
    global model, pred_output

    epoch = 150
    val_gap_num = 1
    val_gap_mod = 0
    init_lr = 0.001

    batchsize = 2
    weight_decay = 0.00001
    cfgsize = size
    # pretrain=True
    pretrain = False

    momentum = 0.9
    decay_factor = 0.5

    if dataset == "Deepcrack":
        root = "./data(Deepcrack)/"    # Deepcrack root directory
    elif dataset == "crack260":
        root = "..."                   #crack260 root directory
    elif dataset == "CrackForest":
        root = "..."                   #CrackForest root directory

    savepath = ("./runs/%s_%s" %(name, dataset))
    if  os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs("./runs/%s_%s" %(name, dataset), exist_ok=False)

    print("path:", savepath)
    criterion = SoftDiceLoss()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(getgpuid())

    if NetName == "DCCMNet":
        model = DCCMNet()

    # model = torch.nn.DataParallel(model, device_ids=[0])

    if pretrain:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['net'], strict=False)
        num_loaded_params = len(model.state_dict())
        print(f"加载:{num_loaded_params}")

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # optimizer=optim.SGD(model.parameters(),lr=init_lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=300, gamma=decay_factor)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    trainset = Datasesloader(root, savepath=savepath, txt="train.txt")
    # trainset = Dataset(root, mode="train.txt", savepath=savepath)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                               shuffle=True, num_workers=4, drop_last=True)

    # testset = Dataset(root, mode="test.txt", savepath=savepath)
    testset = Datasesloader(root, savepath=savepath, txt="test.txt")
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=4, drop_last=True)

    best_loss = 0
    best_acc = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_iou = 0
    best_miou = 0
    best_epoch = 0

    for it in range(epoch):
        model.train()
        # if it==29:
        #     scheduler.step()
        # lr = scheduler.get_lr()[0]

        lr = scheduler.get_lr()[0]
        # print("lr:",lr)
        loss = 0
        bar1 = tqdm(enumerate(train_loader), total=len(train_loader))
        bar1.set_description('Epoch %d --- Training --- :' % it)
        iou = 0
        miou = 0
        # for idx, batch in enumerate(train_loader):
        for idx, (img, label) in bar1:
            # img = batch[0].to(device)
            # label = batch[1].to(device)
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            if outnum == 2:
                pred_output, out1 = model(img)
                loss1 = criterion(pred_output.view(-1, 1), label.view(-1, 1))
                loss2 = criterion(out1.view(-1, 1), label.view(-1, 1))
                output_loss = loss1 + loss2
                loss += output_loss
                output_loss.backward()
                optimizer.step()
            elif outnum == 1:
                pred_output = model(img)
                output_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1))

                loss += output_loss

                output_loss.backward()
                optimizer.step()

            pred_output = torch.sigmoid(pred_output)
            i, _ = metrics.iou_score(pred_output, label)
            iou += i
            mi = metrics.miou(pred_output, label)
            miou += mi
        scheduler.step()

        # trainlosslist = [lr, loss.item() / len(train_loader) * 100]
        # print("trainloss", trainlosslist)
        # lr,loss,iou,miou
        trainlist = [lr, loss.item() / len(train_loader) * 100, iou / len(train_loader) * 100,
                     miou / len(train_loader) * 100]
        # print("Epoch %d,train:" % it, trainlist)
        data_trainloss = pandas.DataFrame([trainlist])
        data_trainloss.to_csv(savepath + '/trainloss.csv', mode='a', header=False, index=False)
        if it % val_gap_num == val_gap_mod:
            bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
            bar2.set_description('Epoch %d --- eval --- :' % it)
            model.eval()
            with torch.no_grad():
                loss = 0
                acc = 0
                precision = 0
                recall = 0
                f1 = 0
                iou = 0
                miou = 0
                for idx, (img, label) in bar2:
                    img = img.to(device)
                    label = label.to(device)
                    if outnum == 2:
                        pred_output, pout1 = model(img)
                        val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) + criterion(pout1.view(-1, 1),
                                                                                                     label.view(-1, 1))

                    elif outnum == 1:
                        pred_output = model(img)
                        val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1))

                    loss += val_loss

                    pred = torch.sigmoid(pred_output)
                    # pred = pred_output
                    ac, p, r, f, = metrics.f1_loss(label[0], pred)
                    acc += ac
                    precision += p
                    recall += r
                    f1 += f
                    i, _ = metrics.iou_score(pred, label[0])
                    iou += i
                    mi = metrics.miou(pred, label[0])
                    miou += mi

                # [acc,precision,recall,f1,iou,mIoU]
                l = len(val_loader)

                f1 = precision / l * 100 * recall / l * 100 * 2 / (precision / l * 100 + recall / l * 100)

                acclist = [it, acc / l * 100, precision / l * 100, recall / l * 100, f1 , iou / l * 100,
                           miou / l * 100, loss.item() / l * 100]
                vallosslist = [loss.item() / len(val_loader) * 100]

                checkpoint = {
                    "net": model.state_dict(),
                }
                if not os.path.exists(savepath + "/models"):
                    os.mkdir(savepath + "/models")

                if (best_miou < (miou / l * 100) or best_f1 < f1  or best_recall < (
                        recall / l * 100)) and it > 30:
                    torch.save(checkpoint, savepath + "/models/model" + str(it) + ".pth")

                if best_miou < (miou / l * 100):
                    best_loss = loss.item() / l * 100
                    best_acc = acc / l * 100
                    best_precision = precision / l * 100
                    best_recall = recall / l * 100
                    best_f1 = f1
                    best_iou = iou / l * 100
                    best_miou = miou / l * 100
                    best_epoch = it

                bestlist = [best_acc, best_precision, best_recall, best_f1, best_iou, best_miou, best_loss]

                # print("valloss:", vallosslist)
                data_valloss = pandas.DataFrame([vallosslist])
                data_valloss.to_csv(savepath + '/valloss.csv', mode='a', header=False, index=False)

                print("Epoch %d,train:" % it, trainlist)
                print("Epoch %d:" % it, acclist)
                print("Best_Epoch %d:" % best_epoch, bestlist)
                data_acc = pandas.DataFrame([acclist])
                data_acc.to_csv(savepath + '/acc.csv', mode='a', header=False, index=False)

            # if it>=(epoch-save_model_num):
        sortresult(savepath + "/acc.csv")


if __name__ == '__main__':

    train(512, name="DCCMNet", NetName="DCCMNet", dataset="Deepcrack")
