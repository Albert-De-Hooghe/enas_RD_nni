
from argparse import ArgumentParser

import logging

from nni.nas.pytorch import apply_fixed_architecture

from sklearn.metrics import cohen_kappa_score
from constants import COMPLEMENT_FOLDER_NAME_Retrain
from macro import GeneralNetwork
from readRD_dataset import RD_Dataset_valid_5_classes, RD_Dataset_train_5_classes
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from nni.nas.pytorch.utils import AverageMeter
import utils

def train(config, train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    # top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):

        print("step {}".format(step))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        # print(z, len(z))
        # try
        logits = model(x)
        # logits = logits.reshape(-1).cuda()
        # y = y.type(torch.cuda.FloatTensor)
        # break
        # except ValueError:
        #
        #     print("path is :{}".format(z))
        # print(logits, y)
        # var_list = []
        # for i in range(len(logits)):
        #     var_list.append(torch.argmax(logits[i]))
        # logits = torch.tensor(var_list)
        # logits = logits.to(device)
        # logits = logits.float()
        print("_____")
        print(logits, y)

        loss = criterion(logits, y)

        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        # top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        # writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    # top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()
    list_labels_validation = []
    list_prediction_validation = []
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(m(logits), y)

            for i in range(len(X)):
                #print("logits are :", logits)
                #print("y is :", y)
                #print("liste des labels for Kappa:",  list_labels_validation)
                #print("list des prediction fo kappa: ", list_prediction_validation)
                list_labels_validation.append(y[i].item())
                list_prediction_validation.append(torch.argmax(logits[i]).item())

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            # top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1))
    kappavalue = cohen_kappa_score(list_labels_validation, list_prediction_validation, weights='quadratic')

    writer.add_scalar("kappa/validation", kappavalue, global_step=cur_step / len(train_loader))
    writer.add_scalar("loss/validation", losses.avg, global_step=cur_step / len(train_loader))
    writer.add_scalar("acc1/validation", top1.avg, global_step=cur_step / len(train_loader))
    # writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg

if __name__ == "__main__":

    logger = logging.getLogger('nni')

    learning_rate = 0.008
    epoch_choisie_pour_retrain = 3808 ### 3808 best

    complement_nom_folder_LR = '_LR_' + str(learning_rate).replace('.', '_') + '/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('outFiles/retrain_first_try'+complement_nom_folder_LR)
    m = nn.Sigmoid()
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./logs/checkpoints_architecture/"+COMPLEMENT_FOLDER_NAME_Retrain+"/"+"epoch_"+ str(epoch_choisie_pour_retrain) +".json")

    args = parser.parse_args()
    size = "search"
    dataset_train, dataset_valid = RD_Dataset_train_5_classes(taille=size), RD_Dataset_valid_5_classes(taille=size)
    #datasets.get_dataset("cifar10", cutout_length=16)

    # for i in range(95):
    #     try:
    #         model = CNN(224, 3, 36, 5, args.layers, auxiliary=True)
    #         apply_fixed_architecture(model, "./checkPoints_150_epochs/epoch_" + str(i) + ".json")
    #         criterion = nn.CrossEntropyLoss()
    #
    #         model.to(device)
    #         criterion.to(device)
    #     except Exception:
    #         print(Exception)
    #         print("etat de la boucle :",i)



    model = GeneralNetwork(num_classes=5)

    apply_fixed_architecture(model, args.arc_checkpoint)




    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    model.to(device)
    criterion.to(device)

    print(model)

    # 10-3, 10-4 , 10-5, 10-6
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    best_top1 = 0.
    for epoch in range(args.epochs):
        # drop_prob = args.drop_path_prob * epoch / args.epochs
        # model.drop_path_prob(drop_prob)

        # training
        train(args, train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(args, valid_loader, model, criterion, epoch, cur_step)
        best_top1 = max(best_top1, top1)

        # print(model.state_dict())
        # Save model
        #torch.save(model.state_dict(), "./savedModels/retrain_20juin_after_looking_search/29/model_epoch{}.pt".format(epoch))
        models_dir = './savedModels/retrain_first_try/' + complement_nom_folder_LR + 'model_'
        model_save_path = '{}epoch{}.pth.tar'.format(models_dir, epoch + 1)
        torch.save(model,
            model_save_path)
        print("model_saved")

        # print("model restore")
        # model16 = CNN(224, 3, 36, 5, 8, auxiliary=True)
        # model16 = torch.load(model_save_path)
        # print("model restored")

        lr_scheduler.step()

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
