from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
# from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

from dataset import name2label_dict, label2name_dict, build_dataloader
from network import ResNet


def validate_model(writer, model, data_loader, step):
    print('validating mdoel...')
    model.eval()

    all_gt = []
    all_pred = []
    for batch_idx, item in enumerate(data_loader):
        im = item['im'].cuda()
        gt = item['label'].squeeze(1).cpu().numpy()
        all_gt.extend(list(gt))
        with torch.no_grad():
            res = model(im)
            pred = torch.argmax(res, dim=1, keepdim=False).cpu().numpy()
            all_pred.extend(list(pred))

    report = classification_report(all_gt, all_pred, target_names=name2label_dict.keys())
    print(report)

    accuracy = accuracy_score(all_gt, all_pred)
    p_scores = precision_score(all_gt, all_pred, average=None)
    r_scores = recall_score(all_gt, all_pred, average=None)
    f_scores = f1_score(all_gt, all_pred, average=None)

    writer.add_scalar('Val/avg_accuracy', accuracy, step)
    writer.add_scalar('Val/avg_precision', np.mean(p_scores), step)
    writer.add_scalar('Val/avg_recall', np.mean(r_scores), step)
    writer.add_scalar('Val/avg_fscore', np.mean(f_scores), step)

    return accuracy, report, all_pred


def train_model(output_dir, exp_id):
    log_dir = os.path.join(output_dir, exp_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    model = ResNet(out_dim=4, encoder='resnet50', pretrained=True).cuda()

    init_lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    train_batch_size = 16
    train_loader = build_dataloader(batch_size=train_batch_size, mode='train_val', num_workers=8)
    val_loader = build_dataloader(batch_size=16, mode='test', num_workers=8)

    # weight = torch.Tensor([0.1438004567630554, 0.08892753894551772, 0.10387256375786466, 0.6633994405335623]).cuda()
    # loss_fn = nn.CrossEntropyLoss(weight=weight)

    loss_fn = nn.CrossEntropyLoss()

    total_epochs = 120

    # logging options
    model_save_interval_epoch = 10
    lr_decay_factor = 0.5
    lr_decay_interval_epoch = 15

    # start training
    total_cnt = len(train_loader.dataset)
    num_steps_per_epoch = total_cnt // train_batch_size

    # best accuracy
    best_accuracy = 0

    step = 0
    for epoch in range(1, total_epochs + 1):
        model.train()

        processed_cnt = 0
        for batch_idx, item in enumerate(train_loader):
            step += 1

            # zero out gradient
            optimizer.zero_grad()

            im = item['im'].cuda()
            label = item['label'].squeeze(1).cuda()
            res = model(im)
            loss = loss_fn(res, label)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            writer.add_scalar('Train/loss', loss, step)

            batch_size = im.shape[0]
            processed_cnt += batch_size

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.8f}'.format(
                epoch, processed_cnt, total_cnt, 100. * processed_cnt / total_cnt, loss,
                optimizer.param_groups[0]['lr']))

            # validate model
            if step % (num_steps_per_epoch // 3) == 0:
                accuracy, report, all_pred = validate_model(writer, model, val_loader, step)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print('best_accuracy updated to: {}'.format(best_accuracy))

                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
                    with open(os.path.join(log_dir, 'best_metric.txt'), 'w') as fp:
                        fp.write(report)

                    with open(os.path.join(log_dir, 'best_pred.txt'), 'w') as fp:
                        for pred in all_pred:
                            fp.write('{}\n'.format(label2name_dict[pred]))

        # whether to save the current model
        if epoch % model_save_interval_epoch == 0 or epoch == total_epochs:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))

        # whether to decay learning rate
        if lr_decay_factor is not None and lr_decay_interval_epoch is not None:
            if epoch % lr_decay_interval_epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay_factor


if __name__ == '__main__':
    output_dir = './runs'
    train_model(output_dir=output_dir, exp_id='resnet50_baseline')
