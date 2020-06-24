from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
# from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

from dataset import name2label_dict, label2name_dict, build_dataloader
from network import ResNet


def validate_model(model, data_loader, out_file=None):
    print('validating mdoel...')
    model.eval()

    all_fnames = []
    all_gt = []
    all_pred = []
    for batch_idx, item in enumerate(data_loader):
        im = item['im'].cuda()
        #fname = item['fname']
        # print(type(fname), fname)
        all_fnames.extend(item['fname'])

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

    all_gt = [label2name_dict[x] for x in all_gt]
    all_pred = [label2name_dict[x] for x in all_pred]
    if out_file is not None:
        with open(out_file, 'w') as fp:
            fp.write('filename\tground-truth\tprediction\n')
            for i in range(len(all_fnames)):
                fp.write('{}\t{}\t{}\n'.format(all_fnames[i],
                                               all_gt[i], 
                                               all_pred[i]))
    return accuracy, report, all_pred


def load_pretrained(model, pretrained):
    # handle the case when model is wrapped inside nn.DataParallel
    try:
        state = model.module.state_dict()
    except AttributeError:
        state = model.state_dict()

    state.update(torch.load(pretrained, map_location=lambda storage, loc: storage))
    model.load_state_dict(state)

    return model


if __name__ == '__main__':
    model = ResNet(out_dim=4, encoder='resnet50', pretrained=True).cuda()

    pretrained_file = '/phoenix/S7/kz298/AppleDiseaseClassification/runs/resnet50_baseline_all/best_model.pth'
    model = load_pretrained(model, pretrained_file)

    val_loader = build_dataloader(batch_size=16, mode='test', num_workers=8)
    
    out_file = '/phoenix/S7/kz298/AppleDiseaseClassification/test_results.txt'
    validate_model(model, val_loader, out_file)
