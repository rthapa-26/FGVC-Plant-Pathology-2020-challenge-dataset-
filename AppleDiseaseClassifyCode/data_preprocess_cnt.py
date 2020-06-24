import os
import numpy as np
from dataset import name2label_dict, label2name_dict

data_list_file = '/phoenix/S7/kz298/AppleDiseaseClassification/data_split/train_val.txt'

all_labels = []
with open(data_list_file) as fp:
    for line in fp.readlines():
        line = line.strip()
        if line:
            label, fpath = line.split('\t')
            all_labels.append(name2label_dict[label])

all_labels = np.array(all_labels)

ratio = []
for i in range(4):
    ratio.append(np.sum(all_labels == i) / len(all_labels))
ratio = np.array(ratio)

for i in range(4):
    print('{}: {}'.format(label2name_dict[i], ratio[i]))


inv_ratio = 1.0 / ratio
inv_ratio = inv_ratio / np.sum(inv_ratio)
for i in range(4):
    print('{}: {:.4f}'.format(label2name_dict[i], inv_ratio[i]))

print(inv_ratio.tolist())