import os

base_dir = '/phoenix/S2/snavely/data/Awais/'

out_dir = '/phoenix/S7/kz298/AppleDiseaseClassification/data_split'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

label_remappings = dict([('healthy', 'Healthy'),
                       ('rust', 'CedarAppleRust'),
                       ('scab', 'Scab'),
                       ('combination', 'Combinations')])


test_list = []
with open(os.path.join(base_dir, 'test_data_labels.csv')) as fp:
    for line in fp.readlines():
        line = line.strip()
        if line:
            label, img_name = line.split(',')
            label = label_remappings[label]

            fpath = os.path.join(base_dir, 'TestData/1600px', img_name[:-4] + '.jpg')

            if os.path.exists(fpath):
                data_item = (label, fpath)
                test_list.append(data_item)
            else:
                print('no file: {}'.format(fpath))


def write_to_file(fpath, data_list):
    with open(fpath, 'w') as fp:
        for data_item in data_list:
            label, fpath = data_item
            fp.write('{}\t{}\n'.format(label, fpath))


write_to_file(os.path.join(out_dir, 'test.txt'), test_list)


print('# test items: {}'.format(len(test_list)))
