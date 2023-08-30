import numpy as np
import json
import random

fic = open('/storage/home/hcocice1/sarora320/wstm/NGNN/data/category_summarize_100.json', 'r', encoding='utf-8')
category_item = json.load(fic)

num_category = len(category_item)

fr = open('../data/cid2rcid_100.json', 'r', encoding='utf-8')

cid2rcid = json.load(fr)

# image_feature_path = '../data/polyvore_image_vectors/'
image_feature_path = '/storage/home/hcocice1/sarora320/wstm/NGNN/data/polyvore_image_vectors-001/'


total_graph = np.zeros((num_category, num_category), dtype=np.float32)  # 构建一个item与item之间的图关系
per_outfit = 8
def load_graph():
    return total_graph

def load_num_category():
    return num_category

def load_train_data(i, batch_size, outfit_list):
    size_ = batch_size
    time = int(batch_size / per_outfit)
    i = i * time
    image_pos = np.zeros((size_, num_category, 2048), dtype=np.float32)
    image_neg = np.zeros((size_, num_category, 2048), dtype=np.float32)
    now_size = 0
    for outfit in outfit_list[i: i + time]:
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        len_ = len(ii)
        j_list = []
        for j in range(len_):
            j_list.append(j)
        for j in range(per_outfit - len_):
            j_list.append(random.randint(0, len_ - 1))
        for j in j_list:
            list_ = []
            for k in range(len(ii)):
                cid = ci[k]
                iid = ii[k]
                rcid = int(cid2rcid[str(cid)])
                image_feature = json.load(
                    open(image_feature_path + str(sid) + '_' + str(iid) + '.json', encoding='utf-8'))  # 读取对应每件item的图像特征
                if k == j:
                    image_pos[now_size][rcid] = image_feature

                    i = random.choice(category_item)
                    rcid_neg = cid2rcid[str(i['id'])]
                    ri = random.choice(i['items'])

                    image_neg[now_size][rcid_neg] = json.load(open(image_feature_path + ri + '.json', encoding='utf-8'))
                else:
                    image_neg[now_size][rcid] = image_feature
                    image_pos[now_size][rcid] = image_feature
                    list_.append(rcid)
            now_size += 1
    return image_pos, image_neg, size_


def load_train_size():
    ftrain = open('/storage/home/hcocice1/sarora320/wstm/NGNN/data/train_no_dup_new_100.json', 'r', encoding='utf-8')
    train_list = json.load(ftrain)
    train_size = len(train_list)
    train_size_ = per_outfit * train_size
    return train_size, train_size_



def load_fitb_data(index, batch_size_test, outfit_list):
    time = int(batch_size_test / 4)
    num_category = load_num_category()

    image = np.zeros((batch_size_test, num_category, 2048), dtype=np.float32)
    outfit_list_ = outfit_list[index * time: (index + 1) * time]
    for i in range(len(outfit_list_)):
        outfit = outfit_list_[i]
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        rcid_list = []
        length = len(ii)
        blank_index = random.randint(0, length - 1)
        for j in range(len(ii)):
            cid = ci[j]
            iid = ii[j]
            rcid = int(cid2rcid[str(cid)])
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json', encoding='utf-8'))

            if j == blank_index:
                image[i * 4][rcid] = image_feature
                r1 = random.choice(category_item)
                rcid_w1 = cid2rcid[str(r1['id'])]
                r11 = random.choice(r1['items'])
                image[i * 4 + 1][rcid_w1] = json.load(open(image_feature_path + r11 + '.json', encoding='utf-8'))

                r2 = random.choice(category_item)
                rcid_w2 = cid2rcid[str(r2['id'])]
                r22 = random.choice(r2['items'])
                image[i * 4 + 2][rcid_w2] = json.load(open(image_feature_path + r22 + '.json', encoding='utf-8'))

                r3 = random.choice(category_item)
                rcid_w3 = cid2rcid[str(r3['id'])]
                r33 = random.choice(r3['items'])
                image[i * 4 + 3][rcid_w3] = json.load(open(image_feature_path + r33 + '.json', encoding='utf-8'))

            else:
                rcid_list.append(rcid)
                image[i * 4][rcid] = image_feature
                image[i * 4 + 1][rcid] = image_feature
                image[i * 4 + 2][rcid] = image_feature
                image[i * 4 + 3][rcid] = image_feature
    return image


def load_test_size():
    ftest = open('/storage/home/hcocice1/sarora320/wstm/NGNN/data/test_no_dup_new_100.json', 'r', encoding='utf-8')
    test_list = json.load(ftest)
    test_size = len(test_list)
    return test_size


def load_auc_data(index, batch_size_auc, outfit_list):
    time = int(batch_size_auc / 2)
    num_category = load_num_category()
    image = np.zeros((batch_size_auc, num_category, 2048), dtype=np.float32)
    outfit_list_ = outfit_list[index * time: (index + 1) * time]
    for i in range(len(outfit_list_)):
        outfit = outfit_list_[i]
        ii = outfit['items_index']
        ci = outfit['items_category']
        sid = outfit['set_id']
        length = len(ii)
        graph_pos = []
        graph_neg = []
        for j in range(length):
            cid = ci[j]
            iid = ii[j]
            rcid = int(cid2rcid[str(cid)])
            image_feature = json.load(open(image_feature_path + str(sid) + '_' + str(iid) + '.json', encoding='utf-8'))
            image[i * 2][rcid] = image_feature
            graph_pos.append(rcid)
        for j in range(length):
            r = random.choice(category_item)
            rcid_w = cid2rcid[str(r['id'])]
            rr = random.choice(r['items'])
            image[i * 2 + 1][rcid_w] = json.load(open(image_feature_path + rr + '.json', encoding='utf-8'))
            graph_neg.append(rcid_w)


    return image

def load_valid_size():
    fvalid = open('/home/ndn/桌面/HyperGCN/data/valid_no_dup_hyper.json', 'r', encoding='utf-8')
    valid_list = json.load(fvalid)
    valid_size = len(valid_list)
    return valid_size

