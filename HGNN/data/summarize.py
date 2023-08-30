import json


f1 = open('/data/polyvore/train_no_dup.json', 'r')
train_list = json.load(f1)
f2 = open('/data/polyvore/valid_no_dup.json', 'r')
valid_list = json.load(f2)
f3 = open('/data/polyvore/test_no_dup.json', 'r')
test_list = json.load(f3)


dict_list = []
f2 = open('/data/category_id.txt', 'r')
line = f2.readline()
while line:
    l = line.split(' ', 1)
    dict = {}
    dict['id'] = int(l[0])
    dict['name'] = l[1].rstrip("\n")
    dict['frequency'] = 0
    dict['items'] = []
    dict_list.append(dict)
    line = f2.readline()


for i in test_list:
    dict = {}
    item_list = i['items']
    for j in item_list:
        category_id = j['categoryid']
        for k in dict_list:
            if k['id'] == category_id:
                m =str(i['set_id'])+'_'+str(j['index'])
                k['items'].append(m)
                k['frequency'] += 1

print ('total category: %d'% len(dict_list))

count_100 = 0
dict_list_100 = []
cate_list_100 = []
for i in dict_list:
    if i['frequency'] >= 100:
        dict_list_100.append(i)
        cate_list_100.append(int(i['id']))
        count_100 += 1

print ('more than 100: %d'% count_100)
cate_list_100 = sorted(cate_list_100)
print (cate_list_100)
cid2rcid = {}
for i in range(len(cate_list_100)):
    cid2rcid[int(cate_list_100[i])] = i


with open("/data/cid2rcid_100.json", "w") as f4:
    json.dump(cid2rcid, f4)

with open("/data/category_summarize_100.json", "w") as f5:
    json.dump(dict_list_100, f5)



