import numpy as np
import json
import random

fit = open('/data/polyvore/train_no_dup.json', 'r')
# fit = open('/data/polyvore/test_no_dup.json', 'r')
# fit = open('/data/polyvore/valid_no_dup.json', 'r')
mmm = json.load(fit)

fittt = open('/data/category_summarize_100.json', 'r')
cat_100 = json.load(fittt)
category = []
for cat in cat_100:
	category.append(cat['id'])

train_new_hy = []
for outfits in mmm:
	outfit = outfits['items']
	t = []
	for m in outfit:
		if m['categoryid'] in category:
			t.append(m)
	outfits['items'] = t
	train_new_hy.append(outfits)

t = []
for outfit in train_new_hy:
	items_category = []
	items_index = []
	train_var = {}
	for item in outfit['items']:
		if item['categoryid'] not in items_category:
			items_category.append(item['categoryid'])
			items_index.append(item['index'])
	train_var['items_category'] = items_category
	train_var['items_index'] = items_index
	train_var['set_id'] = outfit['set_id']
	t.append(train_var)

train_list = []
for outfit in t:
	length = len(outfit['items_index'])
	if length >= 3:
		train_list.append(outfit)

# with open('/data/train_no_dup_hyper.json', 'w') as f1:
#     json.dump(train_new_hy, train_list)
