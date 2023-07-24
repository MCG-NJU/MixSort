import json
import os

train_json = json.load(open('datasets/SportsMOT/annotations/train.json','r'))

img_list = list()
for img in train_json['images']:
    img['file_name'] = 'train/' + img['file_name']
    img_list.append(img)

ann_list = train_json['annotations']
video_list = train_json['videos']
category_list = train_json['categories']

max_img = len(img_list)
max_ann = len(ann_list)
max_video = len(video_list)

val_json = json.load(open('datasets/SportsMOT/annotations/val.json','r'))
img_id_count = 0
for img in val_json['images']:
    img['file_name'] = 'val/' + img['file_name']
    img['id'] +=  max_img
    if img['prev_image_id'] != -1:
        img['prev_image_id'] += max_img
    if img['next_image_id'] != -1:
        img['next_image_id'] += max_img
    img['video_id'] += max_video
    img_list.append(img)
    
for ann in val_json['annotations']:
    ann['id'] +=  max_ann
    ann['image_id'] +=  max_img
    ann_list.append(ann)

for vid in val_json['videos']:
    vid['id'] += max_video
    video_list.append(vid)

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/SportsMOT/annotations/mix.json','w'))