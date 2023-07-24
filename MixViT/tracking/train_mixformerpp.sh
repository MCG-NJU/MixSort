# There are the detailed training settings for MixFormer, MixFormer-L and MixFormer-1k.
# 1. download pretrained CvT models (CvT-21-384x384-IN-22k.pth/CvT-w24-384x384-IN-22k.pth/CvT-21-384x384-IN-1k.pth) at https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

# server 13
#python tracking/train.py --script mixformerpp_vit_multi --config baseline_large_more_data --save_dir /data0/cyt/experiments/MixFormerPP/mixformerpp_vit_large24-add_tnlAvid_mae_pretrain-freeze_pos_emb-ciou_s384_t192_4.5_bs12_blr0.1_lr4e-4_ep100_it90000-accum3-amp --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformerpp_vit_multi --config baseline_large_score --save_dir /data0/cyt/experiments/MixFormerPP/mixformerpp-score_vit_large24-add_tnlAvid-freeze_pos_emb-ciou_s384_t192_4.5_bs12_blr0.1_lr4e-4_ep100-accum3-amp --mode multiple --nproc_per_node 8 \
#  --stage1_model /data0/cyt/experiments/MixFormerPP/models/MixFormer-addtnl-large24_384_ep0090.pth.tar


### MixFormer扩展期刊实验
# 1. MixFormer-ViT-288_128_4.5  train_from_scratch  lr_backbone:1.0
#python tracking/train.py --script mixformer_vit --config baseline_scratch --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_vit_base12-scratch-freeze_pos_emb-ciou_s288_t128_4.5_bs32_blr1.0_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 8

# 2. MixFormer-ViT-224_112_4.5  pretrain_MAE  lr_backbone:0.1
#python tracking/train.py --script mixformer_vit --config baseline_224 --save_dir /data3/cyt/experiments/MixFormer_jour/mixformer_vit_base12-mae_pretrain-freeze_pos_emb-ciou_s288_t128_4.5_bs32_blr0.1_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 8

# 3. MixFormer-ViT-320_128_4.5  pretrain_MAE  lr_backbone:0.1
#python tracking/train.py --script mixformer_vit --config baseline_320 --save_dir /home/cyt/experiments/MixFormer_jour/mixformer_vit_base12-mae_pretrain-freeze_pos_emb-ciou_s320_t128_4.5_bs32_blr0.1_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 8

# 4. MixFormer-ViT-224_112_5.0  pretrain_MAE  lr_backbone:0.1
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,6" python tracking/train.py --script mixformer_vit --config baseline_224 --save_dir /data0/cuiyutao/experiments/MixFormer_jour/mixformer_vit_base12-mae_pretrain-freeze_pos_emb-ciou_s224_t128_5.0_bs32_blr0.1_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 6

# 5. MixFormer-ViT-224_112_4.0  pretrain_MAE  lr_backbone:0.1
#python tracking/train.py --script mixformer_vit --config baseline_224 --save_dir /data3/cyt/experiments/MixFormer_jour/mixformer_vit_base12-mae_pretrain-freeze_pos_emb-ciou_s224_t128_4.0_bs32_blr0.1_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 8

# 6. MixFormer-ViT-288_128_5  pretrain_MAE  lr_backbone:0.1
#python tracking/train.py --script mixformer_vit --config baseline_288 --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_vit_base12-mae-freeze_pos_emb-ciou_s288_t128_5.0_bs32_blr0.1_lr4e-4_ep500_it60000-accum1-no_amp --mode multiple --nproc_per_node 8

# 7. MixFormer-ViT-224_144_4.5  pretrain_MAE  lr_backbone:0.1
#python tracking/train.py --script mixformer_vit --config baseline_288 --save_dir /data3/cyt/experiments/MixFormer_jour/mixformer_vit_base12-extra_sear_tar_emb-mae_pretrain-freeze_pos_emb-ciou_s288_t128_4.5_bs32-lr6e-4-ep400_it60000-accum1-amp --mode multiple --nproc_per_node 8


#python tracking/train.py --script mixformerpp_vit_multi --config baseline_384 --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_vit_multi_base12-mae_pretrain-freeze_pos_emb-ciou_s384_t192_5.0_bs24-lr4e-4-ep300_it60000-accum1-amp --mode multiple --nproc_per_node 8

#python tracking/train.py --script mixformer_vit --config baseline_288_decoder --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_vit_base12-decoder2-mae_pretrain-freeze_pos_emb-ciou_s288_t128_4.5_bs32-lr6e-4-ep400_it60000-accum1-amp --mode multiple --nproc_per_node 8

python tracking/train.py --script mixformerpp_vit_multi --config baseline_288_ablation_10layers --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_ablation_mae_finetune_base-freeze_pos_emb-ciou_s288_t128_5.0_bs32_ep500-accum1-amp --mode multiple --nproc_per_node 8


#python tracking/train.py --script mixformerpp_vit_multi --config baseline_288_got_score --save_dir /home/cyt/experiments/MixFormer_jour/got-score-mixformer-vit_base-upheadv2-mae-freeze_pos_emb-ciou_s288_t128_5.0_bs64lr1e-4_ep40-accum1-noamp --mode multiple --nproc_per_node 8 --stage1_model /home/cyt/experiments/MixFormer_jour/models/MixFormer-ViT-got-uphead-B-multi-288-ep500.pth.tar

#python tracking/train.py --script mixformerpp_vit_multi --config baseline_288_ablation_10layers --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_ablation_mae_4layers_21k_base12-mae-freeze_pos_emb-ciou_s288_t128_5.0_bs32_ep500-accum1-amp --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformerpp_vit_multi --config baseline_large --save_dir /data0/cyt/experiments/MixFormer_jour/mixformer_vit_large24-uphead_v2-mae-freeze_pos_emb-ciou_s384_t192_4.5_bs12_ep500-accum3-amp --mode multiple --nproc_per_node 8

#python tracking/train.py --script mixformerpp_vit_multi --config baseline_uphead --save_dir /data3/cyt/experiments/MixFormer_jour/mixformer_vit_base-uphead_v2-mae-freeze_pos_emb-ciou_s288_t128_5.0_bs12_ep500-accum3-noamp --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformerpp_vit_multi --config baseline_288_score --save_dir /data3/cyt/experiments/MixFormer_jour/mixformer-score_vit_base-upheadv2-mae-freeze_pos_emb-ciou_s288_t128_5.0_bs64_lr1e-4_ep40-accum1-noamp --mode multiple --nproc_per_node 8 --stage1_model /data3/cyt/experiments/MixFormer_jour/models/MixFormer-ViT-uphead-B-multi-288-ep0496.pth.tar
#python tracking/train.py --script mixformerpp_vit_multi --config baseline_288 --save_dir /home/cyt/experiments/MixFormer_jour/mixformer-vit_base-upheadv2-mae-freeze_pos_emb-ciou_s288_t128_4.5_bs32_lr6e-4_ep500-accum1-amp --mode multiple --nproc_per_node 8