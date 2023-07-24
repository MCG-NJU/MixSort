# There are the detailed training settings for MixFormer, MixFormer-L and MixFormer-1k.
# 1. download pretrained CvT models (CvT-21-384x384-IN-22k.pth/CvT-w24-384x384-IN-22k.pth/CvT-21-384x384-IN-1k.pth) at https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-22k
# Stage1: train mixformer without SPM
#python tracking/train.py --script mixformer --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8
## Stage2: train mixformer_online, i.e., SPM (score prediction module)
#python tracking/train.py --script mixformer_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-L-22k
#python tracking/train.py --script mixformer --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-1k
#python tracking/train.py --script mixformer --config baseline_1k --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_1K --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_1K_ONLINE --mode multiple --nproc_per_node 8 \
#     --stage1_model /STAGE1/MODEL


### Training MixFormer-22k_GOT
#python tracking/train.py --script mixformer --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT_ONLINE --mode multiple --nproc_per_node 8 \
#    --stage1_model /STAGE1/MODEL

# server-13
#python tracking/train.py --script mixformer_vit --config baseline --save_dir /data0/cyt/experiments/trackmae_base_256_bs32 --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit --config baseline_large --save_dir /data0/cyt/experiments/trackmae/trackmae_pretrain_base_384_5_bs32_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_multi --config baseline_large --save_dir /data0/cyt/experiments/trackmae/test-trackmae_multi_pretrain_base12_sear384_temp192_f5.0_bs32_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_multi --config baseline_large_score \
#  --save_dir /data0/cyt/experiments/trackmae/mixformer_vit_imagemae_multi_score_base12_sear384_temp192_f5.0_bs32_blr0.1_lr4e-4 \
#  --mode multiple --nproc_per_node 8 --stage1_model /data0/cyt/experiments/trackmae/models/MixFormer_vit_multi_mae_ep0495.pth

# server-08
#python tracking/train.py --script mixformer_vit --config baseline --save_dir /data3/cyt/experiments/trackmae/my_trackmae_ep380_pretrain_base_giou_sear288_4.5_bs16_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_multi --config baseline_large --save_dir /data3/cyt/experiments/trackmae/trackmae_pretrain_large12_sear432_temp192_f4.5_bs6_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8

#python tracking/train.py --script mixformer_vit --config baseline --save_dir /home/cyt/experiments/trackmae/my_trackmae_v2_ep400_pretrain_base_giou_sear288_4.5_bs32_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8

#python tracking/train.py --script mixformer_vit --config masked_vit --save_dir /data0/cyt/experiments/trackmae/my_trackmae_v1_ep400_pretrain-masked_vit-giou_sear288_4.5_bs32_blr0.1_lr4e-4 --mode multiple --nproc_per_node 8
python tracking/train.py --script mixformer --config baseline --save_dir /home/cyt/experiments/MixFormer_jour/mixformer_ablation_dwconv_multistage-ciou_sear288_4.5_bs32_blr0.5_lr4e-4-amp --mode multiple --nproc_per_node 8