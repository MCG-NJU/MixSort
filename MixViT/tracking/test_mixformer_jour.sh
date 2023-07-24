# Different test settings for MixFormer-22k, MixFormerL-22k & MixFormer-1k on LaSOT/TrackingNet/GOT10k/UAV123/OTB100
# First, put your trained MixFormer-online models on SAVE_DIR/models directory. ('vim lib/test/evaluation/local.py' to set your SAVE_DIR)
# Then, uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH. (see 'lib/test/evaluation/local.py')

##########-------------- MixFormer-22k-----------------##########
### LaSOT test and evaluation
#python tracking/test.py mixformer_online baseline --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.55
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

### TrackingNet test and pack
#python tracking/test.py mixformer_online baseline --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_online --cfg_name baseline

### MixFormer-22k-got-only GOT10k test and pack
#python tracking/test.py mixformer_online baseline --dataset got10k_test --threads 32 --num_gpus 8 --params__search_area_scale 4.55 \
#  --params__model mixformer_online_22k_got.pth.tar --params__max_score_decay 0.98
#python lib/test/utils/transform_got10k.py --tracker_name mixformer_online --cfg_name baseline

### UAV123
#python tracking/test.py mixformer_online baseline --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py mixformer_online baseline --dataset otb --threads 28 --num_gpus 7 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline

#python tracking/test.py mixformer_vit_online baseline_large_score --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_vit_score_imagemae.pth.tar --params__search_area_scale 5.0
#rm -r /data3/cyt/experiments/trackmae/test
#python tracking/test.py mixformer_vit baseline --dataset lasot --threads 32 --num_gpus 8 --params__model trackmae_finetune_base_v1_ep500.pth --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

#python tracking/test.py mixformer_vit baseline_224 --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-vit-base-mae_224_4.0_ep0450.pth.tar  --params__search_area_scale 4.5 --params__online_sizes 1
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_224

# CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"
#python tracking/test.py mixformer_vit_decoder baseline_288_decoder --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-vit-decoder-mae-288_128_4.5-ep0370.pth.tar  --params__search_area_scale 4.5 --params__online_sizes 1
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_288_decoder

#python tracking/test.py mixformerpp_vit_online baseline_384 --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-ViT-B-multi-384-ep0300.pth.tar --params__search_area_scale 4.5

### MixViT-L 所有数据集got10k最佳结果
#for sear in 4.7
#do
#  for skip in 25
#  do
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset got10k_test --threads 1 --num_gpus 1 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale ${sear} \
#      --runid best_got_ep37_sear${sear}_skip${skip} --params__update_interval ${skip} --params__online_sizes 2 --params__max_score_decay 0.98
#    python lib/test/utils/transform_got10k.py --tracker_name mixformerpp_vit_online --cfg_name baseline_large_score_best_got_ep37_sear${sear}_skip${skip}
##    python lib/test/utils/transform_trackingnet.py --tracker_name mixformerpp_vit_online --cfg_name baseline_288_score_skip25_os${os}_sear${sear}
#  done
#done

### MixViT-L LaSOT 最佳结果
#for sear in 4.55
#do
#  for os in 2
#  do
#    skip=200
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale ${sear} \
#      --runid lasot_sear${sear}skip${skip}os${os} --params__update_interval ${skip}  --params__online_sizes ${os}
#    python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large_score --run_ids lasot_sear${sear}skip${skip}os${os}
#  done
#done

### MixViT-L OTB最佳结果
#for sear in 4.6
#do
#  for os in 5
#  do
#    skip=12
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset otb --threads 32 --num_gpus 8 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale ${sear} \
#    --runid otb_sear${sear}skip${skip}os${os} --params__update_interval ${skip}  --params__online_sizes ${os}
#    python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large_score --run_ids otb_sear${sear}skip${skip}os${os}
#  done
#done

### MixViT-L UAV最佳结果
#for sear in 4.6 4.55 4.7
#do
#  for skip in 25
#  do
#    os=2
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset uav --threads 32 --num_gpus 8 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale ${sear} \
#    --runid uav_sear${sear}skip${skip}os${os} --params__update_interval ${skip}  --params__online_sizes ${os}
#    python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large_score --run_ids uav_sear${sear}skip${skip}os${os}
#  done
#done

### MixViT-B UAV123最佳结果 68.1
#for sear in 4.7
#do
#  for skip in 20
#  do
#    python tracking/test.py mixformerpp_vit_online baseline_288_score --dataset uav --threads 32 --num_gpus 8 --params__model MixFormer-ViT-uphead-B-multi-288-score-ep50.pth.tar --params__search_area_scale ${sear} \
#      --runid uav_sear${sear}skip${skip} --params__update_interval ${skip} --params__online_sizes 3
#    python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_288_score --run_ids uav_sear${sear}skip${skip}
#  done
#done

### MixViT-B OTB最佳结果71.6
#for sear in 4.45
#do
#  for skip in 6
#  do
##    python tracking/test.py mixformerpp_vit_online baseline_288_score --dataset otb --threads 32 --num_gpus 8 --params__model MixFormer-ViT-uphead-B-multi-288-score-ep50.pth.tar --params__search_area_scale ${sear} \
##    --runid sear${sear}skip${skip} --params__update_interval ${skip}
#    python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_288_score --run_ids sear${sear}skip${skip}
#  done
#done

### MixViT-B LaSOT最佳结果69.5
for sear in 4.55
do
  for skip in 200
  do
    os=3
    python tracking/test.py mixformerpp_vit_online baseline_288_score --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-ViT-uphead-B-multi-288-score-ep50.pth.tar --params__search_area_scale ${sear} \
    --runid lasot_sear${sear}skip${skip}os${os} --params__update_interval ${skip} --params__online_sizes 3
    python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_288_score --run_ids lasot_sear${sear}skip${skip}os${os}
  done
done

### MixViT-B GOT-10k(所有数据训练的)
#for sear in 4.55 4.6 4.5 4.7 4.8
#do
#  for skip in 25 50 80
#  do
#    python tracking/test.py mixformerpp_vit_online baseline_288_score --dataset got10k_test --threads 32 --num_gpus 8 --params__model MixFormer-ViT-uphead-B-multi-288-score-ep50.pth.tar --params__search_area_scale ${sear} \
#    --runid got_ep50_sear${sear}_skip${skip} --params__update_interval ${skip} --params__online_sizes 2 --params__max_score_decay 0.98
#    python lib/test/utils/transform_got10k.py --tracker_name mixformerpp_vit_online --cfg_name baseline_288_score_got_ep50_sear${sear}_skip${skip}
##    python lib/test/utils/transform_trackingnet.py --tracker_name mixformerpp_vit_online --cfg_name baseline_288_score_skip25_os${os}_sear${sear}
#  done
#done

### MixViT-B GOT-10k(got10k训练的)
#for sear in 4.55 4.6 4.5 4.7 4.8
#do
#  for skip in 25 50 80 100
#  do
##    python tracking/test.py mixformerpp_vit_online baseline_288_got_score --dataset got10k_test --threads 32 --num_gpus 8 --params__model MixFormer-ViT-got-score-uphead-B-multi-288-ep40.pth.tar --params__search_area_scale ${sear} \
##    --runid got_ep40_sear${sear}_skip${skip} --params__update_interval ${skip} --params__online_sizes 2 --params__max_score_decay 0.98
#    python lib/test/utils/transform_got10k.py --tracker_name mixformerpp_vit_online --cfg_name baseline_288_got_score_got_ep40_sear${sear}_skip${skip}
##    python lib/test/utils/transform_trackingnet.py --tracker_name mixformerpp_vit_online --cfg_name baseline_288_score_skip25_os${os}_sear${sear}
#  done
#done

#python tracking/test.py mixformer_deit_multi baseline_deit --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-DeiT3-B-multi-384-ep0300.pth.tar --params__search_area_scale 4.5