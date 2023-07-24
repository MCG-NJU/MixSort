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

#python tracking/test.py mixformerpp_vit baseline --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-ViT-B-multi-288-ep0300.pth.tar  --params__search_area_scale 4.5 --params__online_sizes 1
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

#for i in 4.45 4.5 4.6 4.7
#do
#  for os in 2
#  do
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset trackingnet --threads 32 --num_gpus 8 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale $i --params__online_sizes 2 --runid os2sear${i}
#  done
#done

for i in 495 500
do
  python tracking/test.py mixformerpp_vit baseline_288_ablation_10layers --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-ViT-B-ablation_mae_finetune-288-5.0-ep500.pth.tar  --params__search_area_scale 4.55 --runid ep${i}
  python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_288_ablation_10layers --run_ids ep${i}
done

#for i in 500
#do
##  python tracking/test.py mixformerpp_vit baseline_288_ablation_10layers --dataset lasot --threads 32 --num_gpus 8 --params__model MixFormer-ViT-B-mae-6layers.pth.tar  --params__search_area_scale 4.55 --runid ep${i}
#  python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_288_ablation_10layers --run_ids ep${i}
#done

#for i in 4.45 4.5 4.55 4.6 4.7
#do
#  for os in 2
#  do
#    python tracking/test.py mixformerpp_vit_online baseline_large_score --dataset otb --threads 32 --num_gpus 8 --params__model MixFormer-score-ViT-uphead-L-multi-384-ep37.pth.tar --params__search_area_scale $i --params__online_sizes ${os} --runid os${os}sear${i}
#  done
#done

#for i in 4.45 4.5 4.55 4.6 4.7
#do
#  for os in 2
#  do
#    python lib/test/utils/transform_trackingnet.py --tracker_name mixformerpp_vit_online --cfg_name baseline_large_score_os2sear${i}
#  done
#done

#for i in 31 32
#do
#  python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_384_score --run_ids ep${i}
#done
#for i in 4.45 4.5 4.6 4.7
#do
#  for os in 2 3 4
#  do
#    python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_online --cfg_name baseline
#  done
#done