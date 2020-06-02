MODEL_DIR=$1
SAVE_DIR=$2

GPU_ID=1

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./eval.py \
  --pre_trained_models Sintel \
  --pre_trained_models_dir ./snapshots/${MODEL_DIR} \
  --datasets TSS \
  --save_dir ./results/${SAVE_DIR}_TSS_JODS \
  --data_dir ./GLUNet_data/testing_datasets/TSS_CVPR2016/JODS


CUDA_VISIBLE_DEVICES=${GPU_ID} python ./eval.py \
  --pre_trained_models Sintel \
  --pre_trained_models_dir ./snapshots/${MODEL_DIR} \
  --datasets TSS \
  --save_dir ./results/${SAVE_DIR}_TSS_FG3DCar \
  --data_dir ./GLUNet_data/testing_datasets/TSS_CVPR2016/FG3DCar

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./eval.py \
  --pre_trained_models Sintel \
  --pre_trained_models_dir ./snapshots/${MODEL_DIR} \
  --datasets TSS \
  --save_dir ./results/${SAVE_DIR}_TSS_PASCAL \
  --data_dir ./GLUNet_data/testing_datasets/TSS_CVPR2016/PASCAL
