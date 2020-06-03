TRAINSETS_DIR='GLUNet_data/training_datasets'
TRANSFORM='Raw'
CROP_SIZE=520
BATCH_SIZE=40

TRAIN_SET1='sintel_allpair_clean'
## Add more train dataset here ##

python train_GLUNet.py \
  --training_data_dir ${TRAINSETS_DIR}
  --batch-size ${BATCH_SIZE}
  --transform_type ${TRANSFORM}
  --dataset_list ${TRAIN_SET1} ## Add more train datasets with number here
  --input_size ${CROP_SIZE} 
