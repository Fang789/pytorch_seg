#!/bin/bash/

python train.py \
	--backbone 'efficient' \
	--input_height 480 \
	--input_width 360 \
	--epochs 100 \
	--batch_size 4 \
	--datadir './txt/' \
	--weights './weights/' \
	--lr 1e-3 \
	--data_name 'camvid' \
	--gpu_id '0,1'
