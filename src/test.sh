CUDA_VISIBLE_DEVICES=0 python main.py \
    --test_only \
    --data_test Urban100 --dir_data /datasets --scale 2 --model SRResNet \
    --save test_srresnet_x2 --n_feats 64 --save_results --n_resblocks 16 --res_scale 1 \
    --pre_train pretrained/srresnet_x2_@300.pth.tar \
    --test_patch --patch_size 96 \
