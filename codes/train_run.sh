#python train_runet.py  --name runet --inet runet --model runet_model --hyper
#python train_errnet.py --name errnet --hyper --resume --resume_epoch 10
#CUDA_VISIBLE_DEVICES=7 python train_errnet.py --name errnet
CUDA_VISIBLE_DEVICES=3 python train_errnet.py --name swintranet --inet swintranet  --lambda_gan 0 --hyper
#CUDA_VISIBLE_DEVICES=3 python train_errnet.py --name errnet