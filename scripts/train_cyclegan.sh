set -ex
python3 train.py --dataroot ./datasets/bald2hairy --name bald2hairy_imgnum4430_imgsize128_classnum4_best --model cycle_gan --pool_size 50 \
--no_dropout --netG resnet_6blocks --load_size 143 --crop_size 128 --input_nc 4 --class_num 4 --train_split 4430 --no_html --percept_loss True \
--cycle_loss False
