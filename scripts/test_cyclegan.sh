set -ex
python3 test.py --dataroot ./datasets/bald2hairy --name bald2hairy_imgnum4430_imgsize128_classnum4_best --model cycle_gan \
--no_dropout --netG resnet_6blocks --load_size 143 --crop_size 128 --input_nc 4 --class_num 4 --percept_loss True --cycle_loss False \
--one_hot_vector 1 0 1 0
