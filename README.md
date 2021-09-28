# GANiry: Bald-to-Hairy Translation Using CycleGAN

Official PyTorch implementation of GANiry.

> [GANiry: Bald-to-Hairy Translation Using CycleGAN](https://arxiv.org/abs/2109.13126),            
> [Fidan Samet](https://fidansamet.github.io/), Oguz Bakir.        
> *([arXiv pre-print](https://arxiv.org/abs/2109.13126))*            


## Summary
This work presents our computer vision course project called bald
men-to-hairy men translation using CycleGAN. On top of CycleGAN architecture,
we utilize perceptual loss in order to achieve more realistic results. We also
integrate conditional constrains to obtain different stylized and colored hairs
on bald men. We conducted extensive experiments and present qualitative results
in this work.


## Getting Started

### Setup

1. Create new conda environment
    ~~~
    conda create --name ganiry
    ~~~
    
2. Activate the environment
    ~~~
    conda activate ganiry 
    ~~~

3. Install the requirements
    ~~~
    pip install -r requirements.txt
    ~~~

4. Download [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and prepare sub-dataset
    ~~~
    python build_copy.py --dataroot ./datasets/bald2hairy --celeba_path ./datasets/celeba/data
    ~~~

### Training
Number of classes indicates the different hair classes in the dataset.

    python train.py --dataroot ./datasets/bald2hairy --name bald2hairy --no_dropout --netG resnet_6blocks --load_size 143 --crop_size 128 --input_nc 4 --class_num 4 --percept_loss True --cycle_loss False

### Test
One hot vector is the binary encoding of hair classes.

    python test.py --dataroot ./datasets/bald2hairy --name bald2hairy --no_dropout --netG resnet_6blocks --load_size 143 --crop_size 128 --input_nc 4 --class_num 4 --percept_loss True --cycle_loss False --phase test --one_hot_vector 1 0 1 0


## License

GANiry is released under [GNU General Public License](LICENSE). We developed 
GANiry on top of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
Please refer to License of CycleGAN for more details.


## Citation

If you find GANiry useful for your research, please cite our paper as follows.

> F. Samet, O. Bakir, "GANiry: Bald-to-Hairy Translation Using CycleGAN", arXiv, 2021.


BibTeX entry:
```
@misc{samet2021ganiry,
      title={GANiry: Bald-to-Hairy Translation Using CycleGAN}, 
      author={Fidan Samet and Oguz Bakir},
      year={2021},
      eprint={2109.13126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
