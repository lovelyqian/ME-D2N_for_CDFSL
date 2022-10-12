# ME-D2N_for_CDFSL
Repository for the paper : 
ME-D2N: Multi-Expert Domain Decompositional Network for Cross-Domain Few-Shot Learning （to appear in ACM MM 2022）

[paper link :https://arxiv.org/pdf/2210.05280.pdf]()

![image.png](https://upload-images.jianshu.io/upload_images/9933353-4dbd80537b9d49a9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

If you have any questions/advices/potential ideas, welcome to contact me by fuyq20@fudan.edu.cn.


# 1 Dependencies
A anaconda envs is recommended:
```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
```

# 2 datasets
We evaluate our methods on five datasets: mini-Imagenet works as source dataset, cub, cars, places, and plantae serve as the target datasets, respectively. 
1. The datasets can be conviently downloaded and processed as in [FWT](https://github.com/hytseng0509/CrossDomainFewShot).
2. Remember to modify your own dataset dir in the 'options.py'.
3. We follow the same the same auxiliary target images as in our previous work [meta-FDMixup](https://github.com/lovelyqian/Meta-FDMixup), and the used jsons have been provided in the output dir of this repo.

# 3 pretraining
As in most of the previous CD-FSL methods, a pretrained feature extractor `baseline`   is used.
- you can directly download it from [this link](), rename it as 399.tar, and put it to the `./output/checkpoints/baseline` 
- or you can pretrain it as follows:
```
python3 train_metaTeacher.py --modelType pretrain --dataset miniImagenet --name baseline --train_aug
```

# 4 Usages
Our method is target set specific, and we take the cub target set under the 5-way 1-shot setting as an example.

1. Training St-Net
```
python3 train_metaTeacher.py --modelType St-Net --dataset miniImagenet --name St-Net-1shot --train_aug --warmup baseline --n_shot 1
```

2. Training Tt-Net
```
python3 train_metaTeacher.py --modelType Tt-Net --dataset cub --name Tt-Net-target-set-cub-1shot --train_aug --warmup baseline --n_shot 1 --stop_epoch 100
```
- note: as stated in paper, only Tt-Net under 1-shot setting is trained 100 epochs. In other cases, 400 epochs are adopted.

3. Training the ME-D2N student model
```
 python3 train_metaStudent.py --modelType Student --target_set cub --name ME-D2N-target-set-cub-1shot --train_aug --warmup baseline --n_shot 1 --ckp_S output/checkpoints/St-Net-1shot/399.tar --ckp_A output/checkpoints/Tt-Net-target-set-cub-1shot/99.tar
```

4. testing for St-Net/Tt-Net
```
python test.py --name St-Net-1shot --dataset DATASET --save_epoch 399 --n_shot 1
```
- DATASET: miniImagenet/cub/cars/places/plantae  

```
python test.py --name Tt-Net-target-set-cub-1shot --dataset DATASET --save_epoch 99 --n_shot 1
```
- DATASET: miniImagenet/cub

5. testing for ME-D2N
```
python test_twoPaths.py --name ME-D2N-target-set-cub-1shot --target_set cub --dataset DATASET --save_epoch 399 --n_shot 1
```
- DATASET: miniImagenet/cub


# 5 pretrained models
We also provide our pretrained models as follows: (coming soon



- just take them in the right dir. Take ME-D2N for the 1-shot as an example, rename it as 399.tar, and move it to the `ouput/checkpoints/ME-D2N-target-set-cub-1shot/`

# 6 citing
If you find our work or codes useful, please consider citing our work ﾍ|･∀･|ﾉ*~●
```
also coming soon...
```

