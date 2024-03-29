# Experiments

This document specifies how to replicate our results on MNIST and CIFAR-10. 
Results on ImageNet can be reproduced in a similar way.

### Training

Please specify GPU number(s) before run each line by modifying `CUDA_VISIBLE_DEVICES=[num]`.

```
# Baseline: Gaussian (Cohen et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 --id 0

# Baseline: SmoothAdv (Salman et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0

# Baseline: Stability training (Li et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--lbd 2 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.50 \
--lbd 2 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.00 \
--lbd 2 --id 0

# Baseline: MACER (Zhai et al., 2020)
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 16.0 --deferred --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 16.0 --deferred --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 6.0 --deferred --id 0

# Baseline: Consistency (Jeong and Shin, 2020)
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--num-noise-vec 2 --lbd 10 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--num-noise-vec 2 --lbd 5 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 2 --lbd 5 --id 0

# Ours: SmoothMix (MNIST)
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--num-noise-vec 4 --num-steps 3 --alpha 1.0 --mix_step 0 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--num-noise-vec 4 --num-steps 2 --alpha 1.0 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.5 \
--num-noise-vec 4 --num-steps 4 --alpha 1.0 --mix_step 0 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.5 \
--num-noise-vec 4 --num-steps 4 --alpha 1.0 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.0 \
--num-noise-vec 4 --num-steps 8 --alpha 1.0 --mix_step 0 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.0 \
--num-noise-vec 4 --num-steps 8 --alpha 1.0 --mix_step 0 --eta 5.0 --id 0

# Ours: SmoothMix + One-step adversary (MNIST)
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--num-noise-vec 4 --num-steps 3 --alpha 1.0 --mix_step 1 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--num-noise-vec 4 --num-steps 2 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.5 \
--num-noise-vec 4 --num-steps 4 --alpha 1.0 --mix_step 1 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.5 \
--num-noise-vec 4 --num-steps 4 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.0 \
--num-noise-vec 4 --num-steps 8 --alpha 1.0 --mix_step 1 --eta 1.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.0 \
--num-noise-vec 4 --num-steps 8 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0

# Ours: SmoothMix (CIFAR-10)
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--num-noise-vec 2 --num-steps 4 --alpha 2.0 --mix_step 0 --eta 5.0 --id 0

# Ours: SmoothMix + One-step adversary (CIFAR-10)
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--num-noise-vec 2 --num-steps 4 --alpha 2.0 --mix_step 1 --eta 5.0 --id 0
```

### Certification

```
# Baseline: Gaussian (Cohen et al., 2019)
python code/certify.py mnist logs/mnist/cohen/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/cohen/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/cohen/noise_0.5/lenet/0/checkpoint.pth.tar 0.50 \
test/certify/mnist/cohen/0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/cohen/noise_1.0/lenet/0/checkpoint.pth.tar 1.00 \
test/certify/mnist/cohen/0/noise_1.00 --N=100000 --skip=1

# Baseline: SmoothAdv (Salman et al., 2019)
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/0/noise_1.00 --N=100000 --skip=1

# Baseline: Stability training (Li et al., 2019)
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/mnist/stab/lbd_2/0/noise_0.25 --N 100000 --skip 1
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/mnist/stab/lbd_2/0/noise_0.5 --N 100000 --skip 1
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/mnist/stab/lbd_2/0/noise_1.0 --N 100000 --skip 1

# Baseline: MACER (Zhai et al., 2020)
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_6.0/gamma_8.0/beta_16.0/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/macer_deferred30/num_16/lbd_6.0/gamma_8.0/beta_16.0/0/noise_1.00 --N=100000 --skip=1

# Baseline: Gaussian + Consistency (Jeong and Shin, 2020)
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/consistency/cohen/num_2/lbd_10.0/eta_0.5/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_5.0/eta_0.5/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/consistency/cohen/num_2/lbd_5.0/eta_0.5/0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_5.0/eta_0.5/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/consistency/cohen/num_2/lbd_5.0/eta_0.5/0/noise_1.00 --N=100000 --skip=1

# Ours: SmoothMix (MNIST)
python code/certify.py mnist logs/mnist/smix_1.0_3_m0/eta_1.0/num_4/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/smix_1.0_3_m0/eta_1.0/num_4/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_2_m0/eta_5.0/num_4/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/smix_1.0_2_m0/eta_5.0/num_4/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_4_m0/eta_1.0/num_4/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/smix_1.0_4_m0/eta_1.0/num_4/0/noise_0.5 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_4_m0/eta_5.0/num_4/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/smix_1.0_4_m0/eta_5.0/num_4/0/noise_0.5 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_8_m0/eta_1.0/num_4/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/smix_1.0_8_m0/eta_1.0/num_4/0/noise_1.0 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_8_m0/eta_5.0/num_4/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/smix_1.0_8_m0/eta_5.0/num_4/0/noise_1.0 --N=100000 --skip=1

# Ours: SmoothMix + One-step adversary (MNIST)
python code/certify.py mnist logs/mnist/smix_1.0_3_m1/eta_1.0/num_4/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/smix_1.0_3_m1/eta_1.0/num_4/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_2_m1/eta_5.0/num_4/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 \
test/certify/mnist/smix_1.0_2_m1/eta_5.0/num_4/0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_4_m1/eta_1.0/num_4/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/smix_1.0_4_m1/eta_1.0/num_4/0/noise_0.5 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_4_m1/eta_5.0/num_4/noise_0.5/lenet/0/checkpoint.pth.tar 0.5 \
test/certify/mnist/smix_1.0_4_m1/eta_5.0/num_4/0/noise_0.5 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_8_m1/eta_1.0/num_4/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/smix_1.0_8_m1/eta_1.0/num_4/0/noise_1.0 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/smix_1.0_8_m1/eta_5.0/num_4/noise_1.0/lenet/0/checkpoint.pth.tar 1.0 \
test/certify/mnist/smix_1.0_8_m1/eta_5.0/num_4/0/noise_1.0 --N=100000 --skip=1

# Ours: SmoothMix (CIFAR-10)
python code/certify.py cifar10 logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar10/smix_0.5_4_m0/eta_5.0/num_2/0/noise_0.25 --N=100000 --skip=1
python code/certify.py cifar10 logs/cifar10/smix_1.0_4_m0/eta_5.0/num_2/noise_0.5/cifar_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar10/smix_1.0_4_m0/eta_5.0/num_2/0/noise_0.5 --N=100000 --skip=1
python code/certify.py cifar10 logs/cifar10/smix_2.0_4_m0/eta_5.0/num_2/noise_1.0/cifar_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar10/smix_2.0_4_m0/eta_5.0/num_2/0/noise_1.0 --N=100000 --skip=1

# Ours: SmoothMix + One-step adversary (CIFAR-10)
python code/certify.py cifar10 logs/cifar10/smix_0.5_4_m1/eta_5.0/num_2/noise_0.25/cifar_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar10/smix_0.5_4_m1/eta_5.0/num_2/0/noise_0.25 --N=100000 --skip=1
python code/certify.py cifar10 logs/cifar10/smix_1.0_4_m1/eta_5.0/num_2/noise_0.5/cifar_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar10/smix_1.0_4_m1/eta_5.0/num_2/0/noise_0.5 --N=100000 --skip=1
python code/certify.py cifar10 logs/cifar10/smix_2.0_4_m1/eta_5.0/num_2/noise_1.0/cifar_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar10/smix_2.0_4_m1/eta_5.0/num_2/0/noise_1.0 --N=100000 --skip=1
```