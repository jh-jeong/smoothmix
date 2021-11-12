# SmoothMix: Training Confidence-calibrated Smoothed Classifiers for Certified Robustness (NeurIPS2021)

This repository contains code for the paper
**"SmoothMix: Training Confidence-calibrated Smoothed Classifiers for Certified Robustness"**
by [Jongheon Jeong](https://sites.google.com/view/jongheonj), [Sejun Park](https://sites.google.com/site/sejunparksite/), Minkyu Kim, Heung-Chang Lee, Doguk Kim and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html). 

## Dependencies
```
conda create -n smoothmix python=3.7
conda activate smoothmix

# Below is for linux, with CUDA 11.1; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

conda install scipy pandas statsmodels matplotlib seaborn
pip install tensorboardX
```

## Scripts

### Training Scripts

Our code is built upon a previous codebase from several baselines considered in the paper 
([Cohen et al (2019)](https://github.com/locuslab/smoothing); 
[Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial); 
[Jeong and Shin (2020)](https://github.com/jh-jeong/smoothing-consistency)).
The main script is `code/train.py`, and the sample scripts below demonstrate how to run `code/train.py`.
One can modify `CUDA_VISIBLE_DEVICES` to further specify GPU number(s) to work on.

```
# SmoothMix (Ours): MNIST, w/ one-step adversary, eta=5.0 
CUDA_VISIBLE_DEVICES=0 python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.0 \
--num-noise-vec 4 --eta 5.0 --num-steps 8 --alpha 1.0 --mix_step 1 --id 0
```

For a more detailed instruction to reproduce our experiments, see [`EXPERIMENTS.MD`](EXPERIMENTS.MD).

### Testing Scripts

All the testing scripts is originally from https://github.com/locuslab/smoothing:

* The script [certify.py](code/certify.py) certifies the robustness of a smoothed classifier.  For example,

```python code/certify.py mnist model_output_dir/checkpoint.pth.tar 0.50 certification_output --alpha 0.001 --N0 100 --N 100000```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.50,
and certify the MNIST test set with parameters `N0=100`, `N=100000`, and `alpha=0.001`.

* The script [predict.py](code/predict.py) makes predictions using a smoothed classifier.  For example,

```python code/predict.py mnist model_output_dir/checkpoint.pth.tar 0.50 prediction_outupt --alpha 0.001 --N 1000```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.50,
and classify the MNIST test set with parameters `N=1000` and `alpha=0.001`.

* The script [analyze.py](code/analyze.py) contains some useful classes and functions to analyze the result data 
from [certify.py](code/certify.py) or [predict.py](code/predict.py).