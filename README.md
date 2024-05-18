Please make sure you have manually adjusted the best learning rate.

After one training session, it is extremely recommended to reload the weights and lower the lr for another training session. This can be viewed as manul lr_scheduler.

# Improved Patch-Free 3D Medical Image Segmentation
Code for our extended journal paper based on *Super-Resolution Based Patch-Free 3D Image Segmentation with High-Frequency Guidance*.

---

This is an extension to the [PFSeg](https://github.com/Dootmaan/PFSeg) repo. We investigated the flaws of the original method and proposed three new improvements to solve them. The problems includes:

- The scale-inconsistency introduced in the concatation between LR main input and HR guidance patch.
- The limitation of central cropping*
- The underated potential of TFM

We introduced new multi-scale network structure and designed a clever algorithm for high-speed guidance patch searching. TFM has also been used for achieving a better model generalization ability.

Details will be released upon paper publication.

## 1. Prepare your dataset
Please follow the instructions on http://braintumorsegmentation.org/ to get your copy of the BRATS2020 dataset. 

## 2. Install dependencies
Our code should work with Python>=3.5 and PyTorch >=0.4.1.

Please make sure you also have the following libraries installed on your machine:
- PyTorch
- NumPy
- MedPy
- tqdm

Optional libraries (they are only needed when run with the -v flag):
- opencv-python

## 3. Run the code
Firstly, clone our code by running

```
    git clone git@github.com:Dootmaan/PFSeg.git
```
Normally you only have to specify the path to BRATS2020 dataset to run the code.

For example you can use the following command (of course you need to change directory to ./PFSeg first):

```
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u train_PFSeg.py -dataset_path "/home/somebody/BRATS2020/" > train_PFSeg.log 2>&1 &
```

You can also add the -v flag to have verbose output. Our code supports multi-gpu environment and all you have to do is specifying the indices of the available GPUs. Directly running train_PFSeg.py will use all the GPUs on your machine and use the default parameter settings. **The minimun requirements for running our code is a single GPU with 11G video memory.**

Click [here](https://drive.google.com/file/d/1kG2kYU_56-0UV2E2I59c1qYphoYRdziK/view?usp=sharing) to download the pretrained weights for our framework.

Please note that the code uses 6/2/2 train/val/test split by default, while our results in the paper are reported with a 8/2 train/test split so if you would like to verify the results please make sure you change the default dataset split. **We also did a 5-fold cross validation for our method and the results are quite stable, as you can see in the rebuttal part of our paper**. 

More codes and information may be updated later.
<!-- 
---

Below are some experimental results that may be helpful (BRATS2020).

Results with 8/2 train/test split

|  Method   | DSC  |  HD95(mm)  |
|  ----  | ----  | ---- |
| VNet | 0.7991(0.1345) | 13.86(18.6006) |
| UNet3D | 0.8121(0.1277) | 14.63(21.1204) |
| ResUNet3D  | 0.8218(0.1182) | 13.21(19.4195) |
| ResUNet3D⬆ | 0.8089(0.1525) | 8.56(8.0853) |
| Holistic Decomposition+ResUNet3D | 0.8245(0.1474) | 9.21(12.0561) |
| Ours | 0.8382(0.1433) | 7.83(8.6250) |

Results with 6/2/2 train/val/test split

|  Method   | DSC  |  HD95(mm)
|  ----  | ----  | ---- |
| VNet | 0.7776(0.1693) | - |
| UNet3D | 0.7954(0.1368) | - |
| ResUNet3D  | 0.8097(0.1291) | - |
| ResUNet3D⬆ | 0.8016(0.1595) | - |
| Holistic Decomposition+ResUNet3D | 0.8161(0.1645) | - |
| Ours | 0.8329(0.1450) | 8.5131 |

6/2/2 Ablation Study:
|  Method   | UNet  |  ResUNet
|  ----  | ----  | ---- |
| US | 0.7945 | 0.8082 |
| US+SR | 0.8010 | 0.8141 |
| US+SR+TEL  | 0.8106 | 0.8193 |
| US+SR+TEL+SSL | 0.8163 | 0.8281 |
| US+SR+TEL+SSL+SGM | 0.8208 | 0.8329 | -->
