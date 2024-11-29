# Know Your Account: Double graph inference-based Account De-anonymization on Ethereum

This is a Python implementation of DBG4ETH, as described in the following:
> Know Your Account: Double graph inference-based Account De-anonymization on Ethereum


## Requirements
For software configuration, all model are implemented in
- Python 3.9
- Torch 2.1.1
- Pytorch-Geometric 2.4.0
- CUDA 12.1
- scikit-learn 1.2.0
- lightgbm 4.1.0


## Data
Download data from this [page](https://xblock.pro/xblock-eth.html) (Block Transaction item) and place it under the 'data/' path.


## Usage
Execute the following bash commands in the same directory where the code resides:
1. Construct and encode global static graph：
  ```bash
$ cd GSGEncoder
 $ python main_ggc.py -l p --hop 2 -ess averVolume -layer 2 --pooling max --hidden_dim 128 --batch_size 32 --lr 0.001 --dropout 0.2 -undir 1 --drop_scheme degree --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.0 --drop_edge_rate_1 0.3 --drop_edge_rate_2 0.4

  ```
More parameter settings can be found in 'utils/parameters.py'.
2. Construct and encode local dynamic graph：
  ```bash
$ cd LDGEncoder
$ python time_dy.py 
  ```
- Since the dataset size is not consistent across account types, we need to be consistent at the runtime based on the data in the datasets.
-  Labels[Nodes.index(node)] == 0: Different labels correspond to different indices. The index number corresponds to the list of target labels.
target_labels = ['Exchange', 'ICO Wallets', 'Mining', 'Phishing', 'Bridge', 'DeFi']

3. Joint prediction and calibration：
- First, the six calibration values and the ECE change before and after calibration for the two types of predicted values are obtained
- The adaptive calibration weights are calculated based on the ECE difference
- Classification is performed using a classifier
```bash
$ cd modeladd
$ python classification.py
```

## Citation
If you find this work useful, please cite the following:
  ```bash
@article{
  title={Know Your Account: Double Graph Inference-based Account De-anonymization on Ethereum},
  author={Miao, Shuyi and Qiu, Wangjie and Zheng, Hongwei and Zhang, Qinnan and Tu, Xiaofan and Liu, Xunan and Liu, Yang and Dong, Jin and Zheng，zhiming},
  booktitle={2025 IEEE 41rd International Conference on Data Engineering (ICDE)},
  year={2025},
  publisher={IEEE}
}
  ```
