# UAD
**Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation**

### Overview
This repository is a PyTorch implementation of the paper [Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation](https://arxiv.org/pdf/2402.06213.pdf) accepted by the 21st IEEE International Symposium on Biomedical Imaging (ISBI2024). This code is based on the [DECISION](https://github.com/driptaRC/DECISION) repository.

### Prerequisites
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse

### Dataset
- Manually download the datasets DR ([APTOS 2019](https://kaggle.com/competitions/aptos2019-blindness-detection), [DDR](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305377), and [IDRiD](https://www.mdpi.com/2306-5729/3/3/25)) and [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) from the websites and move them into `dataset` directory.
- Run `genlist.py` file located in `dataset` directory, by changing dataset argument in the file, to generate '.txt' file for each dataset.

### Training
- Train source models for every domain in each dataset:
```
bash run_source.sh
```
- Adapt to target for every domain in each dataset:
```
bash run_target.sh
```

### Citation
If you find this code useful for your research, please cite our paper:
```
@misc{song2024multisourcefree,
      title={Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation}, 
      author={Yaxuan Song and Jianan Fan and Dongnan Liu and Weidong Cai},
      year={2024},
      eprint={2402.06213},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

