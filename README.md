# UAD
Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation

### Overview
This repository is a PyTorch implementation of the paper [Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation](https://arxiv.org/pdf/2402.06213.pdf) accepted by the 21st IEEE International Symposium on Biomedical Imaging (ISBI2024). This code is based on the [DECISION](https://github.com/driptaRC/DECISION) repository.

### Dependencies
Create a conda environment with `environment.yml`.

### Dataset
- Manually download the datasets DR ([APTOS 2019](https://kaggle.com/competitions/aptos2019-blindness-detection), [DDR](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305377), and [IDRiD](https://www.mdpi.com/2306-5729/3/3/25)) and [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) from the websites.
- Move `gen_list.py` inside data directory.
- Generate '.txt' file for each dataset using `gen_list.py` (change dataset argument in the file accordingly). 

### Training
- Train source models (shown here for Office with source A)
```
python train_source.py --dset office --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```
- Adapt to target (shown here for Office with target D)
```
python adapt_multi.py --dset office --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt
```
- Distill to single target model (shown here for Office with target D)
```
python distill.py --dset office --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/adapt --output ckps/dist
```

### Citation
If you use this code in your research please consider citing
```
@article{ahmed2021unsupervised,
  title={Unsupervised Multi-source Domain Adaptation Without Access to Source Data},
  author={Ahmed, Sk Miraj and Raychaudhuri, Dripta S and Paul, Sujoy and Oymak, Samet and Roy-Chowdhury, Amit K},
  journal={arXiv preprint arXiv:2104.01845},
  year={2021}
}
```

