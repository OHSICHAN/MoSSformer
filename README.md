# MoSSformer : Motif-Enhanced Spectral Structures for Graph-Based Recommendation

<!-- This is the PyTorch implementation for our SIGIR 2024 paper.  -->
> Sichan OH, Jaekwang KIM, MoSSformer : Motif-Enhanced Spectral Structures for Graph-Based Recommendation
 <!-- [arXiv link](https://arxiv.org/abs/2404.11982) -->

## Environment
- python==3.9.19
- numpy==1.26.4
- pandas==2.2.1
- scipy==1.12.0
- torch==2.2.2
- pyg==2.5.2
- torchsparsegradutils==0.1.2

## Datasets

| Dataset| #Users | #Items | #Interactions | Pos/Neg |
|---|---|---|---|---|
| Amazon-CDs | 51,267 | 46,464 | 895,266 | 1:0.22 |
| Amazon-Music | 3,472 | 2,498 | 49,875 | 1:0.25 | 
| Epinions | 17,894 | 17,660 | 413,774 | 1:0.37 | 
| KuaiRec | 1,411 | 3,327 | 253,983 | 1:5.95 |
| KuaiRand | 16,974 | 4,373 | 263,100 | 1:1.25 |

## Training & Evaluation
* Amazon-CDs
  ```bash
  python -u code/main.py --data=amazon-cds --offset=4 --alpha=0.8 --beta=1 --sample_hop=2 --num_motifs=3 --n_neighbors=20
  ```
* Amazon-Music
  ```bash
  python -u code/main.py --data=amazon-music --offset=4 --alpha=0.8 --beta=1 --sample_hop=3 --num_motifs=3 --n_neighbors=30
  ```
* Epinions
  ```bash
  python -u code/main.py --data=epinions --offset=4 --alpha=0.0 --beta=1 --sample_hop=2 --num_motifs=3 --n_neighbors=20
  ```
* KuaiRec
  ``` bash
  python -u code/main.py --data=KuaiRec --offset=1 --alpha=-0.0 --beta=-0.2 --sample_hop=1 --num_motifs=3 --n_neighbors=15
  ```
* KuaiRand
  ```bash
  python -u code/main.py --data=KuaiRand --offset=1 --alpha=1.0 --beta=1 --sample_hop=2 --num_motifs=3 --n_neighbors=35
  ```

<!-- ## Citation -->
<!-- If you find the paper useful in your research, please consider citing:
```
@inproceedings{chen2024sigformer,
  title={SIGformer: Sign-aware Graph Transformer for Recommendation},
  author={Chen, Sirui and Chen, Jiawei and Zhou, Sheng and Wang, Bohao and Han, Shen and Su, Chanfei and Yuan, Yuqing and Wang, Can},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1274--1284},
  year={2024}
} -->
```
