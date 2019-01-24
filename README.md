# Siamese-Recurrent-Architectures
Siamese networks are networks that have two or more identical sub-networks in them. Siamese networks seem to perform well on similarity tasks and have been used for tasks like sentence semantic similarity, recognizing forged signatures and many more.
[This](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) paper offers a pretty straightforward approach to the common problem of sentence similarity using simamese network, named MaLSTM (“Ma” for Manhattan distance)

This project tries to dig deep in to siamese networks to outperform the MALSTM performance. Following architectures have been considered.


----------------------------------------------------------------------------------------------------
### MALSTM (Original Paper)
#### Architecture
![Alt text](nn/images/MALSTM.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MALSTM.ipynb).
Best result was,

| Optimizer | Transfer Learning| Augmentation|Embedding|RMSE|Pearson|Spearman| 
| ----------|------------------| ------------|---------|----|-------|--------|
| Adagrad  | False | False |word2vec|0.153|0.809|0.741

-----------------------------------------------------------------------------------------------------
### MAGRU
#### Architecture
![Alt text](nn/images/MAGRU.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MAGRU.ipynb).
Best result was,

| Optimizer | Transfer Learning| Augmentation|Embedding|RMSE|Pearson|Spearman| 
| ----------|------------------| ------------|---------|----|-------|--------|
| Adadelta  | True | True |word2vec|0.140|0.838|0.780


-----------------------------------------------------------------------------------------------------
### MABILSTM
#### Architecture
![Alt text](nn/images/MABILSTM.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MABILSTM.ipynb).
Best result was,

| Optimizer | Transfer Learning| Augmentation|Embedding|RMSE|Pearson|Spearman| 
| ----------|------------------| ------------|---------|----|-------|--------|
| Adadelta  | True | False |word2vec|0.164|0.784|0.708

-----------------------------------------------------------------------------------------------------
### MABIGRU
#### Architecture
![Alt text](nn/images/MABIGRU.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MABIGRU.ipynb).
Best result was,

| Optimizer | Transfer Learning| Augmentation|Embedding|RMSE|Pearson|Spearman| 
| ----------|------------------| ------------|---------|----|-------|--------|
| Adadelta  | True | True |word2vec|0.143|0.832|0.773

-----------------------------------------------------------------------------------------------------
### MALSTM-ATTENTION
#### Architecture
![Alt text](nn/images/MALSTM-ATTENTION.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MALSTM-ATTENTION.ipynb).

-----------------------------------------------------------------------------------------------------
### MAGRU-ATTENTION
#### Architecture
![Alt text](nn/images/MAGRU-ATTENTION.jpg)

The experiments can be found [here](Siamese-Recurrent-Architectures%20-%20MAGRU-ATTENTION.ipynb).

