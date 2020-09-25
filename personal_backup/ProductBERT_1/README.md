<h1 align="center">
Pro-BERT: Product Assortment Contextual Embeddings for demand estimation
</h1>

# Requirements

Repository is implemented with following primary requirements.

- Python 3.6.10 
- Pytorch 1.2.0

All requirements & dependencies are configured into two text files namely:-

- env.txt
- requirements.txt

You can exactly replicate the required environment by following the procedure(Environment set-up) mentioned subsequently.

Note:This process invloves configuration using conda virtual environement.It is recommended to use the same in order to comply with the follwing instructions.

## Miniconda Installation

- Download miniconda ,by the follwoing command:-

```bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```
- Install using this command:-
```bash

bash Miniconda3-latest-Linux-x86_64.sh

```
- To get the updated version of conda
```bash

conda update conda

```
Conda installation is complete! Now try following steps.
# Environment set-up


- Create conda environment with the primary configuration of environment

```bash

conda create --name productbertenv --file env.txt

```
- Activate the conda environment

```bash

conda activate productbertenv

```
- Install other required packages & dependencies

```bash

pip install -r requirements.txt

```

# Files Description


- ***PRO-Bert.ipynb'***: Main notebook of the implementation
- ***PRO-Bert.py***: Python file version of main notebook
- ***MNIST-DeepNeuralChoice.ipynb***:Experimentation notebook with MNIST[classification]
- ***MNIST-DeepNeuralChoice-Set2Set.ipynb***.ipynb:Experimentation notebook with MNIST[SET to SET]
- ***modelling_bert.py***: Modelling of the proBERT .
- ***configuration_bert.py***: Model Configuration
- ***configuration_utils.py***:Utils & hepler functions required for configuration
- ***modelling_utils.py***:All Utility functions required for modelling 

# Data



Data is present in the directory ***'DATA'*** in the form of two csv files namely :-

- ***transaction_final.csv*** :- Transactional sales data of 582 stores ,92353 products for 102 weeks( nearly 2 years)

| Attributes of transaction_final.csv  | Description  | Cardinality|
|-|-|-|
|STORE_ID | Id to uniquely identify Store|582|
|PRODUCT_ID|Id to uniquely identify Product|92353|
|WEEK_NO|Week number of the transaction|102|
|WEEK_QUANTITY|Number of units sold of the product in that store & week|*|
|INDEX| Product token index for the model input |92353|
    
- ***products_final.csv*** :- Information of the products involved in the transactions .

| Attributes of products_final.csv  | Description  |Cardinality|
|-|-|-|
|STORE_ID | Id to uniquely identify Store|582|
|PRODUCT_ID|Id to uniquely identify Product|92353|
|MANUFACTURER|Id to specify the manufacturer|6476|
|DEPARTMENT|Text description of the product's department|44|
|BRAND| Specification of **National** or **Private** brand |2|
|COMMODITY_DESC|Primary level description of the commodity|308|
|SUB_COMMODITY_DESC|More granular description of the commodity|2383|
|INDEX| Product token index for the model input |92353|

# Notebooks

In total, there are three python notebooks ,each for distinctive purpose as specified below:-

> PRO-Bert.ipynb :- Actual implementation of the proposed architecture.Uses data present in `DATA` folder.[Main Notebook].
(Same code is present in the form of python file - PRO-BERT.py)

Other two notebooks are to perform experiments adapted from [Deep Neural Choice Model](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwihtbuH0MnqAhVHyDgGHV8JCnMQFjABegQIBRAB&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI16%2Fpaper%2Fdownload%2F12098%2F11674&usg=AOvVaw23ab3SojoyCkPb89NmepM1)

> MNIST-DeepNeuralChoice-Set2Set.ipynb :- Notebook to perform experimentation as mentioned, predicts quantity for each image present in choice set subjected to the probability distribution mentioned in Set to Set manner.Uses **MNIST** dataset.

> MNIST-DeepNeuralChoice.ipynb :- Notebook for the same experiment, but acts as choice model,where it tries to choose one option from the set as per the distribution specified.Uses **MNIST** dataset.
    
`In order to run these notebooks, just run cells in the sequential order specified.`

`Plots_assortment.ipynb :- An additional notebook for plotting loss,time with assortment sizes.`



# Implementation


Model(ProductBERT) implementation,configuration,utils are all present in the directory called `pro-bert`.
The corresponding files related to this sector are:-
- ***modelling_bert.py***
- ***configuration_bert.py***
- ***configuration_utils.py***
- ***modelling_utils.py***
- ***file_utils.py***

Brief details of which are already specified in the section **Files Description**.


# Configuration

To obtain the results that we have achieved,following are the configurational details.Concerned file ***configuration_bert.py***

- For the core model results[PRO-Bert.ipynb or PRO-Bert.py]:
```python
def __init__(
        self,
        vocab_size=92354,
        hidden_size=64, 
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=10,  
        layer_norm_eps=1e-6, 
        pad_token_id=0,
        **kwargs
    ):
```
- For MNIST experimentation[MNIST-DeepNeuralChoice.ipynb & MNIST-DeepNeuralChoice-Set2Set.ipynb]
```python
def __init__(
        self,
        vocab_size=92354,
        hidden_size=784, 
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,  
        layer_norm_eps=1e-8, 
        pad_token_id=0,
        **kwargs
    ):
```
# Learning Rate & Optimizer

- We use ADAM optimization function & learning rates specific for the task as mentioned.
- Core model[PRO-Bert.ipynb or PRO-Bert.py]:
```python

import torch.optim
optimizer=torch.optim.Adam(model.parameters(),lr=0.05, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)

```
- MNIST experimentation[MNIST-DeepNeuralChoice.ipynb & MNIST-DeepNeuralChoice-Set2Set.ipynb]:
```python

import torch.optim
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

```
# Deciding assortment size


Size of the assortment can be varied and can be changed accordingly ***( in PRO-Bert.ipynb)***.

```python
from keras.preprocessing.sequence import pad_sequences

assortment_sizes=[32,40,64,80,128,160,256,300,320,400]
MAX_LEN = assortment_sizes[0]

print('\nPadding/truncating all assortments to %d values...' % MAX_LEN)
product_tokens= pad_sequences(assortment, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
demand_labels= pad_sequences(labels, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print('\nDone.')

```

# Architecture


A glimpse of our architecture

```python

ProductBert(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(92354, 784, padding_idx=0)
      (position_embeddings): Embedding(512, 784)
      (token_type_embeddings): Embedding(2, 784)
      (LayerNorm): LayerNorm((784,), eps=1e-08, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=784, out_features=784, bias=True)
              (key): Linear(in_features=784, out_features=784, bias=True)
              (value): Linear(in_features=784, out_features=784, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=784, out_features=784, bias=True)
              (LayerNorm): LayerNorm((784,), eps=1e-08, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=784, out_features=64, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=64, out_features=784, bias=True)
            (LayerNorm): LayerNorm((784,), eps=1e-08, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=784, out_features=784, bias=True)
              (key): Linear(in_features=784, out_features=784, bias=True)
              (value): Linear(in_features=784, out_features=784, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=784, out_features=784, bias=True)
              (LayerNorm): LayerNorm((784,), eps=1e-08, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=784, out_features=64, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=64, out_features=784, bias=True)
            (LayerNorm): LayerNorm((784,), eps=1e-08, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=784, out_features=784, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=784, out_features=1, bias=True)
)

```
# Results 



| Assortment Size | Training Loss |Validation Loss|Training Time(hh:mm:ss) for an epoch |Validation Time(hh:mm:ss) for an epoch|
|-|-|-|-|-|
| 32 |1.57 |1.58 |'0:00:35'|'0:00:02'|
| 40 | 1.54 |1.56 |'0:00:38'|'0:00:02'|
| 64 |1.50 |1.52 |'0:00:47'|'0:00:02'|
| 80 | 1.47 |1.50 |'0:00:55'|'0:00:03'|
| 128 | 1.40 |1.45 |'0:01:17'|'0:00:04'|
| 160 |1.36|1.41 |'0:01:38'|'0:00:05'|
| 256 |1.33|1.38 |'0:02:41'|'0:00:07'|
| 300 |1.322 |1.376 |'0:03:24'|'0:00:09'|
| 320 |1.320|1.370 |'0:03:36'|'0:00:10'|
| 400 |1.31 |1.369 |'0:05:04'|'0:00:14'|

