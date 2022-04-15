
# Disheng's work 
---------
# Pytorch Basics 

[Key note of neural nerwork  in Pytorch](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/weekly%20report/week1/pytorch%20in%20TL.key)



* Dataflow in neural network
  input, forward propagation, (softmax), loss, backward propagation, optimization, parameters update 



* Components of defining a neural network in Pytorch
  network structure, input  and output(shape), activation function, loss function, optimizor



* Basic data format ——**tensor** 
  Tensors are similar to [NumPy’s](https://numpy.org/) ndarrays, except that tensors can run on GPUs or other hardware accelerators.

  it is easier to do complex operations in tensor than Numpy.



* AUTOMATIC DIFFERENTIATION—— **TORCH.AUTOGRAD**
  pytorch can automatically calculate **DIFFERENTIATION** for all of parameters in the network, and use them in back propagation.



There are reasons you might want to disable gradient tracking:

- * To mark some parameters in your neural network as **frozen parameters**. 
  - This is a very common scenario for [finetuning a pretrained network](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)---HTL
  - To **speed up computations** when you are only doing forward pass,
  -  because computations on tensors that do not track gradients would be more efficient





( **Attention**: tensor.grad.zero_()

  In [PyTorch](https://github.com/pytorch/pytorch), for every mini-batch during the *training* phase, we typically want to explicitly set the gradients to zero before starting to do back propragation.

  Because PyTorch *accumulates the gradients* on subsequent backward passes. 

  This accumulating behaviour is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple *mini-batches* )





# Metamap Lite
  [Instruction of installing and running MateMap Lite](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/Metamap%20Lite_instruction%20.docx)



# Embedding (supervised & unsupervised)

## Word2vec (negative sampling) ---**2013**
  supervised method:   
   *  skip-grams      
   * CBOW (Continuous Bag of Words)      

* ### theory

* ### code
   [**implement**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/Disheng_code/word2vec.ipynb)




## Bert

* ### theory

* ### code
   [**implement**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/Disheng_code/BERT%20Word%20Embeddings.ipynb)





## GloVe---**2014**
unsupervised method (leverlage co-occurrence information)


In NLP field, it is common understanding that the meaning of specific word is defined by its context.

so **why not use co-occurrence count (word-word matrix)  to represent the meaning of word**

<img src="images/co_occurrence.png" width="700" >



**problem of using co-occurrence matrix directly**:
1. sparse
2. very high dimensions


* ### theory:  
  GloVe (Global Vectors) 
  > 'GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.'

  **GloVe model** leverages co-occurrence information, generating dense  word vector.

  **'the global corpus statistics are captured directly by the model'**



<img src="images/mathematical_definition.png" width="700" >


  **comparing with word2vec**

<img src="images/comparison_performance.png" width="700" >


* ### useful links

  [**Stanford Official website**](https://nlp.stanford.edu/projects/glove/)   
  [**Original Paper**](https://aclanthology.org/D14-1162.pdf)



* ### code

  [**implement**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/Disheng_code/GloVe.ipynb)


## Topic model
unsupervised method
1. LSA(Latent semantic analysis)
2. SVD(Singular value decomposition) 



# Data Exploration (disease severity)



### Jupyter Notebok

  1. [**data_ exploration.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/data_%20exploration.ipynb):   
  overall data exploration in all of tables



  3. [**R3_1971_YE_DEMOGRAPHICS_2021_03_01__Exploration.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/R3_1971_YE_DEMOGRAPHICS_2021_03_01__%20Exploration.ipynb):   
  Data exploration in DEMOGRAPHICS

  4. [**R3_1971_YE_DIAGNOSES_2021_03_01__Diagnoses.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/R3_1971_YE_DIAGNOSES_2021_03_01__Diagnoses.ipynb):     
  Data exploration in DIAGNOSES

  5. [**R3_1971_YE_ENCOUNTER_2021_03_01__ Exploration.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/R3_1971_YE_ENCOUNTER_2021_03_01__%20Exploration.ipynb):    
  Data exploration in ENCOUNTER

  6. [**severity exploration.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/severity%20exploration%20.ipynb):   
  Use Diagnoses table to get all of study_id of Covid patient.
  Use those Covid patient study id to extract data from Encounter table and do some exploration

  2. [**lab_result---covid.ipynb**](https://github.com/DishengLL/disheng-s_work/blob/main/liu_disheng_working/csv%20analysis/lab_result---%20covid%20.ipynb):   
  Use Lab result to extract covid patients, and use those patients' id to extract all of encounter records in Encounter table.







