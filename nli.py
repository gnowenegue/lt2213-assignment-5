# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: LT2213 Assignment 5
#     language: python
#     name: lt2213-assignment-5
# ---

# %% [markdown]
# # Natural Language Inference
#
# **Adam Ek, Bill Noble, Simon Dobnik, and others**

# %% [markdown]
# The lab is an exploration and learning exercise to be done in a group and also in discussion with the teachers and other students.
#
# Before starting, please read the instructions on how to work in groups on Canvas.
#
# Write all your answers and the code in the appropriate boxes below.
#
# In this lab we will work with neural networks for natural language inference. Our task is: given a premise sentence P and hypothesis H, what entailment relationship holds between them? Is H entailed by P, contradicted by P or neutral towards P?
#
# Given a sentence P, if H definitely describe something true given P then it is an **entailment**. If H describe something that's *maybe* true given P, it's **neutral**, and if H describe something that's definitely *false* given P it's a **contradiction**. 
#
# **Dependencies**
#
# * Pytorch
#     * Installation instructions: https://pytorch.org/
#     * Tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
#     * Some useful basic operations: https://jhui.github.io/2018/02/09/PyTorch-Basic-operations
# * ...
#
# **Running the code**
#
# As we are learning about the models, and also what methods work and do not work for our semantic tasks, we are not interested in achieving a state-of-the-art performance. We are learning about different implementations and differences in performance in different conditions.
#
# **On using generative AI for this assignment:** For this lab, the use of generative AI is permitted as a supporting tool, provided it is done in a responsible and conscious manner and that you state clearly with each question how it was used. However, generative AI must never replace the intellectual work you are expected to carry out. Note that the purpose of this lab is to learn some basic coding of the main neural architectures used in natural language processing. You are responsible for ensuring that such tools are used in a way that supports the development of the skills the module is designed to promote. It is your responsibility to ensure that submitted work is the result of independent intellectual effort.
#
# **Getting help:** We encourage you to use Canvas discussions to post questions and interact with teachers and also each other. Provide useful tips, but of course do not reveal the exact answer across the groups as each group should should work out their own solutions. Remember that in most cases there is also not a single correct answer and implementations may differ.

# %% [markdown]
# ## 1. Data

# %% [markdown]
# We will explore natural language inference using neural networks on the SNLI dataset, described in [1]. 
#
# There are two options for loading and working with the data.
#
# 1. Download the data directly from the [SNLI website](https://nlp.stanford.edu/projects/snli/) and write a dataloader based on your dataloader from **A3: Distributed Representations and Language Models**.
# 2. Use the `datasets` library to load the version on the [HuggingFace hub](https://huggingface.co/datasets/stanfordnlp/snli). Follow the steps in [the documentation](https://huggingface.co/docs/datasets/v2.19.0/loading#hugging-face-hub) for loading the dataset.
#
# [you can remove the template for whatever code you don't use]
#
# The data is organized as follows:
#
# * Column 1: Premise (sentence1)
# * Column 2: Hypothesis (sentence2)
# * Column 3: Relation (gold_label)
#
# **[3 marks]**

# %%
# %pip install -r packages.txt

# %%
import datasets
from datasets import load_dataset

# enable progress bars
datasets.enable_progress_bar()

# set verbose logging
datasets.logging.set_verbosity_info()

dataset = load_dataset("csv", data_files={
    "train": "./simple_snli_1.0/simple_snli_1.0_train.csv",
    "validation": "./simple_snli_1.0/simple_snli_1.0_dev.csv",
    "test": "./simple_snli_1.0/simple_snli_1.0_test.csv"
}, delimiter="\t", column_names=["premise", "hypothesis", "label"])

# %%
# print the first 10 items of the training dataset
for i in range(10):
    print(f"Item {i}:")
    print(f"Premise:    {dataset['train'][i]['premise']}")
    print(f"Hypothesis: {dataset['train'][i]['hypothesis']}")
    print(f"Label:      {dataset['train'][i]['label']}")
    print("-" * 50)

# %% [markdown]
# Notice that the dataset comes as a dictionary-like object with three splits: `'test'`, `'train'`, and `'validation'`. Each item is a dictionary containing a `'premise'`, `'hypothesis'`, and `'label'`.

# %% [markdown]
# ## 2. Tokenization
#
# This data does not come pre-tokenized. Instead of training our own tokenizer, we can use the BERT tokenizer like in the preivous assignment. Even though we aren't using BERT the tokenizer works with any model. See the documentation on [using a pretrained tokenizer](https://huggingface.co/docs/tokenizers/en/quicktour#using-a-pretrained-tokenizer). **[1 mark]**

# %%
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

ex = dataset['train'][0]
print("Example Premise:", ex['premise'])
print("Encoded IDs:", tokenizer.encode(ex['premise']).ids)

# %% [markdown]
# ## 2. Model

# %% [markdown]
# In this part, we'll build the model for predicting the relationship between H and P.
#
# We will process each sentence using an LSTM. Then, we will construct some representation of the sentence. When we have a representation for H and P, we will combine them into one vector which we can use to predict the relationship.

# %% [markdown]
# We will train a model described in [2], the BiLSTM with max-pooling model. The procedure for the model is roughly:
#
#     1) Encode the Hypothesis and the Premise using one shared bidirectional LSTM (or two different LSTMS)
#     2) Perform max over the tokens in the premise and the hypothesis
#     3) Combine the encoded premise and encoded hypothesis into one representation
#     4) Predict the relationship 

# %% [markdown]
# ### Creating a representation of a sentence
#
# Let's first consider step 2 where we perform pooling. There is a builtin function in pytorch for this, but we'll implement it from scratch. 
#
# Let's consider the general case, what we want to do for these methods is apply some function $f$ along dimension $i$, and we want to do this for all $i$'s. As an example we consider the matrix S with size ``(N, D)`` where N is the number of words and D the number of dimensions:
#
# $S = \begin{bmatrix}
#     s_{11} & s_{12} & s_{13} & \dots  & s_{1d} \\
#     s_{21} & s_{22} & s_{23} & \dots  & s_{2d} \\
#     \vdots & \vdots & \vdots & \ddots & \vdots \\
#     s_{n1} & s_{n2} & s_{n3} & \dots  & s_{nd}
# \end{bmatrix}$
#
# What we want to do is apply our function $f$ on each dimension, taking the input $s_{1d}, s_{2d}, ..., s_{nd}$ and generating the output $x_d$. 
#
# You will implement the max pooling method. When performing max-pooling, $max$ will be the function which selects a _maximum_ value from a vector and $x$ is the output, thus for each dimension $d$ in our output $x$ we get:
#
# \begin{equation}
#     x_d = max(s_{1d}, s_{2d}, ..., s_{nd})
# \end{equation}
#
# This operation will reduce a batch of size ``(batch_size, num_words, dimensions)`` to ``(batch_size, dimensions)`` meaning that we now have created a sentence representation based on the content of the representation at each token position. 
#
# Create a function that takes as input a tensor of size ``(batch_size, num_words, dimensions)`` then performs max pooling and returns the result (the output should be of size: ```(batch_size, dimensions)```). [**4 Marks**]

# %%
import torch

def max_pooling(input_tensor):
    output_tensor = ...
    return output_tensor

# test_unpooled = torch.rand(32, 100, 512)
# test_pooled = pooling(test_unpooled)
# print(test_pooled.size()) # should be torch.Size([32, 512])


# %% [markdown]
# ### Combining sentence representations
#
# Next, we need to combine the premise and hypothesis into one representation. We will do this by concatenating four tensors (the final size of our tensor $X$ should be ``(batch_size, 4d)`` where ``d`` is the number of dimensions that you use): 
#
# $$X = [P; H; |P-H|; P \cdot H]$$
#
# Here, what we do is concatenating P, H, P times H, and the absolute value of P minus H, then return the result.
#
# Implement the function. **[4 marks]**

# %%
def combine_premise_and_hypothesis(hypothesis, premise):
    output = ...
    return output

# test_hypothesis = test_pooled.clone()
# test_premise = test_pooled.clone()
# test_combined = combine_premise_and_hypothesis(test_hypothesis, test_premise)
# print(test_combined.size()) # should be torch.Size([32, 400])


# %% [markdown]
# ### Creating the model
#
# Finally, we can build the model according to the procedure given previously by using the functions we defined above. Additionaly, in the model you should use *dropout*. For efficiency purposes, it's acceptable to only train the model with either max or mean pooling. 
#
# Implement the model [**8 marks**]

# %%
import torch.nn as nn

class SNLIModel(nn.Module):
    def __init__(self, ...):
        # your code goes here
        self.embeddings = ...
        self.rnn = ...
        self.classifier = ...
        
    def forward(self, premise, hypothesis):
        p = ...
        h = ...
        
        p_pooled = pooling(...)
        h_pooled = pooling(...)
        
        ph_representation = combine_premise_and_hypothesis(...)
        predictions = ...
        
        return predictions


# %% [markdown]
# ## 3. Training

# %% [markdown]
# As before, implement the training and testing of the model. SNLI can take a very long time to train, so I suggest you only run it for one or two epochs. **[10 marks]** 
#
# **Tip for efficiency:** *when developing your model, try training and testing the model on one batch (for each epoch) of data to make sure everything works! It's very annoying if you train for N epochs to find out that something went wrong when testing the model, or to find that something goes wrong when moving from epoch 0 to epoch 1.*

# %%
epochs = 2
batch_size = 32

train_iter = dataset['train'].iter(batch_size=batch_size)

loss_function = ...
optimizer = ...
model = ...

for _ in range(epochs):
    for batch in train_iter:
    # train model
        ...
    
# test model after all epochs are completed

# %% [markdown]
# ## 4. Testing
#
# Test the model on the testset. For each example in the test set, compute a prediction from the model (`entailment`, `contradiction` or `neutral`). Compute precision, recall, and F1 score for each label. **[10 marks]**

# %% [markdown]
# Suggest a _baseline_ that we can compare our model against **[2 marks]**

# %% [markdown]
# **Your answer should go here**

# %% [markdown]
# Suggest some ways (other than using a baseline) in which we can analyse the models performance **[3 marks]**.

# %% [markdown]
# **Your answer should go here**

# %% [markdown]
# Suggest some ways to improve the model **[3 marks]**.

# %% [markdown]
# **Your answer should go here**

# %% [markdown]
# ## Readings
#
# [1] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP). 
#
# [2] Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). Supervised learning of universal sentence representations from natural language inference data. arXiv preprint arXiv:1705.02364.

# %% [markdown]
#

# %% [markdown]
# ## Your reflections on this lab
#
# Write below your general thoughts, experiences, or reflections on how you worked on this lab.

# %%

# %% [markdown]
# ## Statement of contribution
#
# Briefly state how many times you have met for discussions, who was present, to what degree each member contributed to the discussion and the final answers you are submitting.

# %%

# %% [markdown]
# ## Marks
#
# The assignment has 45 marks.
