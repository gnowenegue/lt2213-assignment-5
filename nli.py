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

# %%

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
    # input_tensor has shape: (batch_size, num_words, dimensions)
    # we take the maximum value along the words dimension (dim=1)
    # torch.max returns a tuple of (values, indices), we only need the values
    output_tensor, _ = torch.max(input_tensor, dim=1)
    return output_tensor

test_unpooled = torch.rand(32, 100, 512)
test_pooled = max_pooling(test_unpooled)
print(test_pooled.size()) # should be torch.Size([32, 512])


# %% [markdown]
# output:torch.Size([32, 512])

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
    # P times H
    product = premise * hypothesis

    # the absolute value of P minus H
    abs_diff = torch.abs(premise - hypothesis)

    output = torch.cat((hypothesis, premise, product, abs_diff), dim=1)
    return output


test_hypothesis = test_pooled.clone()
test_premise = test_pooled.clone()
test_combined = combine_premise_and_hypothesis(test_hypothesis, test_premise)
print(test_combined.size())  # should be torch.Size([32, 400])

# %% [markdown]
# Output:torch.Size([32, 2048])

# %% [markdown]
# ### Creating the model
#
# Finally, we can build the model according to the procedure given previously by using the functions we defined above. Additionaly, in the model you should use *dropout*. For efficiency purposes, it's acceptable to only train the model with either max or mean pooling. 
#
# Implement the model [**8 marks**]

# %%
import torch.nn as nn

# model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_CLASSES = 3
DROPOUT_RATE = 0.1


class SNLIModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    ):
        super(SNLIModel, self).__init__()

        # embed token ids to dense vectors
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)

        # shared bidirectional LSTM encoder
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

        # dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # linear classifier projecting combined representation to output classes
        # hidden_size * 2 because the LSTM is bidirectional
        # multiplied by 4 because we concatenate 4 vectors: P, H, |P-H|, and P*H
        self.classifier = nn.Linear(4 * (hidden_size * 2), num_classes)

    def forward(self, premise, hypothesis):
        # step 0: look up embeddings for both sentences
        p_emb = self.embeddings(premise)
        h_emb = self.embeddings(hypothesis)

        # step 1: encode both sentences using the shared BiLSTM
        p_encoded, _ = self.rnn(p_emb)
        h_encoded, _ = self.rnn(h_emb)

        # step 2: max pool over the words of both encoded sentences
        p_pooled = max_pooling(p_encoded)
        h_pooled = max_pooling(h_encoded)

        # step 3: combine premise and hypothesis representations
        ph_representation = combine_premise_and_hypothesis(p_pooled, h_pooled)

        # step 4: apply dropout and predict relationship classes
        ph_representation = self.dropout(ph_representation)
        predictions = self.classifier(ph_representation)

        return predictions


# %% [markdown]
# ## 3. Training

# %% [markdown]
# As before, implement the training and testing of the model. SNLI can take a very long time to train, so I suggest you only run it for one or two epochs. **[10 marks]** 
#
# **Tip for efficiency:** *when developing your model, try training and testing the model on one batch (for each epoch) of data to make sure everything works! It's very annoying if you train for N epochs to find out that something went wrong when testing the model, or to find that something goes wrong when moving from epoch 0 to epoch 1.*

# %%
from torch.nn.utils.rnn import pad_sequence

def tokenize_and_pad(sentences):
    # tokenize each sentence and pad to the same length within the batch
    encoded = [torch.tensor(tokenizer.encode(s).ids) for s in sentences]
    return pad_sequence(encoded, batch_first=True, padding_value=0)

epochs = 2
batch_size = 32
label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}

loss_function = nn.CrossEntropyLoss()
model = SNLIModel(vocab_size=len(tokenizer.get_vocab()))
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    model.train()
    train_iter = dataset['train'].iter(batch_size=batch_size)
    total_loss = 0
    batch_count = 0

    for batch in train_iter:
        # skip examples with missing fields or unknown labels
        valid = [
            (p, h, l)
            for p, h, l in zip(batch['premise'], batch['hypothesis'], batch['label'])
            if l in label_map and p is not None and h is not None
        ]
        if not valid:
            continue

        premises, hypotheses, batch_labels = zip(*valid)

        # tokenize and pad premises and hypotheses
        p_tensor = tokenize_and_pad(premises)
        h_tensor = tokenize_and_pad(hypotheses)
        l_tensor = torch.tensor([label_map[l] for l in batch_labels])

        optimizer.zero_grad()
        predictions = model(p_tensor, h_tensor)
        loss = loss_function(predictions, l_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if batch_count % 500 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_count} | Avg loss: {total_loss / batch_count:.4f}")

    print(f"Epoch {epoch+1} complete | Avg loss: {total_loss / batch_count:.4f}")

print("Training complete.")

# %% [markdown]
# Output: MLTGPU 
#
# Epoch 1 | Batch 500 | Avg loss: 0.9560
# Epoch 1 | Batch 1000 | Avg loss: 0.8953
# Epoch 1 | Batch 1500 | Avg loss: 0.8565
# Epoch 1 | Batch 2000 | Avg loss: 0.8303
# Epoch 1 | Batch 2500 | Avg loss: 0.8139
# Epoch 1 | Batch 3000 | Avg loss: 0.7970
# Epoch 1 | Batch 3500 | Avg loss: 0.7848
# Epoch 1 | Batch 4000 | Avg loss: 0.7747
# Epoch 1 | Batch 4500 | Avg loss: 0.7651
# Epoch 1 | Batch 5000 | Avg loss: 0.7576
# Epoch 1 | Batch 5500 | Avg loss: 0.7513
# Epoch 1 | Batch 6000 | Avg loss: 0.7444
# Epoch 1 | Batch 6500 | Avg loss: 0.7387
# Epoch 1 | Batch 7000 | Avg loss: 0.7332
# Epoch 1 | Batch 7500 | Avg loss: 0.7286
# Epoch 1 | Batch 8000 | Avg loss: 0.7243
# Epoch 1 | Batch 8500 | Avg loss: 0.7202
# Epoch 1 | Batch 9000 | Avg loss: 0.7158
# Epoch 1 | Batch 9500 | Avg loss: 0.7128
# Epoch 1 | Batch 10000 | Avg loss: 0.7096
# Epoch 1 | Batch 10500 | Avg loss: 0.7067
# Epoch 1 | Batch 11000 | Avg loss: 0.7030
# Epoch 1 | Batch 11500 | Avg loss: 0.6998
# Epoch 1 | Batch 12000 | Avg loss: 0.6969
# Epoch 1 | Batch 12500 | Avg loss: 0.6944
# Epoch 1 | Batch 13000 | Avg loss: 0.6925
# Epoch 1 | Batch 13500 | Avg loss: 0.6904
# Epoch 1 | Batch 14000 | Avg loss: 0.6879
# Epoch 1 | Batch 14500 | Avg loss: 0.6858
# Epoch 1 | Batch 15000 | Avg loss: 0.6835
# Epoch 1 | Batch 15500 | Avg loss: 0.6814
# Epoch 1 | Batch 16000 | Avg loss: 0.6793
# Epoch 1 | Batch 16500 | Avg loss: 0.6776
# Epoch 1 | Batch 17000 | Avg loss: 0.6755
# Epoch 1 complete | Avg loss: 0.6747
# Epoch 2 | Batch 500 | Avg loss: 0.6105
# Epoch 2 | Batch 1000 | Avg loss: 0.6104
# Epoch 2 | Batch 1500 | Avg loss: 0.6083
# Epoch 2 | Batch 2000 | Avg loss: 0.6037
# Epoch 2 | Batch 2500 | Avg loss: 0.6032
# Epoch 2 | Batch 3000 | Avg loss: 0.5999
# Epoch 2 | Batch 3500 | Avg loss: 0.5998
# Epoch 2 | Batch 4000 | Avg loss: 0.5993
# Epoch 2 | Batch 4500 | Avg loss: 0.5979
# Epoch 2 | Batch 5000 | Avg loss: 0.5968
# Epoch 2 | Batch 5500 | Avg loss: 0.5972
# Epoch 2 | Batch 6000 | Avg loss: 0.5959
# Epoch 2 | Batch 6500 | Avg loss: 0.5954
# Epoch 2 | Batch 7000 | Avg loss: 0.5941
# Epoch 2 | Batch 7500 | Avg loss: 0.5931
# Epoch 2 | Batch 8000 | Avg loss: 0.5919
# Epoch 2 | Batch 8500 | Avg loss: 0.5913
# Epoch 2 | Batch 9000 | Avg loss: 0.5904
# Epoch 2 | Batch 9500 | Avg loss: 0.5899
# Epoch 2 | Batch 10000 | Avg loss: 0.5891
# Epoch 2 | Batch 10500 | Avg loss: 0.5884
# Epoch 2 | Batch 11000 | Avg loss: 0.5871
# Epoch 2 | Batch 11500 | Avg loss: 0.5864
# Epoch 2 | Batch 12000 | Avg loss: 0.5856
# Epoch 2 | Batch 12500 | Avg loss: 0.5853
# Epoch 2 | Batch 13000 | Avg loss: 0.5850
# Epoch 2 | Batch 13500 | Avg loss: 0.5847
# Epoch 2 | Batch 14000 | Avg loss: 0.5837
# Epoch 2 | Batch 14500 | Avg loss: 0.5833
# Epoch 2 | Batch 15000 | Avg loss: 0.5826
# Epoch 2 | Batch 15500 | Avg loss: 0.5820
# Epoch 2 | Batch 16000 | Avg loss: 0.5814
# Epoch 2 | Batch 16500 | Avg loss: 0.5810
# Epoch 2 | Batch 17000 | Avg loss: 0.5802
# Epoch 2 complete | Avg loss: 0.5800
# Training complete.

# %% [markdown]
# ## 4. Testing
#
# Test the model on the testset. For each example in the test set, compute a prediction from the model (`entailment`, `contradiction` or `neutral`). Compute precision, recall, and F1 score for each label. **[10 marks]**

# %%
from sklearn.metrics import classification_report

model.eval()

all_preds = []
all_true = []

# run inference on the full test set without computing gradients
with torch.no_grad():
    for batch in dataset['test'].iter(batch_size=batch_size):
        # skip examples with missing fields or unknown labels
        valid = [
            (p, h, l)
            for p, h, l in zip(batch['premise'], batch['hypothesis'], batch['label'])
            if l in label_map and p is not None and h is not None
        ]
        if not valid:
            continue

        premises, hypotheses, batch_labels = zip(*valid)

        p_tensor = tokenize_and_pad(premises)
        h_tensor = tokenize_and_pad(hypotheses)

        # take the class with the highest logit as the prediction
        predictions = model(p_tensor, h_tensor)
        pred_labels = torch.argmax(predictions, dim=1).tolist()
        true_labels = [label_map[l] for l in batch_labels]

        all_preds.extend(pred_labels)
        all_true.extend(true_labels)

print(classification_report(
    all_true,
    all_preds,
    target_names=["entailment", "neutral", "contradiction"]
))

# %% [markdown]
# Output: MLTGPU
#
#              precision    recall  f1-score   support
#
#    entailment       0.79      0.85      0.82      3368
#       neutral       0.76      0.69      0.72      3219
# contradiction       0.79      0.80      0.80      3237
#
#      accuracy                           0.78      9824
#     macro avg       0.78      0.78      0.78      9824
#  weighted avg       0.78      0.78      0.78      9824

# %% [markdown]
# Suggest a _baseline_ that we can compare our model against **[2 marks]**

# %% [markdown]
# **Your answer should go here**
#
# A strong and simple baseline is a majority class baseline, where we always predict the most frequent label in the training data. Since SNLI is roughly balanced across entailment, contradiction, and neutral, this gives about ~33% accuracy, which serves as a lower bound that any trained model should outperform.
# A slightly stronger baseline is a bag-of-words(BOW) approach combined with logistic regression. In this method we represent the premise and hypothesis using averaged word embeddings (such as Glove or even random embeddings), concatenate them, and train a logistic regression classifier on top. This baseline is fast to train and typically achieves around 65 - 70% accuracy on SNLI, providing a good comparison point to show that more advanced models like a BiLSTM with max-pooling (~78%) are learning deeper sentence structure rather than just relying on surface-level word overlap.
#

# %% [markdown]
# Suggest some ways (other than using a baseline) in which we can analyse the models performance **[3 marks]**.

# %% [markdown]
# **Your answer should go here**
#
# We can analyse model performance in few ways beyond just overall accuracy or baseline comparison
#
# One approach is per-class error analysis, where we look at the confusion matrix to understand which labels are most often confused with each other, such as neutral vs entailment. We can also manually inspect incorrectly predicted examples to detect patterns in the errors, for example whether the models struggles with negation, numerical reasoning or longer and more complex sentences.
#
# Another useful method is sentence length analysis, where test examples are grouped into buckets based on the length of the premise or hypothesis. We then measure accuracy within each bucket to see whether performance drops for longer inputs, which is a common issue for models that rely on fixed- size sentence representations.
#
# Finally, we can do stress testing on challenging or adversarial examples, such as datasets designed to test negation, antonyms, or world knowledge. This helps us understand whether the model is truly learning inference or simply relying on shallow cues like word overlap.

# %% [markdown]
# Suggest some ways to improve the model **[3 marks]**.

# %% [markdown]
# **Your answer should go here**
#
# One way to improve the model is to use pre-trained word embeddings like Glove or fastText instead of randomly initialised embeddings. These embeddings already capture rich semantic relationships between words, which gives the model a much better starting point and usually improves performance on SNLI by a few percentage points.
#
# Another important is to include an attention mechanism instead of relying only on max-pooling. For example, a cross-attention layer can allow the hypothesis to attend to relevant parts of the premise (and vice versa), helping the model focus on important word alignments rather than compressing the entire sentence into a single fixed vector.
#
# We can also improve performance by using a stronger classifier and better training strategy. Instead of a single linear layer, a small multi-layer perceptron (MLP)with a non-linear activation (like ReLU) and dropout can capture more complex decision boundaries. In addition,training for more epochs and using learning rate scheduling e.g., reducing the learning rate when validation performance stops improving can help the model converge better.
#
# Finally a more advanced improvement is to fine-tune a pretrained transformer model like BERT on SNLI, which already encodes deep contextual knowledge and typically achieves much higher accuracy compared to LSTM-based models.
#
#

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

# %% [markdown]
# working on this lab helped us understand how NLI models actually work in practice, not just in theory. It was interesting to see how even a relatively simple model like a  BiLSTM can already give decent results, but still has clear limitations compared to more advanced pretrained models like BERT.
#
# We also got a better idea of why baselines are important. At first it felt like just an extra step, but comparing against a simple baseline made it much clearer whether the model was actually learning something useful. Doing error analysis was also quite helpful because it showed that the model sometimes depends on simple patterns like word overlap and can struggle with things like negation or more complex sentences.
#
# Overall the lab was useful for getting hands on experience with training and evaluating NLP models, and it made us think more about what the model is actually learning rather than just looking at accuracy numbers.
#

# %% [markdown]
# ## Statement of contribution
#
# Briefly state how many times you have met for discussions, who was present, to what degree each member contributed to the discussion and the final answers you are submitting.

# %%

# %% [markdown]
# ## Marks
#
# The assignment has 45 marks.
