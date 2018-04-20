
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Description-Task-1:-RNN-Language-Modelling-(30-+10-Points)" data-toc-modified-id="Description-Task-1:-RNN-Language-Modelling-(30-+10-Points)-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Description Task 1: RNN Language Modelling (30 +10 Points)</a></span><ul class="toc-item"><li><span><a href="#1a)-Language-Modelling-(30-Points)" data-toc-modified-id="1a)-Language-Modelling-(30-Points)-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>1a) Language Modelling (30 Points)</a></span></li><li><span><a href="#Conditional-Generation-(10-Points)" data-toc-modified-id="Conditional-Generation-(10-Points)-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Conditional Generation (10 Points)</a></span></li></ul></li><li><span><a href="#Code-for-Task-1" data-toc-modified-id="Code-for-Task-1-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Code for Task 1</a></span><ul class="toc-item"><li><span><a href="#Setup-and-preparation" data-toc-modified-id="Setup-and-preparation-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Setup and preparation</a></span></li><li><span><a href="#Data-preprocessing" data-toc-modified-id="Data-preprocessing-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Data preprocessing</a></span></li><li><span><a href="#RNN" data-toc-modified-id="RNN-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>RNN</a></span></li><li><span><a href="#Training" data-toc-modified-id="Training-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Evaluation</a></span></li><li><span><a href="#Conditional-Generation/Sampling-(Task-1.2)" data-toc-modified-id="Conditional-Generation/Sampling-(Task-1.2)-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Conditional Generation/Sampling (Task 1.2)</a></span></li><li><span><a href="#Pulling-everything-together" data-toc-modified-id="Pulling-everything-together-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Pulling everything together</a></span></li></ul></li></ul></div>

# # Natural Language Understanding: Project 1
# 
# [__Natural Language Understanding, Spring 2018, ETHZ__](http://www.da.inf.ethz.ch/teaching/2018/NLU/)
# 
# [__Project 1__ (ETHZ network)](http://www.da.inf.ethz.ch/teaching/2018/NLU/material/project.pdf)

# # Project to-do list:
# 
# Somewhat in order of importance:
# 
# - ~~change code to unroll RNN in time instead of using dynamic_rnn~~
# - ~~make sure the target data fed into the crossentropy metric is really in correct form~~
# - ~~try own implementation of basic RNN cell instead of TF-prefab RNN or LSTM cell~~
# - ~~change implementation to use the Xavier initializer instead of the uniform distribution currently used (see below)~~
# - ~~change all `tf.Variable` variable inits to the better practice form like `W = tf.get_variable(name='example', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())` which also includes the proper weight init~~
# - ~~include dropout at input and/or RNN cell level for regularization~~
# - ~~clean up namespaces, tensor naming~~
# - ~~(Started 2.4.2018, but TBC)** build in all reporting for Tensorboard~~
# - ~~adapt code to allow for differently sized timesteps~~
# - ~~make arrangements to save trained model~~
# - ~~implement perplexity function~~
# - ~~Maybe needs rewrite to use stock LSTM cell again** implement sampling function for conditional text generation~~
# - ~~implement result output function~~
# - ~~adapt code to allow for use of pretrained word2vec embedding~~
# - ~~cleanup code~~
# - run actual experiments

# ## Description Task 1: RNN Language Modelling (30 +10 Points)

# ### 1a) Language Modelling (30 Points)
# Your task is to build a simple LSTM language model. To be precise, we assume that words are independent given the recurrent hidden state; we compute a new hidden state given the last hidden state and last word, and predict the next word given the hidden state:
# $$ P(w_1,\dots,w_n) = 􏰀\prod_{t=1}^{n}P(w_t|\mathbf{h}_t)$$
# $$ P(w_t|\mathbf{h}_t) = \text{softmax}(\mathbf{Wh}_t)$$
# $$ \mathbf{h}_t = f(\mathbf{h}_{t−1}, w_{t-1}^{*})$$
# 
# where $f$ is the LSTM recurrent function, $\mathbf{W} \in \mathbb{R}^{|V|×d}$ are softmax weights and $\mathbf{h_0}$ is either an all-zero constant or a trainable parameter.
# You can use the tensorflow cell implementation __[1](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell)__ to carry out the recurrent computation in $f$. However, you must construct the actual RNN yourself (e.g. don’t use tensorflow’s `static_rnn` or `dynamic_rnn` or any other RNN library). That means, you will need to use a python loop that sets up the unrolled graph. To make your life simpler, please follow these design choices:
# 
# __Model and Data specification__
# 
# - Use a special sentence-beginning symbol `<bos>` and a sentence-end symbol `<eos>` (please use exactly these, including brackets). The `<bos>` symbol is the input, when predicting the first word and the `<eos>` symbol you require your model to predict at the end of every sentence.
# - Use a maximum sentence length of 30 (including the `<bos>` and `<eos>` symbol). Ignore longer sentences during training and testing.
# - Use a special padding symbol `<pad>` (please use exactly this, including brackets) to fill up sentences of length shorter than 30. This way, all your input will have the same size.
# - Use a vocabulary consisting of the 20K most frequent words in the training set, including the symbols `<bos>`, `<eos>`, `<pad>` and `<unk>`. Replace out-of-vocabulary words with the `<unk>` symbol before feeding them into your network (don’t change the file content).
# - Provide the ground truth last word as input to the RNN, not the last word you predicted. This is common practice.
# - Language models are usually trained to minimize the cross-entropy. Use tensorflow’s `tf.nn.sparse_softmax_cross_entropy_with_logits` to compute the loss (*This operation fuses the computation of the soft-max and the cross entropy loss given the logits. For numerical stability, it’s very important to use this function.*). Use the AdamOptimizer with default parameters to minimize the loss. Use `tf.clip_by_global_norm` to clip the norm of the gradients to 5.
# - Use a batch size of 64.
# - Use the data at __[6](https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB)__. Don’t pre-process the input further. All the data is already white-space tokenized and
# lower-cased. One sentence per line.
# - To initialize your weight matrices, use the `tf.contrib.layers.xavier_initializer()` initializer introduced in __[5](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)__.
# 
# __Experiments__
# All experiments should not run for longer than, say, four hours on the GPU. For this task, your
# grade won’t improve with performance.
# 
# - __Experiment A__: Train your model with word-embedding dimensionality of 100 and a hidden state size of 512 and compute sentence perplexity on the evaluation set (see submission format below).
# - __Experiment B__: It is common practice, to pretrain word embeddings using e.g. `word2vec`. This should make your model train faster as words will come already with some useful representation. Use the code at __[3](http://da.inf.ethz.ch/teaching/2018/NLU/material/load_embeddings.py)__ to load these word embeddings __[4](https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U)__ trained on the same corpus. Train your model again and compute evaluation perplexity.
# - __Experiment C__: It is often desirable to make the LSTM more powerful, by increasing the hidden dimensionality. However, this will naturally increase the parameters $\mathbf{W}$ of the softmax. As a compromise, one can use a larger hidden state, but down-project it before the softmax. Increase the hidden state dimensionality from 512 to 1024, but down-project $h_t$ to dimensionality 512 before predicting $w_t$ as in
# $$ \mathbf{\tilde{h}}_t = \mathbf{W}_P\mathbf{h}_t$$
# where $W_P$ are parameters. Train your model again and compute evaluation perplexity.
# 
# __Submission and grading__
# - Grading scheme: 100% correctness.
# - Deadline April 20th, 23:59:59.
# - You are not allowed to copy-paste any larger code blocks from existing implementations.
# - Hand in
#     - Your python code
#     - __Three__ result files containing sentence-level perplexity numbers on the __test__ set (to be distributed) for
# all three experiments. Recall that perplexity of a sentence $S = ⟨w_1, \dots , w_n⟩$ with respect to your model $p(w_t|w_1, \dots, w_{t−1})$ is defined as
# $$ \text{Perp} = 2^{-\frac{1}{n} \sum_{t=1}^{n}\log_2 p(w_t|w_1,\dots,w_{t−1})}$$
# The `<eos>` symbol is part of the sequence, while the `<pad>` symbols (if any) are not. Be sure to have the basis of the exponential and the logarithm match.<br>
# __Input format sentences.test__<br>
# One sentence (none of them is longer than 28 tokens) per line:<br>
#          ```beside her , jamie bounced on his seat .
#          i looked and saw claire montgomery looking up at me .
#          people might not know who alex was , but they knew to listen to him .```<br>
# __Required output format groupXX.perplexityY__<br>
# (where XX is your group __number__ and Y ∈ {A,B,C} is the experiment). One perplexity number per line:<br>
#          $10.232$<br>
#          $2.434$<br>
#          $5.232$<br>
# Make sure to have equally many lines in the output as there are in the input – otherwise your submission will be rejected automatically.
#     - You have to submit at https://cmt3.research.microsoft.com/NLUETHZ2018

# ### Conditional Generation (10 Points)
# Let’s use your trained language model from above to generate sentences. Given an initial sequence of words, your are asked to __greedily__ generate words until either your model decides to finish the sentence (it generated `<eos>`) or a given maximum length has been reached. Note, that this task does not involve any training. Please see the tensorflow documentation on how to save and restore your model from above.
# There are several ways how to implement the generation. For example, you can define a graph that computes just one step of the RNN given the last input and the last state (both from a new placeholder).
# $$ \text{state}_t, p_t = f(\text{state}_{t−1},w_{t−1}) $$
# That means, for a prefix of size $m$ and a desired length of $n$, you run this graph $n$ times. The first $m + 1$ times you take the input form the prefix. For the rest of the sequence, you take the most likely2 word $w^{t−1} = \text{argmax}_w p_{t−1}(w)$ from the last step.
# 
# - Grading scheme: 100% correctness.
# - Deadline April 20th, 23:59:59.
# - You are not allowed to copy-paste any larger code blocks from existing implementations.
# - Hand in
#     - Your python code
#     - Your continued sentences of length up to 20. Use your trained model from experiment __C__ in task 1.1.
#     __Input format sentences.continuation__ One sentence (of length less than 20) per line:<br>
#          ```beside her ,
#          i
#          people might not know```<br>
#     The `<bos>` symbol is not explicitly in the file, but you should still use it as the first input.<br>
#     __Required output format groupXX.continuation__ (where XX is your __group number__)<br>
#          ```beside her , something happened ! <eos>
#          i do n’t recall making a noise , but i must have , because bob just looked up from his
#          people might not know the answer . <eos>```
#     - You have to submit at https://cmt3.research.microsoft.com/NLUETHZ2018
# 

# __Infrastructure__
# 
# You must use Tensorflow, but any programming language is allowed. However, we strongly recommend `python3`. You have access to two compute resources: Unlimited CPU usage on Euler and GPU usage on Leonhard. Note that the difference in speed is typically a factor between 10 and 100.

# ## Code for Task 1
# 

# ### Setup and preparation

# Make sure you have done the following:
# 
# - Download data from https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB and unpack into `./data/` subdirectory
# - Download embeddings from https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U and unpack into `./data/` subdirectory
# - Download helper function from http://da.inf.ethz.ch/teaching/2018/NLU/material/load_embeddings.py and put into `./helpers/` subdirectory

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0: ANY, 1: filter INFO, 2: filter WARNINGS, 3: filter ERROR
import time

import tensorflow as tf
import numpy as np

from collections import Counter
from gensim import models
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class RNNConfig():
    '''Class holding all configuration vars for training the RNN
    
    '''
    
    def __init__(self,
                 data_dir='./data',
                 out_dir='./outputs',
                 save_dir='./{}-checkpoints',
                 validation_split=.1,
                 max_sentence_length=30,
                 vocab_length=20000,
                 embedding_size=100,
                 external_embedding=False,
                 rnn_size=512,
                 rnn_size_factor=1,
                 n_steps=30,
                 learning_rate=0.001,
                 keep_prob=0.5,
                 grad_clip=5,
                 batch_size=64,
                 num_epochs=1,
                 save_every_n=500,
                 validate_every_n=250,
                 report_every_n=20,
                 summary_every_n=10,
                 max_to_keep=5,
                 init_scale=.1):
        
        # Data directory 
        self.data_dir = data_dir
        
        # Output directory 
        self.out_dir = out_dir
        
        # Save directory
        self.save_dir = save_dir
        
        # Percentage of the training data used for validation (default: 10%)
        self.validation_split = validation_split
        
        # Maximum source sentence length as given by task description (default: 30)
        self.max_sentence_length = max_sentence_length
        
        # Size of the vocabulary (default: 20k)
        self.vocab_length = vocab_length
        
        # Dimensionality of word embedding layer (default: 100)
        self.embedding_size = embedding_size
        
        # Whether to use externally fed embedding (default: False)
        self.external_embedding = external_embedding
        
        # Dimensionality of RNN (i.e. hidden) layer (default: 512)
        self.rnn_size = rnn_size
        
        # Integer to factor the size of the hidden layer [Task 1.1C] (default: 1)
        self.rnn_size_factor = rnn_size_factor
        
        # Number of time steps for RNN (default: 30)
        self.n_steps = n_steps
        
        # Learning rate (default: 0.001)
        self.learning_rate = learning_rate
        
        # Dropout rate (default: 0.5)
        self.keep_prob = keep_prob
        
        # Gradient clipping treshold (default: 5.0)
        self.grad_clip = grad_clip
        
        # Batch Size (default: 64)
        self.batch_size = batch_size
        
        # Number of training epochs (default: 10)
        self.num_epochs = num_epochs
        
        # Save model after this many steps (default: 500)
        self.save_every_n = save_every_n
        
        # Validate after this many steps (default: 250)
        self.validate_every_n = validate_every_n
        
        # Print report after this many steps (default: 20)
        self.report_every_n = report_every_n
        
        # Record summary after this many steps (default: 10)
        self.summary_every_n = summary_every_n
        
        # Number of checkpoints to save (default: 5)
        self.max_to_keep = max_to_keep
        
        # Scale range for uniform variable inits (default: 0.1)
        self.init_scale = init_scale


# In[3]:


# Initialize config variables to defaults
CONFIG = RNNConfig()

# Define shorthands for common initializers used all over code
ones = tf.ones_initializer()
unif = tf.random_uniform_initializer(-CONFIG.init_scale, CONFIG.init_scale)
xavi = tf.contrib.layers.xavier_initializer()
zeros = tf.zeros_initializer()

# Read all data from files
data_dir = CONFIG.data_dir

with open(data_dir+'/sentences.train', 'r') as f:
    train_data = f.read()
    
with open(data_dir+'/sentences.eval', 'r') as f:
    eval_data = f.read()
    
with open(data_dir+'/sentences.continuation', 'r') as f:
    continuation_data = f.read()
    
with open(data_dir+'/sentences_test.dms', 'r') as f:
    test_data = f.read()


# In[4]:


# Have a peek at the given raw data

print('Training data sample:\n', 20*'=')
print(train_data[:100], '\n', 80*'.')

print('\n Evaluation data sample:\n', 20*'=')
print(eval_data[:100], '\n', 80*'.')

print('\n Continuation data sample:\n', 20*'=')
print(continuation_data[:100], '\n', 80*'.')

print('\n Test data sample:\n', 20*'=')
print(test_data[:100], '\n', 80*'.')


# ### Data preprocessing

# In[5]:


# Splitting data into sentences
def split_data2sentences(data):
    text = ''.join(data)
    sentences = text.split('\n')
    return sentences

train_sentences = split_data2sentences(train_data[:-1]) #[:-1] to get rid of trailing '\n'
eval_sentences = split_data2sentences(eval_data[:-1])
continuation_sentences = split_data2sentences(continuation_data[:-1])
test_sentences = split_data2sentences(test_data[:-1])

# Get sentences from training data and look at sample
print('Sample training sentences:\n', train_sentences[:5], '\n')

# Make text contiguous again, break into words for vocabulary and look at sample
words = ' '.join(train_sentences).split()
print('Sample words:\n', words[:20], '\n')

## Generate the dictionary from the training data
# Make a word counter and show top frequency words
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
print('Top frequency words:\n', vocab[:20], '\n')

# Clip word counter to defined length and append special symbol words
symbols = ['<bos>', '<eos>', '<pad>', '<unk>']
vocab = vocab[:CONFIG.vocab_length-len(symbols)] # Limit to CONFIG.vocab_length minus the last 4 to replace w/ symbols
for each in symbols:
    vocab.append(each)
    
# Make a vocabulary to convert words to integers
vocab_to_int = {word: i for i, word in enumerate(vocab, 0)} # consider starting with 1 if 0 gives dead cells

# Make a vocabulary to get words from integers at the end
int_to_vocab = dict(enumerate(vocab))


# In[6]:


# Encode sentences to integers and insert symbol words where necessary

### ATTN: Remove next line after finishing, keeping data set small for speedup
# train_sentences = train_sentences[:2000]

def encode_sentences(sentences):
    max_sentence_length = CONFIG.max_sentence_length # Given by task description
    sentences_ints = [] # List to hold converted-to-int sentences
    for each in tqdm(sentences):
        sentence = each.split()
        if len(sentence) <= max_sentence_length-2: # -2 to allow for <bos>, <eos>
            sentence_int = [vocab_to_int['<bos>']] # Start sentence list w/ <bos>
            sentence_int += [vocab_to_int.get(word, vocab_to_int['<unk>']) for word in sentence]            
            sentence_int.append(vocab_to_int['<eos>']) # End sentence w/ <eos>
            while len(sentence_int) < max_sentence_length: # Pad length if necessary
                sentence_int.append(vocab_to_int['<pad>'])
            sentences_ints.append(sentence_int) 
    encoded = np.array(sentences_ints) # Convert list of sentences to np array
    return encoded

train_encoded = encode_sentences(train_sentences)
eval_encoded = encode_sentences(eval_sentences)
test_encoded = encode_sentences(test_sentences)
# Set data preparation complete flag
data_ready = True


# In[7]:


# Modified load_embedding code not to mess with own session handling code

def load_embedding(vocab, path, dim_embedding, vocab_size):
    '''
      vocab          A dictionary mapping token strings to vocabulary IDs
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
      vocab_size     Size of the vocabulary to be embedded
      Returns np.ndarray of size (vocab_size, dim_embedding) containing pretrained embedding
    '''

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)  
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    return external_embedding


# In[8]:


# Load external embedding
filename = 'wordembeddings-dim100.word2vec'
external_embedding = load_embedding(vocab_to_int, data_dir+'/'+filename, CONFIG.embedding_size, CONFIG.vocab_length)

# Set external embeddingready flag
external_emb_ready = True


# ### RNN 

# In[9]:


def variable_summaries(var):
    '''Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    
        From TensorBoard documentation
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# In[10]:


class RNNLanguageModel:
    '''Main element: Class representing the complete RNN language model
    
    TBC
    '''
    
    def __init__(self,
                 vocab_length,
                 batch_size,
                 n_steps,
                 rnn_size,
                 rnn_size_factor,
                 learning_rate,
                 grad_clip,
                 embedding_size,
                 external_embedding):
        
        # Reset tensorflow graph for clean slate
        tf.reset_default_graph()
        
        # TF Placeholders:
        with tf.name_scope('input_layer'):
            self.inputs = tf.placeholder(tf.int32, [batch_size, n_steps], name='inputs')
            self.targets = tf.placeholder(tf.int64, [batch_size, n_steps], name='targets')
            self.target_weights = tf.placeholder(tf.float32, [batch_size, n_steps],
                                                 name='target_weights')

            self.ext_embedding_matrix = tf.placeholder(tf.float32, [vocab_length, embedding_size],
                                                       name='ext_embedding_matrix')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer
        with tf.name_scope('embedding_layer'):
            # Create embedding matrix
            embedding_matrix = tf.get_variable(name='embedding_matrix',
                                           shape=[vocab_length, embedding_size],
                                           initializer=unif)

            # If we're using the pretrained embedding
            if (external_embedding == True):
                embedding_matrix.assign(self.ext_embedding_matrix)

            # Lookup inputs in embedding matrix
            self.embeddings = tf.nn.embedding_lookup(embedding_matrix, self.inputs,
                                                         name='embeddings')

            tf.summary.histogram('embeddings', self.embeddings)

            # Embedding layer dropout (use during training)
            self.embeddings = tf.nn.dropout(self.embeddings, self.keep_prob, name='embeddings_dropout')
            
           # if (self.keep_prob < 1):
           #     self.embeddings = tf.nn.dropout(self.embeddings, keep_prob, name='embeddings_dropout')

        # RNN layer
        with tf.name_scope('hidden_layer'):

            # RNN cell: GRUCell trains faster than BasicLSTMCell w/ similar results
            cell = tf.nn.rnn_cell.GRUCell(rnn_size*rnn_size_factor)

            # Dropout wrapper (use during training)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
            
            # Initialize cell to zero state
            self.initial_state = cell.zero_state(batch_size, tf.float32)

            # Unroll RNN through time
            # dynamic_rnn solution would be:
            # self.rnn_output, self.final_state = tf.nn.dynamic_rnn(cell, self.embeddings, initial_state=self.initial_state)

            state = self.initial_state 
            outputs = []
            with tf.variable_scope('RNN'):
                for i in range(n_steps):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = cell(self.embeddings[:, i, :], state)
                    outputs.append(output)
            self.rnn_output, self.final_state = tf.stack(outputs, axis=1), state

            # to keep only last output would be:
            # self.rnn_output = self.rnn_output[:, -1, :]

            # Reshape hidden layer output: one row per input and step, i.e. ((batch_size*n_steps), rnn_size)            
            self.rnn_output = tf.reshape(self.rnn_output, [-1, rnn_size*rnn_size_factor])

            variable_summaries(self.rnn_output)    

            # If hidden layer increased by factor [1.1C], project down
            if (rnn_size_factor > 1):
                W_p = tf.get_variable(name='downprojection_weight', shape=[rnn_size*rnn_size_factor,
                                                                          rnn_size],
                                      initializer=xavi)
                b_p = tf.get_variable(name='downprojection_bias', shape=[rnn_size,],
                                      initializer=zeros)

                self.rnn_output = tf.nn.xw_plus_b(self.rnn_output, W_p, b_p)

                variable_summaries(W_p)
                variable_summaries(b_p)

        # Softmax output layer
        with tf.name_scope('softmax_layer'):

            # out_size is vocab_length
            out_size = vocab_length

            # RNN outputs to softmax layer:
            W_softmax = tf.get_variable(name="softmax_weight", shape=[rnn_size, out_size],
                                        initializer=xavi)
            b_softmax = tf.get_variable(name="softmax_bias", shape=[out_size],
                                        initializer=zeros)

            variable_summaries(W_softmax)
            variable_summaries(b_softmax)

            # Calculate logits from softmax layer
            self.logits = tf.nn.xw_plus_b(self.rnn_output, W_softmax, b_softmax, name='logits')
            
            variable_summaries(self.logits)
            
            # Finally, get word probabilities from logits
            self.predictions = tf.nn.softmax(self.logits, name='predictions')

        # Metrics
        with tf.name_scope('metrics'):

            # Reshape targets and their weights from (batch_size, n_steps) to 1D
            y = tf.reshape(self.targets, [-1])
            y_weights = tf.reshape(self.target_weights, [-1])

            # Calculate crossentropy: Use sparse routine for numerical stability
            # and also to avoid having to one-hot encode the targets
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                      logits=self.logits)

            # Multiply crossentropy vector by 1D-reshaped weights to nullify effect of <pad>
            crossent *= y_weights

            # Get scalar crossentropy: With aim to average over both n_steps and batch_size
            crossent = tf.reduce_sum(crossent)

            # Calculate total weight and make sure it doesn't equal zero for the division below
            total_weight = tf.reduce_sum(y_weights)
            total_weight += 1e-12

            # Calculate total loss (across batch_size AND n_steps): Divide summed crossentropy by total weight
            self.loss = crossent / total_weight

            tf.summary.scalar('loss', self.loss)

            # Calculate perplexity: Uses natural logarithm as does sparse_softmax_cross_entropy_with_logits
            self.perplexity = tf.exp(self.loss)

            tf.summary.scalar('perplexity', self.perplexity)

            # Best prediction
            self.best_prediction = tf.argmax(self.predictions, 1, name='best_prediction')

            # Accuracy
            correct_predictions = tf.cast(tf.equal(self.best_prediction, y), tf.float32)
            correct_predictions *= y_weights      
            correct_predictions = tf.reduce_sum(correct_predictions)
            self.accuracy = correct_predictions / total_weight

            tf.summary.scalar('accuracy', self.accuracy)

        # Optimizer
        with tf.name_scope('optimizer'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            train_op = tf.train.AdamOptimizer(learning_rate)
            self.optimizer = train_op.apply_gradients(zip(grads, tvars))      

        # Merge summaries for TensorBoard    
        self.merged = tf.summary.merge_all()


# In[11]:


# Build batch generator for training    
def get_batches(source_arr, batch_size, n_steps, shuffle=True):
    '''Generator which returns features x and targets y
    
    Inputs
    ------
    source_arr: A np.ndarray of sentences in rows to generate features and targets from
    batch_size: An int number of sequences required per batch
    n_steps: Number of time steps for RNN to consider; defines sequence length
    
    Outputs
    -------
    x: A np.ndarray of feature sequences according to parameters above
    y: A np.ndarray of target sequences according to parameters above
    y_weights: A np.ndarray of floats: 0.0 were target sequence element = <pad>, 1.0 else
    '''
    
    source_rows = source_arr.shape[0]
    source_cols = source_arr.shape[1]
    
    # Make sure time steps doesn't exceed available information in sentence (+1 is the wraparound)
    assert (n_steps <  source_cols+1), "No point in looking further back than source is long"
    
    # Wrap around source col 0 <bos> to end for full length unpadded source_cols, otherwise <pad>
    wrap_around = np.array([vocab_to_int['<bos>'] if (source_arr[i, -1] == vocab_to_int['<eos>'])                             else vocab_to_int['<pad>'] for i in range(source_rows)])
    source_arr = np.hstack((source_arr, wrap_around.reshape((-1, 1))))
    
    # How many sequences we can get from an input sentence (row)
    n_seq_per_row = source_cols+1 - n_steps 
        
    # How many 'batch blocks' we can get
    n_blocks = source_rows // batch_size
    
    # Shuffle rows of sequences to improve training
    if shuffle:
        np.random.shuffle(source_arr)
    
    # Crop array to only produce full batches
    source_arr = source_arr[:n_blocks*batch_size, :]

    # Reshape source_arr for easier batch generation
    source_arr = source_arr.reshape((batch_size, -1))

    # Generate batches
    for j in range(0, source_arr.shape[1], source_cols+1):
        for jj in range(n_seq_per_row):
            # Feature sequence:
            x = source_arr[:, j+jj : j+jj+n_steps]
            # Target:
            y = source_arr[:, j+jj+1:j+jj+n_steps+1]
            # Target weights:
            y_weights = np.array([[0. if (int_to_vocab[j] == '<pad>') else 1. for j in i] for i in y])
            yield x, y, y_weights


# In[12]:


# Development only: playing around with batches

shuffled_rows_ind = np.random.permutation(len(train_encoded))
validation_split_ind = int(0.1 * len(train_encoded))
source_train = train_encoded[shuffled_rows_ind[validation_split_ind:]]
batches = get_batches(source_train, 3, 30)

x, y, y_weights = next(batches)
print('x\n',x.shape,'\n', x, '\n')
print('y\n',y.shape,'\n', y, '\n')
print('y_weights\n',y_weights.shape,'\n', y_weights, '\n')
for i in range(x.shape[0]):
        print([int_to_vocab[x[i,j]] for j in range(x[i].shape[0])], ' -> ', [int_to_vocab[y[i,j]] for j in range(y[i].shape[0])])


# ### Training

# In[12]:


# Run training on exp for epochs with rnn_size_factor factor and external_embedding emb and save to save_dir
def train(exp, CONFIG):
    # Check data preparation complete, external embedding ready
    assert data_ready == True, 'Need to run data preparation first'
    assert external_emb_ready == True, 'Need to import external embedding first'

    # Split training/validation data
    np.random.seed(42)
    shuffled_rows_ind = np.random.permutation(len(train_encoded))
    validation_split_ind = int(CONFIG.validation_split * len(train_encoded))
    source_train = train_encoded[shuffled_rows_ind[validation_split_ind:]]
    source_validation = train_encoded[shuffled_rows_ind[:validation_split_ind]]

    # Set shorter handles for training loop variables
    save_every_n = CONFIG.save_every_n
    validate_every_n = CONFIG.validate_every_n
    report_every_n = CONFIG.report_every_n
    summary_every_n = CONFIG.summary_every_n
    keep_prob = CONFIG.keep_prob
    epochs = CONFIG.num_epochs
    save_dir = CONFIG.save_dir.format(exp)
    
    # Set some helper variables
    num_training_steps = len(source_train) // CONFIG.batch_size                          * (source_train.shape[1]+1 - CONFIG.n_steps) * epochs
    report_detail = False
    if CONFIG.n_steps < 15: # This is just to keep the screen from overflowing
        report_detail = True # Only eye-candy anyway
    def str_hms(delta_time):
        h, rem = divmod(delta_time, 3600)
        m, s = divmod(rem, 60)
        return '{:0>2}:{:0>2}:{:05.2f}'.format(int(h),int(m),s)

    # Create model instance
    model = RNNLanguageModel(vocab_length=CONFIG.vocab_length,
                             batch_size=CONFIG.batch_size,
                             n_steps=CONFIG.n_steps,
                             rnn_size=CONFIG.rnn_size,
                             rnn_size_factor=CONFIG.rnn_size_factor,
                             learning_rate=CONFIG.learning_rate,
                             grad_clip=CONFIG.grad_clip,
                             embedding_size=CONFIG.embedding_size,
                             external_embedding=CONFIG.external_embedding)

    # Setup TensorBoard logging
    # Run in project directory to view: "tensorboard --logdir ./runs"
    now = time.strftime('%y-%m-%d-%H-%M-%S')
    log_dir = './runs'
    log_subdir = '{}/{}-run-{}/'.format(log_dir, exp, now)
    writer = tf.summary.FileWriter(log_subdir, tf.get_default_graph())
    
    # Setup model saving
    saver = tf.train.Saver(max_to_keep=CONFIG.max_to_keep)

    # Run training
    print('Starting training on experiment {}:'.format(exp))
    print('Run "tensorboard --logdir {}" in project dir to monitor, current run is {}'.format(log_dir, log_subdir))
    print('Batch size: {}\t Time steps: {}\t RNN layer size: {}\t External embedding: {}'.format(CONFIG.batch_size,
                                                                                           CONFIG.n_steps,
                                                                                           CONFIG.rnn_size*CONFIG.rnn_size_factor,
                                                                                           CONFIG.external_embedding))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_start = time.time()
        counter = 1

        for ep in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0.

            validation_batches = get_batches(source_validation, batch_size=CONFIG.batch_size,
                                    n_steps=CONFIG.n_steps)

            for x_train, y_train, y_train_weights in get_batches(source_train, batch_size=CONFIG.batch_size,
                                    n_steps=CONFIG.n_steps):
                
                start = time.time()

                feed_dict = {model.inputs: x_train,
                             model.targets: y_train,
                             model.target_weights: y_train_weights,
                             model.keep_prob: keep_prob,
                             model.ext_embedding_matrix: external_embedding,
                             model.initial_state: new_state}

                loss, perplexity, accuracy, best_prediction, new_state, _ =                 sess.run([model.loss, model.perplexity, model.accuracy, model.best_prediction,
                          model.final_state, model.optimizer], feed_dict=feed_dict)

                batch_time = time.time() - start

                if (counter % summary_every_n == 0):
                    writer.add_summary(sess.run(model.merged, feed_dict=feed_dict), counter)

                if (counter % report_every_n == 0):
                    print('Epoch: {}/{}\t'.format(ep+1, epochs),
                          'Train step: {}/{}\t'.format(counter, num_training_steps-1),
                          'Batch loss: {:.3f}\t'.format(loss),
                          'Batch perplexity: {:.3f}\t'.format(perplexity),
                          'Batch accuracy: {:.1%}\t'.format(accuracy),
                          '{:.2f}s/batch'.format(batch_time))
                    if (report_detail == True):
                        print(' '.join([int_to_vocab[x_train[0,j]] for j in range(x_train[0].shape[0])]),
                              ' -> ', int_to_vocab[best_prediction[0]] , ' vs. ', int_to_vocab[y_train[0, -1]])

                if (counter % save_every_n == 0):
                    saver.save(sess, '{}/run-{}_i{}_s{}.ckpt'.format(save_dir, now,
                                                                     counter, CONFIG.rnn_size))
                if (counter % validate_every_n == 0):
                    x_validation, y_validation, y_validation_weights = next(validation_batches)
                    feed_dict = {model.inputs: x_validation,
                                 model.targets: y_validation,
                                 model.target_weights: y_validation_weights,
                                 model.keep_prob: 1.0,
                                 model.ext_embedding_matrix: external_embedding,
                                 model.initial_state: new_state}

                    loss, perplexity, accuracy = sess.run([model.loss, model.perplexity, model.accuracy],
                                                          feed_dict=feed_dict)
                    
                    print('Epoch: {}/{}\t'.format(ep+1, epochs),
                          '* Validation batch *\t',
                          'Batch loss: {:.3f}\t'.format(loss),
                          'Batch perplexity: {:.3f}\t'.format(perplexity),
                          'Batch accuracy: {:.1%}\t***'.format(accuracy))
                
                counter += 1
           
        train_time = time.time() - train_start

        writer.close()
        saver.save(sess, '{}/run-{}_i{}_s{}.ckpt'.format(save_dir, now, counter, CONFIG.rnn_size))

        print('*** Experiment {}: Training complete:\t{} epoch(s) with {} batches each'.format(exp,
                                                                                               epochs,
                                                                                               num_training_steps))
        print('*** Training time:\t', str_hms(train_time))


# ### Evaluation

# In[13]:


def eval_perplexities(checkpoint, CONFIG, test):
    perplexities = []
    counter = 1
    
    model = RNNLanguageModel(vocab_length=CONFIG.vocab_length,
                             batch_size=1,
                             n_steps=CONFIG.n_steps,
                             rnn_size=CONFIG.rnn_size,
                             rnn_size_factor=CONFIG.rnn_size_factor,
                             learning_rate=CONFIG.learning_rate,
                             grad_clip=CONFIG.grad_clip,
                             embedding_size=CONFIG.embedding_size,
                             external_embedding=CONFIG.external_embedding)
    if test:
        perp_target = test_encoded
    else:
        perp_target = eval_encoded
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        for x_eval, y_eval, y_eval_weights in get_batches(perp_target, batch_size=1,
                                    n_steps=30, shuffle=False):

            if (counter % 50 == 0): print('Evaluating sentence {}/{}'.format(counter, len(perp_target)))
            #if counter == 100 : break ### Dev only

            new_state = sess.run(model.initial_state)
            feed_dict = {model.inputs: x_eval,
                         model.targets: y_eval,
                         model.target_weights: y_eval_weights,
                         model.keep_prob: 1,
                         model.ext_embedding_matrix: external_embedding,
                         model.initial_state: new_state}

            perplexity = sess.run(model.perplexity, feed_dict=feed_dict)
            perplexities.append(perplexity)
            counter += 1          
    return perplexities


# In[14]:


# Get perplexities for experiment based on last checkpoint in save_dir and write to file
def perp(exp, CONFIG, test=False):
    
    save_dir = CONFIG.save_dir.format(exp)
    
    # Get latest checkpoint
    checkpoint = tf.train.latest_checkpoint(save_dir)

    # Calculate perplexities:
    perplexities = eval_perplexities(checkpoint, CONFIG, test)
    
    # Write to file
    filename = 'group01.perplexity{}'.format(exp)
    if test:
        filename += '.test_set'
    if not os.path.exists(CONFIG.out_dir):
            os.makedirs(CONFIG.out_dir)
    with open('{}/{}'.format(CONFIG.out_dir, filename), 'w') as f:
        for p in perplexities:
            f.write('{:.3f}\n'.format(p))
    print('Experiment {}: Using {}, evalulation perplexities:'.format(exp, checkpoint))
    print('mean: {:.3f} std: {:.3f} min: {:.3f}, max: {:.3f}'.format(np.mean(perplexities),
                                                                     np.std(perplexities),
                                                                     np.min(perplexities),
                                                                     np.max(perplexities)))
    print('{} perplexity values written to {}/{}'.format(len(perplexities), CONFIG.out_dir, filename))


# ### Conditional Generation/Sampling (Task 1.2)

# In[15]:


def generate_samples(checkpoint, max_generate_n, CONFIG):
    
    # Generate continuation text
    # n_samples = 100 # Dev only, use next line for production
    n_samples = len(continuation_sentences)
   
    continued_sentences = []
    counter = 1
    
    model = RNNLanguageModel(vocab_length=CONFIG.vocab_length,
                             batch_size=1, # for sampling: Feeding single words
                             n_steps=1,# for sampling: Feeding single words
                             rnn_size=CONFIG.rnn_size,
                             rnn_size_factor=CONFIG.rnn_size_factor,
                             learning_rate=CONFIG.learning_rate,
                             grad_clip=CONFIG.grad_clip,
                             embedding_size=CONFIG.embedding_size,
                             external_embedding=CONFIG.external_embedding)
    
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        saver.restore(sess, checkpoint)
        for i in range(n_samples):
            new_state = sess.run(model.initial_state)
            
            sentence = [w for w in continuation_sentences[i].split()]
            primer_feed = [vocab_to_int['<bos>']]
            [primer_feed.append(vocab_to_int.get(word, vocab_to_int['<unk>'])) for word in sentence]
            
            for w in primer_feed:
                x = np.atleast_2d(w)
                feed_dict = {model.inputs: x,
                             model.keep_prob: 1.0,
                             model.initial_state: new_state}
                best_prediction, new_state =                 sess.run([model.best_prediction, model.final_state], feed_dict=feed_dict)
            sentence.append(int_to_vocab[best_prediction[0]]) # First generated word from trigger samples
            while len(sentence) < max_generate_n: # Generating remaining words from predictions
                x[0, 0] = best_prediction
                feed_dict = {model.inputs: x,
                             model.keep_prob: 1.0,
                             model.initial_state: new_state}
                best_prediction, new_state =                 sess.run([model.best_prediction, model.final_state], feed_dict=feed_dict)
                sentence.append(int_to_vocab[best_prediction[0]])
                if (sentence[-1] == '<eos>'):
                    break 
            output = ' '.join(sentence)
            if (counter % 50 == 0):
                print('Generated sentence {}/{}'.format(counter, n_samples))
                print(continuation_sentences[i], ' -> ', output)
  
            continued_sentences.append(output)
            counter += 1
    return continued_sentences


# In[16]:


# Generate samples based on save_dir and write to out_dir
def sample(exp, CONFIG):
    save_dir = CONFIG.save_dir.format(exp)
    
    # Get latest checkpoint
    checkpoint = tf.train.latest_checkpoint(save_dir)

     # Sentence generation length limit
    max_generate_n = 20 # Hard-coded by task description
    
    # Generate samples
    continued_sentences = generate_samples(checkpoint, max_generate_n, CONFIG)
    
    # Write to file
    filename = 'group01.continuation'
    if not os.path.exists(CONFIG.out_dir):
            os.makedirs(CONFIG.out_dir)
    with open('{}/{}'.format(CONFIG.out_dir, filename), 'w') as f:
        for s in continued_sentences:
            f.write('{}\n'.format(s))
    print('Using {} :'.format(save_dir))
    print('{} continued sentences written to {}/{}'.format(len(continuation_sentences), CONFIG.out_dir, filename))


# ### Pulling everything together

# In[18]:


### Finally, set up and run all tasks
def run():    
    # Define all experiments
    exp_dict = {}
    exp_parameters = [ ('A', [('epochs', 2), ('factor', 1), ('emb', False)])]#, # Experiment A
                      #('B', [('epochs', 2), ('factor', 1), ('emb', True )])]#, # Experiment B
                      #('C', [('epochs', 2), ('factor', 2), ('emb', True )])]   # Experiment C
    for label, para_list in exp_parameters:
        exp_dict.setdefault(label, dict(para_list))
    
    # Define source experiment for conditional generation
    sample_source = 'C'

    # Setup up CONFIGs and run experiments
    for exp in exp_dict:
        CONFIG = RNNConfig(num_epochs=exp_dict[exp]['epochs'],
                           rnn_size_factor=exp_dict[exp]['factor'],
                           external_embedding=exp_dict[exp]['emb'])
        train(exp, CONFIG)
        perp(exp, CONFIG)
        if (exp == sample_source):
            sample(exp, CONFIG)
            perp(exp, CONFIG, test=True)


# In[ ]:


# Ready, set, ...
run()

