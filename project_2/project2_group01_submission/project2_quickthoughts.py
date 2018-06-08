
# coding: utf-8

# # Task description

# In[ ]:


# from IPython.display import IFrame
# IFrame('project2.pdf', width=600, height=1200)


# # Preparation

# To use this notebook, make sure you have placed the following into a ```./data/``` subdirectory:
# - Place all cloze task files for training and validation as in the polybox directory into ```./data/cloze/```
# - Download and place Stanford's GloVe 6B vector set [glove.6B](http://nlp.stanford.edu/data/glove.6B.zip) as in the polybox directory into ```./data/glove/```

# # Import statements

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd

import time
from collections import OrderedDict


# # Config variables

# In[ ]:


N_SENT_IN_STORY = 5
ENCODER_DIM = 600
MAX_SEQ_LENGTH = 30
MAX_VOCAB_SIZE = 400000
EMBEDDING_DIM = 300
ENCODER_TYPE = 'GRU'
OOV_TOKEN = '<unk>'
CONTEXT_SIZE = 1

LEARNING_RATE = 0.001
CLIP_GRAD_NORM = 5.0

EPOCHS = 5
BATCH_SIZE = 100
DROPOUT_RATE = 0.25
SAVE_DIR = './checkpoints/'
SUMMARIES_DIR = './summaries/'
MAX_TO_KEEP = 1
DISPLAY_STEP = 100
VALIDATE_STEP = 20
SAVE_STEP = 5000


# # Functions

# In[ ]:


def get_word2vec(word2vec_file='./data/glove.6B/glove.6B.{}d.txt'.format(EMBEDDING_DIM)):
    word2vec = {}
#     with open(os.path.join(word2vec_file), encoding='utf8') as f:
    with open(word2vec_file, encoding='utf8') as f:
        for row in f:
            values = row.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    return word2vec


# In[ ]:


def get_training_data(train_file='./data/cloze/train_stories.csv'):
    train = pd.read_csv(train_file)
    train.drop_duplicates(subset='storyid') # make sure we have no duplicates
    titles = np.expand_dims(train['storytitle'].values, axis=1)
    sentences_1 = np.expand_dims(train['sentence1'].values, axis=1)
    sentences_2 = np.expand_dims(train['sentence2'].values, axis=1)
    sentences_3 = np.expand_dims(train['sentence3'].values, axis=1)
    sentences_4 = np.expand_dims(train['sentence4'].values, axis=1)
    sentences_5 = np.expand_dims(train['sentence5'].values, axis=1)
    mains = np.column_stack((sentences_1, sentences_2, sentences_3, sentences_4))
    stories = np.hstack((mains, sentences_5))
    sentences = [s for story in stories for s in story]
    print('{} has {} stories with a total of {} sentences.'.format(train_file,
                                                                   len(stories),
                                                                   len(sentences)))
    return sentences


# In[ ]:


def get_val_data(val_file='./data/cloze/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'):
    validation = pd.read_csv(val_file)
    sentences_4 = np.expand_dims(validation['InputSentence4'].values, axis=1)
    quiz_1 = np.expand_dims(validation['RandomFifthSentenceQuiz1'].values, axis=1)
    quiz_2 = np.expand_dims(validation['RandomFifthSentenceQuiz2'].values, axis=1)
    answers = np.expand_dims(validation['AnswerRightEnding'].values, axis=1)
    quizzes = np.hstack((sentences_4, quiz_1, quiz_2))
    sentences = [s for quiz in quizzes for s in quiz]
    print('{} has {} quizzes with a total of {} sentences.'.format(val_file,
                                                                   len(quizzes),
                                                                   len(sentences)))
    return sentences, answers


# In[ ]:


def get_test_data(test_file='./data/cloze/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'):
    test = pd.read_csv(test_file)
    sentences_4 = np.expand_dims(test['InputSentence4'].values, axis=1)
    quiz_1 = np.expand_dims(test['RandomFifthSentenceQuiz1'].values, axis=1)
    quiz_2 = np.expand_dims(test['RandomFifthSentenceQuiz2'].values, axis=1)
    answers = np.expand_dims(test['AnswerRightEnding'].values, axis=1)
    quizzes = np.hstack((sentences_4, quiz_1, quiz_2))
    sentences = [s for quiz in quizzes for s in quiz]
    print('{} has {} quizzes with a total of {} sentences.'.format(test_file,
                                                                   len(quizzes),
                                                                   len(sentences)))
    return sentences, answers


# In[ ]:


# This is an ugly-but-necessary hack thx to the NLU test data being of a different format than all the other data files
def get_NLU_test_data(test_file='./data/cloze/test_nlu18.csv'):
    test = pd.read_csv(test_file, header=None, encoding='latin-1')
    sentences_4 = np.expand_dims(test[3].values, axis=1)
    quiz_1 = np.expand_dims(test[4].values, axis=1)
    quiz_2 = np.expand_dims(test[5].values, axis=1)
    quizzes = np.hstack((sentences_4, quiz_1, quiz_2))
    sentences = [s for quiz in quizzes for s in quiz]
    print('{} has {} quizzes with a total of {} sentences.'.format(test_file,
                                                                   len(quizzes),
                                                                   len(sentences)))
    return sentences


# In[ ]:


def text_to_word_sequence(text):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    split = ' '
    text = text.lower().translate({ord(c): split for c in filters})
    seq = text.split(split)
    return [t for t in seq if t]


# In[ ]:


def generate_vocabs(texts):
    word_counts = OrderedDict()
    max_seq_len = 0
    for text in texts:
        if isinstance(text, list):
            seq = text
        else:
            seq = text_to_word_sequence(text)
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
        for w in seq:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1
    word_counts = list(word_counts.items())
    word_counts.sort(key = lambda x: x[1], reverse=True)
    sorted_vocab = [word_count[0] for word_count in word_counts]
    word2idx = dict(list(zip(sorted_vocab, list(range(1, len(sorted_vocab) + 1)))))
    i = word2idx.get(OOV_TOKEN)
    if i is None:
        word2idx[OOV_TOKEN] = len(word2idx) + 1
    idx2word = {value : key for key, value in word2idx.items()}
    return word2idx, idx2word, max_seq_len


# In[ ]:


def tokenize_pad_mask(texts, seq_length):
    vectors, masks = [], []
    for text in texts:
        if isinstance(text, list):
            seq = text
        else:
            seq = text_to_word_sequence(text)
        seq = seq[:seq_length]
        vector, mask  = [], []
        for w in seq:
            vector.append(word2idx.get(w, word2idx[OOV_TOKEN]))
            mask.append(1)
        while len(vector) < seq_length:
            vector.append(0)
            mask.append(0)
        vectors.append(vector)
        masks.append(mask)
    return np.array(vectors, dtype='int64'), np.array(masks, dtype='int8')


# In[ ]:


def get_inputs():
    encoder_inputs = tf.placeholder(tf.int64, [None, None], name='encoder_inputs')
    encoder_input_masks = tf.placeholder(tf.int8, [None, None], name='input_masks')
    encoder_targets = tf.placeholder(tf.float32, [None, None], name='encoder_targets')
    label_weights = tf.placeholder(tf.float32, [None,], name='label_weights')
    dropout_rate = tf.placeholder(tf.float32, [], name='dropout_rate')
    return encoder_inputs, encoder_input_masks, encoder_targets, label_weights, dropout_rate


# In[ ]:


def get_embedding_matrix(num_words):
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, idx in word2idx.items():
        if idx < num_words:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
    return embedding_matrix


# In[ ]:


def get_embeddings(encode_ids, trainable=False):
    word_embeddings = []
    encode_emb = []
    for suffix in ['_f', '_g']:
        word_emb = tf.get_variable(name='word_embedding'+suffix,
                                   shape=embedding_matrix.shape,
                                   trainable=trainable)
        word_emb.assign(embedding_matrix)
        word_embeddings.append(word_emb)
        encode_ = tf.nn.embedding_lookup(word_emb, encode_ids)
        encode_emb.append(encode_)
    return word_embeddings, encode_emb


# In[ ]:


def make_rnn_cells(num_units, cell_type):
    if cell_type == 'GRU':
        return tf.nn.rnn_cell.GRUCell(num_units=num_units)
    elif cell_type == 'LSTM':
        return tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    else:
        raise ValueError('Invalid cell type given')    


# In[ ]:


def rnn_encoder(embeds, mask, scope, num_units=600, cell_type='GRU'):
    sequence_length = tf.to_int32(tf.reduce_sum(mask, 1), name='length')
    cell_fw = make_rnn_cells(num_units, cell_type)
    cell_bw = make_rnn_cells(num_units, cell_type)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                      cell_bw=cell_bw,
                                                      inputs=embeds,
                                                      sequence_length=sequence_length,
                                                      dtype=tf.float32,
                                                      scope=scope)
    if cell_type == 'LSTM':
        states = [states[0][1], states[1][1]]
    state = tf.concat(states, 1)
    return state


# In[ ]:


def bow_encoder(embeds, mask):
    mask_expand = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    embeds_masked = embeds * mask_expand
    return tf.reduce_sum(embeds_masked, axis=1)


# In[ ]:


def get_thought_vectors(encode_emb, encode_mask):
    suffixes = ['_f', '_g']
    thought_vectors = []
    for i in range(len(suffixes)):
        with tf.variable_scope('encoder' + suffixes[i]) as scope:
            if ENCODER_TYPE == 'GRU':
                encoded = rnn_encoder(encode_emb[i], encode_mask, scope,
                                     ENCODER_DIM, ENCODER_TYPE)
            elif ENCODER_TYPE == 'LSTM':
                encoded = rnn_encoder(encode_emb[i], encode_mask, scope,
                                     ENCODER_DIM, ENCODER_TYPE)
            elif ENCODER_TYPE == 'bow':
                encoded = bow_encoder(encode_emb[i], encode_mask)
            else:
                raise ValueError('Invalid encoder type given')

        thought_vector = tf.identity(encoded, name='thought_vector' + suffixes[i])
        thought_vectors.append(thought_vector)
    return thought_vectors


# In[ ]:


def get_targets_weights(batch_size, is_cloze, n_sent_in_story, is_quiz=False, quiz_answer=None):
    if is_quiz:
        assert quiz_answer in [1, 2], 'must indicate correct quiz answer'
        targets = np.zeros((3, 3), dtype='float32')
        targets[0, quiz_answer] = 1
        targets[quiz_answer, 0] = 1
        weights = np.array([1,0])
    else:
        context_idx = list(range(-CONTEXT_SIZE, CONTEXT_SIZE + 1))
        context_idx.remove(0)
        weights = np.ones(batch_size - 1)
        if is_cloze:
            sub_targets = np.zeros((n_sent_in_story, n_sent_in_story), dtype='float32')    
            for i in context_idx:
                sub_targets += np.eye(n_sent_in_story, k=i)
            targets = np.zeros((batch_size, batch_size), dtype='float32')
            weights = np.ones(batch_size - 1)
            for i in range(n_sent_in_story - 1, len(weights), n_sent_in_story):
                weights[i] = 0
            for i in range(0, batch_size, n_sent_in_story):
                targets[i:i+n_sent_in_story, i:i+n_sent_in_story] += sub_targets
        else:
            targets = np.zeros((batch_size, batch_size), dtype='float32')
            for i in context_idx:
                targets += np.eye(batch_size, k=i)
        targets /= np.sum(targets, axis=1, keepdims=True)
    return targets, weights


# In[ ]:


def get_scores(thought_vectors, dropout_rate):
    def use_dropout():
        a, b = thought_vectors[0], thought_vectors[1]
        dropout_mask_shape = tf.transpose(tf.shape(a))
        dropout_mask = tf.random_uniform(dropout_mask_shape) > DROPOUT_RATE
        dropout_mask = tf.where(dropout_mask,
                                tf.ones(dropout_mask_shape),
                                tf.zeros(dropout_mask_shape))
        dropout_mask *= (1/dropout_rate)
        a *= dropout_mask
        b *= dropout_mask
        return a, b
    def no_dropout():
        return thought_vectors[0], thought_vectors[1]
    a, b = tf.cond(dropout_rate > 0, use_dropout, no_dropout)

    scores = tf.matmul(a, b, transpose_b=True)
    scores = tf.matrix_set_diag(scores, tf.zeros_like(scores[0]))
    return scores


# In[ ]:


def get_labels_predictions(scores, n_sent_in_story=N_SENT_IN_STORY, is_cloze=True):
    bwd_scores = scores[1:  ]
    fwd_scores = scores[ :-1]
    bwd_predictions = tf.to_int64(tf.argmax(bwd_scores, axis=1))
    fwd_predictions = tf.to_int64(tf.argmax(fwd_scores, axis=1))
    bwd_labels = tf.range(tf.shape(bwd_scores)[0])
    fwd_labels = bwd_labels + 1
    
    return (bwd_labels, fwd_labels), (bwd_predictions, fwd_predictions)#, label_weights


# In[ ]:


def get_batch_acc(labels, predictions, label_weights):
    total_weight = tf.reduce_sum(label_weights)
    bwd_acc = tf.cast(tf.equal(tf.to_int64(labels[0]) , predictions[0]), tf.float32)
    bwd_acc *= label_weights
    bwd_acc = tf.reduce_sum(bwd_acc)
    bwd_acc /= total_weight
    fwd_acc = tf.cast(tf.equal(tf.to_int64(labels[1]), predictions[1]), tf.float32)
    fwd_acc *= label_weights
    fwd_acc = tf.reduce_sum(fwd_acc)
    fwd_acc /= total_weight
    return bwd_acc, fwd_acc    


# In[ ]:


def get_batches(inputs, masks, batch_size, is_cloze=True, n_sent_in_story=5, is_quiz=False, quiz_answers=None, shuffle=True):
    if is_cloze:
        assert (batch_size % n_sent_in_story) == 0, 'batch_size must be multiple of n_sent_in_story for cloze task training.'
    rows, cols = inputs.shape
    if shuffle and is_cloze and not is_quiz:        
        row_blocks = rows // n_sent_in_story
        shuffle_idx = np.random.permutation(row_blocks) 
        inputs = inputs.reshape((row_blocks, -1, cols))[shuffle_idx].reshape((-1, cols))
        masks = masks.reshape((row_blocks, -1, cols))[shuffle_idx].reshape((-1, cols))
    n_batches = len(inputs) // batch_size
    for batch_i in range(n_batches):
        start_i = batch_i * batch_size
        batch_inputs = inputs[start_i : start_i + batch_size]
        batch_masks = masks[start_i : start_i + batch_size]
        if is_quiz:
            batch_targets, batch_weights = get_targets_weights(batch_size, is_cloze, n_sent_in_story,
                                                               is_quiz, quiz_answers[batch_i])
        else:
            batch_targets, batch_weights = get_targets_weights(batch_size, is_cloze, n_sent_in_story)
        yield batch_inputs, batch_masks, batch_targets, batch_weights


# # Setup actions

# In[ ]:


print('Loading cloze story training data...')
sentences = get_training_data()
print('Loaded training stories.')


# In[ ]:


print('Loading cloze story validation data...')
validation_sentences, validation_answers = get_val_data()
print('Loaded validation stories.')


# In[ ]:


print('Loading cloze story test data...')
test_sentences, test_answers = get_test_data()
print('Loaded test stories.')


# In[ ]:


print('Loading NLU 2018 story test data...')
NLU_test_sentences = get_NLU_test_data()
print('Loaded NLU 2018 test stories.')


# In[ ]:


print('Generating vocabulary from training data ...')
word2idx, idx2word, max_seq_len = generate_vocabs(sentences)
print('Found {} unique word tokens\n       Longest sentence has {} tokens.'.format(len(word2idx), max_seq_len))


# In[ ]:


print('Loading pretrained word embedding vectors...')
word2vec = get_word2vec()
print('Loaded {} word vectors.'.format(len(word2vec)))


# In[ ]:


num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

print('Constructing embedding matrix...')
embedding_matrix = get_embedding_matrix(num_words)
print('Finished embedding matrix has shape {}.'.format(embedding_matrix.shape))


# In[ ]:


seq_length = min(max_seq_len, MAX_SEQ_LENGTH)

print('Word2idx, pad and mask training sentences to length {} ...'.format(seq_length))
enc_sentences, enc_masks = tokenize_pad_mask(sentences, seq_length)
print('{} training sentences processed.'.format(len(enc_sentences)))


# In[ ]:


print('Word2idx, pad and mask validation sentences to length {} ...'.format(seq_length))
validation_inputs, validation_masks = tokenize_pad_mask(validation_sentences, seq_length)
print('{} validation sentences sentences processed.'.format(len(validation_inputs)))


# In[ ]:


print('Word2idx, pad and mask test sentences to length {} ...'.format(seq_length))
test_inputs, test_masks = tokenize_pad_mask(test_sentences, seq_length)
print('{} test sentences sentences processed.'.format(len(test_inputs)))


# In[ ]:


print('Word2idx, pad and mask NLU 2018 test sentences to length {} ...'.format(seq_length))
NLU_test_inputs, NLU_test_masks = tokenize_pad_mask(NLU_test_sentences, seq_length)
print('{} NLU 2018 test sentences sentences processed.'.format(len(NLU_test_inputs)))


# # Build graph

# In[ ]:


print('Building graph...')
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
   
    with tf.name_scope('input_data'):
        encoder_inputs, encoder_input_masks, encoder_targets, label_weights, dropout_rate = get_inputs()
        
    with tf.name_scope('embeddings'):
        word_embeddings, encode_emb = get_embeddings(encoder_inputs, trainable=True)
        
    with tf.name_scope('encoders'):
        thoughts = get_thought_vectors(encode_emb, encoder_input_masks)
 
    with tf.name_scope('losses_accuracies'):
        scores = get_scores(thoughts, dropout_rate)
        labels, predictions = get_labels_predictions(scores)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=encoder_targets,
                                                       logits=scores))
        tf.summary.scalar('batch_ent_loss', loss)
        
        bwd_acc, fwd_acc = get_batch_acc(labels, predictions, label_weights)
        tf.summary.scalar('batch_bwd_accuracy', bwd_acc)
        tf.summary.scalar('batch_fwd_accuracy', fwd_acc)
        
        _, stream_bwd_acc = tf.metrics.accuracy(labels[0], predictions[0], weights=label_weights)
        _, stream_fwd_acc = tf.metrics.accuracy(labels[1], predictions[1], weights=label_weights)
        tf.summary.scalar('stream_bwd_accuracy', stream_bwd_acc)
        tf.summary.scalar('stream_fwd_accuracy', stream_fwd_acc)
        
    with tf.name_scope('optimization'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), CLIP_GRAD_NORM)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    merged = tf.summary.merge_all()
print('Graph assembled.')


# # Run training

# In[ ]:


### BEGIN TRAINING SECTION - comment out if you want to use the latest trained model in SAVE_DIR 
#print('Starting training...')
#print('Run "tensorboard --logdir {}" in the current directory to keep you from doing other work in the meantime.'.format(SUMMARIES_DIR))
#start_time = time.strftime('%y-%m-%d-%H-%M-%S')
#with tf.Session(graph=train_graph) as sess:
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
#    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
#    train_writer = tf.summary.FileWriter('{}run-{}/{}'.format(SUMMARIES_DIR, start_time, 'train'), sess.graph)
#    valid_writer = tf.summary.FileWriter('{}run-{}/{}'.format(SUMMARIES_DIR, start_time, 'valid'), sess.graph)
#    step = 0
#    for e in range(EPOCHS):
#        valid_batch =  get_batches(validation_inputs, validation_masks, batch_size=3, n_sent_in_story=3,
#                                   is_quiz=True, quiz_answers=answers, shuffle=False)
#        for batch_i, (batch_inputs, batch_masks, batch_targets, batch_weights) in         enumerate(get_batches(enc_sentences, enc_masks, batch_size=BATCH_SIZE, n_sent_in_story=5, shuffle=True)):
#            
#            feed_dict = {encoder_inputs: batch_inputs,
#                         encoder_input_masks: batch_masks,
#                         encoder_targets: batch_targets,
#                         label_weights: batch_weights,
#                         dropout_rate: DROPOUT_RATE}
#              
#            _, batch_loss, bwd_accuracy, fwd_accuracy, summary = sess.run([train_op,
#                                                                           loss,
#                                                                           bwd_acc,
#                                                                           fwd_acc,
#                                                                           merged],
#                                                                           feed_dict=feed_dict)
#            train_writer.add_summary(summary, step)
#            
#            if batch_i % DISPLAY_STEP == 0 and batch_i > 0:
#                print('Epoch {:>3} Batch {:>4}/{} - Batch bwd acc: {:>3.2%}, Batch fwd acc: {:>3.2%}, Batch loss: {:>6.4f}'
#                      .format(e, batch_i, len(enc_sentences) // BATCH_SIZE, bwd_accuracy, fwd_accuracy, batch_loss))
#                
#            if batch_i % VALIDATE_STEP == 0 and batch_i > 0:
#                valid_input, valid_mask, valid_target, valid_weight = next(valid_batch)
#                feed_dict = {encoder_inputs: valid_input,
#                             encoder_input_masks: valid_mask,
#                             encoder_targets: valid_target,
#                             label_weights: valid_weight,
#                             dropout_rate: 0}
#                
#                valid_loss, stream_bwd_accuracy, stream_fwd_accuracy, summary = sess.run([loss,
#                                                                                             stream_bwd_acc,
#                                                                                             stream_fwd_acc,
#                                                                                             merged],
#                                                                                             feed_dict=feed_dict)
#                
#                valid_writer.add_summary(summary, step)
#            
#            if batch_i % SAVE_STEP == 0 and batch_i > 0:
#                saver.save(sess, '{}/run-{}_ep_{}_step_{}_enc_{}_bsize_{}.ckpt'.format(
#                    SAVE_DIR, start_time, e, step, ENCODER_TYPE, BATCH_SIZE))
#            
#            step += 1
#            
#        saver.save(sess, '{}/run-{}_ep_{}_step_{}_enc_{}_bsize_{}.ckpt'.format(
#            SAVE_DIR, start_time, e, step, ENCODER_TYPE, BATCH_SIZE))
#    train_writer.close()
#    valid_writer.close()
#print('Training finished.')
### END OF TRAINING SECTION


# # Get _cloze_ predictions

# In[ ]:


print('Checking for trained model at latest checkpoint...')
checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
assert checkpoint is not None, 'No checkpoints found, check SAVE_DIR & README.txt'
print('Found model {}.'.format(checkpoint))


# In[ ]:


print('Checking validation score...')
cloze_preds = []
with tf.Session(graph=train_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    for valid_i, (valid_input, valid_mask, valid_target, valid_weight)     in enumerate(get_batches(validation_inputs, validation_masks, batch_size=3, n_sent_in_story=3,
                             is_quiz=True, quiz_answers=validation_answers, shuffle=False)):
        scr = sess.run(scores,
                      {encoder_inputs: valid_input,
                       encoder_input_masks: valid_mask,
                       encoder_targets: valid_target,
                       label_weights: valid_weight,
                       dropout_rate: 0})
        cloze_pred = np.argmax(scr[0, 1:]) + 1
        cloze_preds.append(cloze_pred)
    cloze_preds = np.array(cloze_preds).reshape((-1, 1))
cloze_score = np.mean((validation_answers == cloze_preds))
print('For checkpoint {}:'.format(checkpoint))
print('Validation score: {:>3.2%}'.format(cloze_score))


# In[ ]:


# Only run at very end after training & when everything else is done
print('Checking test score...')
cloze_preds = []
with tf.Session(graph=train_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    for test_i, (test_input, test_mask, test_target, test_weight)     in enumerate(get_batches(test_inputs, test_masks, batch_size=3, n_sent_in_story=3,
                             is_quiz=True, quiz_answers=test_answers, shuffle=False)):
        scr = sess.run(scores,
                      {encoder_inputs: test_input,
                       encoder_input_masks: test_mask,
                       encoder_targets: test_target,
                       label_weights: test_weight,
                       dropout_rate: 0})
        cloze_pred = np.argmax(scr[0, 1:]) + 1
        cloze_preds.append(cloze_pred)
    cloze_preds = np.array(cloze_preds).reshape((-1, 1))
cloze_score = np.mean((test_answers == cloze_preds))
print('For checkpoint {}:'.format(checkpoint))
print('Test score: {:>3.2%}'.format(cloze_score))


# In[ ]:


# Only run at very end after training & when everything else is done
print('Generating NLU 2018 test predictions...')
cloze_preds = []
with tf.Session(graph=train_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    for test_i, (test_input, test_mask, test_target, test_weight)     in enumerate(get_batches(NLU_test_inputs, NLU_test_masks, batch_size=3, n_sent_in_story=3,
                             is_quiz=True, quiz_answers=np.ones((len(NLU_test_inputs), 1), dtype='int8'), shuffle=False)):
        scr = sess.run(scores,
                      {encoder_inputs: test_input,
                       encoder_input_masks: test_mask,
                       encoder_targets: test_target,
                       label_weights: test_weight,
                       dropout_rate: 0})
        cloze_pred = np.argmax(scr[0, 1:]) + 1
        cloze_preds.append(cloze_pred)
    cloze_preds = np.array(cloze_preds).reshape((-1, 1))
output_file = 'NLU_2018_test_preds.csv'
np.savetxt(output_file, cloze_preds, fmt='%1u', delimiter=',')
print('NLU 2018 test predictions written to {} - Bye!'.format(output_file))

