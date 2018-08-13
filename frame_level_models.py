# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim
from tensorflow import flags
import losses

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")

flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "attention",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")

flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")

flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 1, "Number of LSTM layers.")
flags.DEFINE_string("lstm_pooling_method","last","Pooling method for lstm layers")
flags.DEFINE_string("rgb_frame_level_model","DbofModel",
                    "Frame level model to process rgb features")
flags.DEFINE_string("audio_frame_level_model","DbofModel",
                    "Frame level model to porcess audio features")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    with tf.variable_scope(scope,tf.AUTO_REUSE):
      output = slim.fully_connected(
          avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(1e-8))

    return {"predictions": output,"features": avg_pooled,"loss":losses.CrossEntropyLoss().calculate_loss(output,labels)}

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    with tf.variable_scope(scope,tf.AUTO_REUSE):
      stacked_lstm = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0)
                  for _ in range(number_of_layers)
                  ])

      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32,
                                         swap_memory=True)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      if FLAGS.lstm_pooling_method == 'last':
        inp = state[-1].h
      else:
        inp = utils.FramePooling(outputs, FLAGS.lstm_pooling_method)
      results= aggregated_model().create_model(
          model_input=inp,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)
      results['features'] = inp
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results

class ResidualLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', is_training=True, **unused_params):

    with tf.variable_scope(scope,tf.AUTO_REUSE):
     
      with tf.variable_scope('lstm1',tf.AUTO_REUSE): 
       	lstm1 = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0)
                  ])

      	outputs1, _ = tf.nn.dynamic_rnn(lstm1, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32,
                                         swap_memory=True)
      with tf.variable_scope('lstm2',tf.AUTO_REUSE):
      	lstm2 = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0)
                  ])

      	outputs2, _ = tf.nn.dynamic_rnn(lstm2, outputs1,
                                         sequence_length=num_frames,
                                         dtype=tf.float32,
                                         swap_memory=True)
      with tf.variable_scope('lstm3',tf.AUTO_REUSE):
      	lstm3 = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0)
                  ])

      	outputs , state = tf.nn.dynamic_rnn(lstm3, outputs2+outputs1,
                                         sequence_length=num_frames,
                                         dtype=tf.float32,
                                         swap_memory=True)

      if FLAGS.lstm_pooling_method =='last':
        inp = state[-1].h
      else:
        inp = utils.FramePooling(outputs, FLAGS.lstm_pooling_method)
      
      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

      results= aggregated_model().create_model(
          model_input=inp,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)
      results['features'] = inp
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results


class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   labels,
                   scope='default',
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    with tf.variable_scope(scope,tf.AUTO_REUSE):
      num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        model_input = utils.SampleRandomFrames(model_input, num_frames,
                                               iterations)
      else:
        model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                 iterations)
      max_frames = model_input.get_shape().as_list()[1]
      feature_size = model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(model_input, [-1, feature_size])
      tf.summary.histogram("input_hist", reshaped_input)

      if add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn")

      cluster_weights = tf.get_variable("cluster_weights",
        [feature_size, cluster_size],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_weights", cluster_weights)
      activation = tf.matmul(reshaped_input, cluster_weights)
      if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn")
      else:
        cluster_biases = tf.get_variable("cluster_biases",
          [cluster_size],
          initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
        tf.summary.histogram("cluster_biases", cluster_biases)
        activation += cluster_biases
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("cluster_output", activation)

      activation = tf.reshape(activation, [-1, max_frames, cluster_size])
      activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

      hidden1_weights = tf.get_variable("hidden1_weights",
        [cluster_size, hidden1_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
      tf.summary.histogram("hidden1_weights", hidden1_weights)
      activation = tf.matmul(activation, hidden1_weights)
      if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn")
      else:
        hidden1_biases = tf.get_variable("hidden1_biases",
          [hidden1_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("hidden1_biases", hidden1_biases)
        activation += hidden1_biases
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("hidden1_output", activation)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      results = aggregated_model().create_model(
          model_input=activation,
          vocab_size=vocab_size,
          **unused_params)
      results['features'] = activation
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results

class DeeperDbofModel(models.BaseModel):
  
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   labels,
                   scope='default',
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    
    with tf.variable_scope(scope,tf.AUTO_REUSE):
      num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        model_input = utils.SampleRandomFrames(model_input, num_frames,
                                               iterations)
      else:
        model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                 iterations)
      max_frames = model_input.get_shape().as_list()[1]
      feature_size = model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(model_input, [-1, feature_size])

      tf.summary.histogram("input_hist", reshaped_input)
      reshaped_input=tf.expand_dims(reshaped_input,-1)
      reshaped_input=tf.expand_dims(reshaped_input,-1)

      out1 = tf.layers.conv2d(reshaped_input,128,(32,1),activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      out1_norm = tf.layers.batch_normalization(out1,training=is_training)
      out1_pool = tf.layers.max_pooling2d(out1_norm,(8,1),2,padding='same')

      out2 = tf.layers.conv2d(out1_pool,256,(32,1),activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      out2_norm = tf.layers.batch_normalization(out2,training=is_training)
      out2_pool = tf.layers.max_pooling2d(out2_norm,(8,1),2,padding='same')

      out3 = tf.layers.conv2d(out2_pool,256,(32,1),activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      out3_norm = tf.layers.batch_normalization(out3,training=is_training)
      out3_pool = tf.layers.max_pooling2d(out3_norm,(8,1),2,padding='same')

      out = tf.reduce_max(out3_pool,axis=[2,3])
      activation = tf.reshape(out, [-1, max_frames,out.shape[1]])
      cluster_size = out.shape[1]

      activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

      activation = tf.layers.dense(activation,hidden1_size,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

      tf.summary.histogram("activation", activation)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      results =  aggregated_model().create_model(
          model_input=activation,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)

      results['features'] = activation
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results

class BiLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', is_training=True, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    with tf.variable_scope(scope,tf.AUTO_REUSE):
      stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0) for _ in range(number_of_layers)])

      stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=1.0) for _ in range(number_of_layers)])

      outputs, state = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw,stacked_lstm_bw, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)
      
      if FLAGS.lstm_pooling_method == 'last':
        l = [state[i][-1].h for i in range(2)]
      else:
        l = [utils.FramePooling(outputs[0], FLAGS.lstm_pooling_method),
            utils.FramePooling(outputs[1], FLAGS.lstm_pooling_method)]
      
      output = tf.concat(l,1)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

      results = aggregated_model().create_model(
          model_input=output,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)
      results['features'] = output
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results

class SeparateModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', is_training=True,**unused_params):
    results={}
    with tf.variable_scope(scope,tf.AUTO_REUSE):
      rgb_input = tf.slice(model_input,[0,0,0],[-1,-1,1024])
      audio_input = tf.slice(model_input,[0,0,1024],[-1,-1,128])

      rgb_model = globals()[FLAGS.rgb_frame_level_model]
      audio_model = globals()[FLAGS.audio_frame_level_model]

      rgb_results = rgb_model().create_model(model_input=rgb_input,vocab_size=vocab_size,
                                             num_frames=num_frames,labels=labels,
                                             scope='rgb', is_training=is_training, **unused_params)

      audio_results = audio_model().create_model(model_input=audio_input,vocab_size=vocab_size,
                                                num_frames=num_frames,labels=labels,
                                                scope='audio',is_training=is_training**unused_params)
      if labels != None:
        results['loss'] = rgb_results['loss'] + audio_results['loss']
      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      features = rgb_results['features'] + audio_results['features']
      output = aggregated_model().create_model(
          model_input=features,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)

      if labels != None:
        results['loss'] += 6*losses.CrossEntropyLoss().calculate_loss(output['predictions'],labels)
      results['predictions'] = output['predictions']
    return results


class StackedBiLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', is_training=True, **unused_params):
    
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    
    for i in range(1,number_of_layers):
      with tf.variable_scope(scope,tf.AUTO_REUSE):
        with tf.variable_scope("lstm"+str(i),tf.AUTO_REUSE):
      	  lstm_fw1 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0) ])

      	  lstm_bw1 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=1.0) ])

      	  outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw1,lstm_bw1, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)
      	  model_input = tf.add(outputs[0],outputs[1])
      
    with tf.variable_scope("lstm"+str(number_of_layers),tf.AUTO_REUSE):
      	lstm_fw2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0) ])

      	lstm_bw2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=1.0) ])

      	outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw2,lstm_bw2, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)

    if FLAGS.lstm_pooling_method == 'last':
      l = [state[i][-1].h for i in range(2)]
    else:
      l = [utils.FramePooling(outputs[0], FLAGS.lstm_pooling_method),
           utils.FramePooling(outputs[1], FLAGS.lstm_pooling_method)]
      
    output = tf.concat(l,1)

    aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

    results = aggregated_model().create_model(
          model_input=output,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)
    results['features'] = output
    if labels != None:
      results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results



class TimeScaleLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', **unused_params):

    lstm_size = FLAGS.lstm_cells
    with tf.variable_scope(scope,tf.AUTO_REUSE):
      cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)])

      outputs1, _ = tf.nn.dynamic_rnn(cells, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True,scope='first')
      cells1 = tf.contrib.rnn.MultiRNNCell(
              [tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)])

      outputs2, state2 = tf.nn.dynamic_rnn(cells1, outputs1[:,0:300:2,:],
                                         sequence_length=num_frames/2,
                                         dtype=tf.float32, swap_memory=True,scope='second')
      
      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

      if FLAGS.lstm_pooling_method == 'last':
        output = state2[-1].h
      else:
        output = utils.FramePooling(outputs2, FLAGS.lstm_pooling_method)
      results = aggregated_model().create_model(
          model_input=output,
          vocab_size=vocab_size,
          **unused_params)
      results['features'] = output
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results

class ResidualStackedBiLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, scope='default', **unused_params):
    
    lstm_size = FLAGS.lstm_cells
    
    with tf.variable_scope(scope,tf.AUTO_REUSE):
      with tf.variable_scope("lstm1",tf.AUTO_REUSE):
      	lstm_fw1 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2, forget_bias=1.0) ])

      	lstm_bw1 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2,forget_bias=1.0) ])

      	outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw1,lstm_bw1, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)
      	outputs = tf.concat(outputs,2)
      with tf.variable_scope("lstm2",tf.AUTO_REUSE):
      	lstm_fw2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2, forget_bias=1.0) ])

      	lstm_bw2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2,forget_bias=1.0) ])

      	outputs1, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw2,lstm_bw2, outputs,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)
        outputs1 = tf.concat(outputs1,2)
      with tf.variable_scope("lstm3",tf.AUTO_REUSE):
      	lstm_fw3 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2, forget_bias=1.0) ])

      	lstm_bw3 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size/2,forget_bias=1.0) ])

      	outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw3,lstm_bw3, outputs+outputs1,
                                         sequence_length=num_frames,
                                         dtype=tf.float32, swap_memory=True)

      if FLAGS.lstm_pooling_method == 'last':
        l = [state[i][-1].h for i in range(2)]
      else:
        l = [utils.FramePooling(outputs[0], FLAGS.lstm_pooling_method),
             utils.FramePooling(outputs[1], FLAGS.lstm_pooling_method)]
     
        
      output = tf.concat(l,1)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

      results = aggregated_model().create_model(
          model_input=output,
          vocab_size=vocab_size,
          **unused_params)
      results['features'] = output
      if labels != None:
        results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
    return results
