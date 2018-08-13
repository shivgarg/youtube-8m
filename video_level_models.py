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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils
import model_utils
import losses
from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_string("googlenet_pooling","max","Googlenet pooling method")
flags.DEFINE_string("residualcnn_pooling","max","ResidualCNN pooling method")
flags.DEFINE_integer("residualcnn_x",320,"X hyperparameter of residualcnn")
class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, scope='default', **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    with tf.variable_scope(scope,tf.AUTO_REUSE):
        output = slim.fully_connected(
            model_input, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   scope='default',
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
    return {"predictions": final_probabilities}

class AutoEncoderModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   labels = None,
                   scope='default',
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):

      reshaped_input=tf.expand_dims(model_input,-1)
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
      
      encoded = tf.reduce_max(out3_pool,axis=[2,3])
      
      decode = tf.expand_dims(encoded,-1)
      decode = tf.expand_dims(decode,-1)
      decode1 = tf.layers.conv2d(decode,64, (4,1), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      decode1_batch = tf.layers.batch_normalization(decode1,training=is_training)
      decode1_upsample = tf.layers.conv2d_transpose(decode1_batch,256,(8,1),strides=(2,1),padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
      
      decode2 = tf.layers.conv2d(decode1_upsample,64,1,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
      decode2 = tf.layers.conv2d(decode2,64, (4,1), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      decode2_batch = tf.layers.batch_normalization(decode2,training=is_training)
      decode2_upsample = tf.layers.conv2d_transpose(decode2_batch,256,(8,1),strides=(2,1),padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
      
      decode3 = tf.layers.conv2d(decode2_upsample,64,1,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
      decode3 = tf.layers.conv2d(decode3,64, (4,1), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='same')
      decode3_batch = tf.layers.batch_normalization(decode3,training=is_training)
      decode3_upsample = tf.layers.conv2d_transpose(decode3_batch,256,(8,1),strides=(2,1),padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
      
      decoded = tf.reduce_max(decode3_upsample,axis=[2,3])

      results = {}
      results['loss'] = 500*tf.losses.mean_squared_error(model_input,decoded)

      output = MoeModel().create_model(encoded,vocab_size)
      results['predictions'] = output['predictions']
      if labels is not None:
        results['loss'] += losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)

      return results

class AutoEncoderModelFC(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   labels,
                   scope='default',
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
      
      out1 = slim.fully_connected(model_input,768,weights_regularizer=slim.l2_regularizer(l2_penalty))
      out1_drop  = slim.dropout(out1,is_training=is_training)

      out2 = slim.fully_connected(out1_drop,512,weights_regularizer=slim.l2_regularizer(l2_penalty))
      out2_drop  = slim.dropout(out2,is_training=is_training)

      encoded = slim.fully_connected(out2_drop,384,weights_regularizer=slim.l2_regularizer(l2_penalty))
      
      out4 = slim.fully_connected(encoded,640,weights_regularizer=slim.l2_regularizer(l2_penalty))
      out4_drop  = slim.dropout(out4,is_training=is_training)
    
      out5 = slim.fully_connected(out4_drop,768,weights_regularizer=slim.l2_regularizer(l2_penalty))
      decoded = slim.fully_connected(out5,int(model_input.shape[1]),weights_regularizer=slim.l2_regularizer(l2_penalty))

      results = {}
      encoder_loss = 500*tf.losses.mean_squared_error(model_input,decoded)
      tf.summary.scalar("encoder_loss",encoder_loss)
      results['loss'] = encoder_loss

      output = MoeModel().create_model(encoded,vocab_size,scope="final_layer")
      output1 = MoeModel().create_model(out4,vocab_size,scope="intermediate_layer")
    
      results['predictions'] = (output['predictions'] + output1['predictions'])/2
      if labels is not None:
        prediction_loss = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
        tf.summary.scalar("prediction_loss",prediction_loss)
        results['loss'] += prediction_loss

      return results

class GoogleNetModel(models.BaseModel):

    def inception_module(self,inp,param,scope):

        with tf.variable_scope(scope,tf.AUTO_REUSE):
            # 1x1 
            out1 = slim.convolution(inp,param[0],1,1)

            # 3x3
            out2 = slim.convolution(inp,param[1],1,1)
            out2 = slim.convolution(out2,param[2],(9,1),1)

            # 5x5
            out3 = slim.convolution(inp,param[3],1,1)
            out3 = slim.convolution(out3,param[4],(25,1),1)

            # pool
            out4 = slim.max_pool2d(inp,(9,1),1,padding='SAME')
            out4 = slim.convolution(out4,param[5],1,1)

            output = tf.concat([out1,out2,out3,out4],3)

            return output


    def create_model(self, model_input, vocab_size, labels, scope='default', is_training=True, **unused_params):
        
        with tf.variable_scope(scope,tf.AUTO_REUSE):
            reshaped_input = tf.expand_dims(model_input,-1)
            reshaped_input = tf.expand_dims(reshaped_input,-1)
            
            conv1 = slim.convolution(reshaped_input,64,[49,1],stride=(4,1))
            max_pool1 = slim.max_pool2d(conv1,(9,1),(2,1),padding='SAME')
            norm1 = tf.nn.local_response_normalization(max_pool1)

            conv2 = slim.convolution(norm1,64,1,1)
            conv3 = slim.convolution(conv2,192,(9,1),1)
            norm2 = tf.nn.local_response_normalization(conv3)
            max_pool2 = slim.max_pool2d(norm2,(9,1),(2,1),padding='SAME')

            inception3a = self.inception_module(max_pool2,[64,96,128,16,32,32],'3a')
            inception3b = self.inception_module(inception3a,[128,128,192,32,96,64],'3b')

            max_pool3 = slim.max_pool2d(inception3b,(9,1),(2,1),padding='SAME')

            inception4a = self.inception_module(max_pool3,[192,96,208,16,48,64],'4a')
            inception4b = self.inception_module(inception4a,[160,112,224,24,64,64],'4b')
            inception4c = self.inception_module(inception4b,[128,128,256,24,64,64],'4c')
            inception4d = self.inception_module(inception4c,[112,144,288,32,64,64],'4d')
            inception4e = self.inception_module(inception4d,[256,160,320,32,128,128],'4e')

            max_pool4 = slim.max_pool2d(inception4e,(9,1),(2,1),padding='SAME')

            inception5a = self.inception_module(max_pool4,[256,160,320,32,128,128],'5a')
            inception5b = self.inception_module(inception5a,[384,192,384,48,128,128],'5b')

            inter1 = tf.squeeze(inception4a,axis=[2])
            inter2 = tf.squeeze(inception4d,axis=[2])
            output = tf.squeeze(inception5b,axis=[2])
            inter1 = model_utils.FramePooling(inter1,FLAGS.googlenet_pooling)
            inter2 = model_utils.FramePooling(inter2,FLAGS.googlenet_pooling)
            output = model_utils.FramePooling(output,FLAGS.googlenet_pooling)

            inter_results1 = MoeModel().create_model(inter1,vocab_size,'inter1')
            inter_results2 = MoeModel().create_model(inter2,vocab_size,'inter2')
            results = MoeModel().create_model(output,vocab_size,'final')
            results['features'] = output
            if labels != None:
                results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
                results['loss'] += losses.CrossEntropyLoss().calculate_loss(inter_results1['predictions'],labels)
                results['loss'] += losses.CrossEntropyLoss().calculate_loss(inter_results2['predictions'],labels)

            return results


class ResidualCNN(models.BaseModel):

    def residual_module(self,params,inp,scope='default'):

        with tf.variable_scope(scope,tf.AUTO_REUSE):
            depth = len(params)
            out = inp
            for i in range(depth):
                out = slim.convolution(out,params[i],(9,1),rate=(2*i+1,1))
            return inp+out

    def create_model(self, model_input, vocab_size, labels, scope='default', is_training=True, ** unused_params):
        X = FLAGS.residualcnn_x
        with tf.variable_scope(scope,tf.AUTO_REUSE):
            fc = slim.fully_connected(model_input,X,weights_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            reshaped_input = tf.expand_dims(fc,-1)
            reshaped_input = tf.expand_dims(reshaped_input,-1)

            conv1 = slim.convolution(reshaped_input,64,[49,1])
            conv1_norm = slim.batch_norm(conv1,is_training=is_training)
            
            module1 = self.residual_module([128,192,64],conv1_norm,'module1')
            module1_norm = slim.batch_norm(module1,is_training=is_training)

            conv2 = slim.convolution(module1_norm,128,1)
            conv2_norm = slim.batch_norm(conv2,is_training=is_training)

            module2 = self.residual_module([256,512,128],conv2_norm,'module2')
            module2_norm = slim.batch_norm(module2,is_training=is_training)

            conv3 = slim.convolution(module2_norm,256,1)
            conv3_norm = slim.batch_norm(conv3,is_training=is_training)

            module3 = self.residual_module([512,256],conv3_norm,'module3')
            module3_norm = slim.batch_norm(module3,is_training=is_training)

            conv4 = slim.convolution(module3_norm,X,1)
            conv4_norm = slim.batch_norm(conv4,is_training=is_training)

            module4 = self.residual_module([512,X],conv4_norm,'module4')
            
            features = tf.squeeze(module4,[2])
            features = model_utils.FramePooling(features,FLAGS.residualcnn_pooling) + fc
            results = MoeModel().create_model(features,vocab_size)
            results['features'] = features
            if labels != None:
                results['loss'] = losses.CrossEntropyLoss().calculate_loss(results['predictions'],labels)
            return results