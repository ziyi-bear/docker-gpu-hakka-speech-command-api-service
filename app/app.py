# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# 服務
from flask import Flask, flash, request, redirect, render_template
import json
import os

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

# 20系列顯示卡的問題解決 參考資料來源：https://github.com/tensorflow/tensorflow/issues/24496
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

FLAGS = None
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['wav', 'bmp', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

classCount = os.getenv('CLASS_COUNT', None)
sample_rate = os.getenv('SAMPLE_RATE', None)
soundlenght = os.getenv('SOUND_LENGTH', None)
step = os.getenv('STEP', None)

graph_path = '/graph.pb'
labels_path = '/labels.txt'
wav_path = './small.wav'

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session(config=config) as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    outcome = dict()

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      outcome[human_string] = score
      print('%s (score = %.5f)' % (human_string, score))

    return outcome


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  return run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

# 檢測接收到的資料格式
def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def run_wav_test():
  global classCount, sample_rate, soundlenght, step

  if request.method == 'POST':
    # 如果使用者沒有打算夾帶音訊檔案的話則不處理
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    # 使用者有打算夾帶音訊檔案
    file = request.files['file']
    # 但是實際上夾帶的內容是空的 if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    # 有夾帶內容(確實內容，但是需要驗證格式)
    if file and allowed_file(file.filename):
      print('使用者傳送的檔案名稱為{}'.format(file.filename))
      fix_file_name = 'small.wav'
      # 將檔案暫時儲存起來以供之後進行處理
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], fix_file_name))
      # 準備進行辨識工作
      outcomeDict = label_wav(wav_path, labels_path, graph_path, 'wav_data:0', 'labels_softmax:0', 3)
      return json.dumps(str(outcomeDict))
  return render_template('hakka-uplaod.html', classCount=int(classCount), sample_rate=int(sample_rate), soundlenght=int(soundlenght), step=int(step))

if __name__ == "__main__":
  # Only for debugging while developing
  app.run(host="0.0.0.0", port=80, debug=True)
