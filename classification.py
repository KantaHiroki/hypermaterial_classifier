import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np
from keras.models import load_model
import tensorflow as tf
import numpy as np
import json
import os
import io
import codecs


container = f"""
<div>
    <h1>Tsai-type icosahedral hyper material Classification</h1>
</div>
"""

classification_result = f"""
<div>
    <h2>Classification Results</h2>
</div>
"""

# モデルを読み込む
model = load_model('./text_classification_app/model/model.hdf5')


LABELS = ['iQC', '1/1AC', '2/1AC', 'Others']

def read_pattern_in_file(filename, lines):
    tth_list = []
    Intensity_list = []
    # try:
    #     lines = file.read().decode('utf-8', errors='ignore')
    # except:
    #     lines = file.read().decode('shift-jis', errors='ignore')
    for line in lines.split('\n'):
        if '*' in line:
            continue
        elif '#' in line:
            continue
        line_list = line[:-1].split()
        try:
            tth = float(line_list[0])
        except:
            continue
        if 20 <= tth < 80:
            tth_list.append(tth)
            Intensity = float(line_list[1])
            Intensity_list.append(Intensity)
    if len(Intensity_list)!=6000:
        tth_interval = tth_list[1]-tth_list[0]
        Intensity_list = data_compensate(tth_list, Intensity_list, tth_interval)
    # if len(Intensity_list)!=6000:
        # print("=============================")
        # print('Error! Insufficient data length, expect 6000')
        # print(filename)
        # print("=============================")
        # return 'error1'
    return Intensity_list

def data_compensate(tth_list, Intensity_list, tth_interval):
    tth_len = 6000
    tth_len_ = int((80.0-20.0)/tth_interval)
    step = int(tth_len/tth_len_)
    pattern_list = []
    for i in range(len(tth_list)):
        intensity1 = Intensity_list[i]
        pattern_list.append(intensity1)
        if i==len(tth_list)-1:
            intensity2 = Intensity_list[i]
        else:
            intensity2 = Intensity_list[i+1]
            intensity = (intensity1+intensity2)/step
        pattern_list.append(intensity)
    return pattern_list

def preprocess_pattern(Intensity_list):
    x_test = np.array([Intensity_list], np.float64)
    x_test = x_test-np.min(x_test, axis = 1).reshape(1, 1)
    x_test = x_test/np.max(x_test, axis = 1).reshape(1, 1)
    tf.keras.backend.set_floatx('float64')
    x_test_ = x_test[..., tf.newaxis]
    return x_test_


def predict(filename, file):
    # file = file.decode()
    pattern = read_pattern_in_file(filename, file)
    if pattern=='error1':
        results = 'Insufficient data length, expect 6000'
        return results, 'error'
    pattern = preprocess_pattern(pattern)
    prediction = model.predict(pattern, verbose=0)
    prediction = np.round(prediction, decimals=7).tolist()[0]

    # 最大確率のラベルを特定
    max_index = np.argmax(prediction)
    max_label = LABELS[max_index]

    results = {
        'iQC': prediction[0],
        'AC11': prediction[1],
        'AC21': prediction[2],
        'others': prediction[3]
    }

    return results, max_label


def on_button_click(change):
    output.clear_output()
    try:
        # アップロードされたファイルをすべて処理
        results_html = ""
        for filename, fileinfo in change['new'].items():
            # ファイルの内容を文字列としてデコード
            try:
                # ファイルの内容を文字列としてデコード（Shift-JISエンコーディング）
                content = fileinfo['content'].decode('shift_jis').strip()
            except UnicodeDecodeError:
                # 他のエンコーディングにフォールバック（ここではEUC-JPを例として追加）
                content = fileinfo['content'].decode('euc_jp').strip()

            # 予測を行う
            result, max_label  = predict(filename, content)
            iQC_pred, AC21_pred, AC11_pred, Others_pred = format(result['iQC'],'.5f'), format(result['AC21'],'.5f'), format(result['AC11'],'.5f'), format(result['others'],'.5f'), 

            # 結果をHTMLとして整形
            if max_label == 'iQC':
              results_html += f"""
              <div style="border:1px solid #ccc; padding: 0 2px; margin-bottom:5px; margin-right: auto; width: 60%; min-width: 500px; max-width: 600px;">
                  <h3 style="margin: 10px 0;">File: {filename}</h3>
                  <p style="font-size:20px; margin-left:20px">Prediction: <strong><span style="font-size:25px; color: #4682b4;">{max_label}</span></strong>.</p>
                  <p style="font-size:16px; margin-left:20px">iQC: {iQC_pred}, 2/1AC: {AC21_pred}, 1/1AC: {AC11_pred}, Others: {Others_pred}</p>
              </div>
              """
            elif max_label == '1/1AC':
              results_html += f"""
              <div style="border:1px solid #ccc; padding: 0 2px; margin-bottom:5px; margin-right: auto; width: 60%; min-width: 500px; max-width: 600px;">
                  <h3 style="margin: 10px 0;">File: {filename}</h3>
                  <p style="font-size:20px; margin-left:20px">Prediction: <strong><span style="font-size:25px; color: #ffa500;">{max_label}</span></strong>.</p>
                  <p style="font-size:16px; margin-left:20px">iQC: {iQC_pred}, 2/1AC: {AC21_pred}, 1/1AC: {AC11_pred}, Others: {Others_pred}</p>
              </div>
              """
            elif max_label == '2/1AC':
              results_html += f"""
              <div style="border:1px solid #ccc; padding: 0 2px; margin-bottom:5px; margin-right: auto; width: 60%; min-width: 500px; max-width: 600px;">
                  <h3 style="margin: 10px 0;">File: {filename}</h3>
                  <p style="font-size:20px; margin-left:20px">Prediction: <strong><span style="font-size:25px; color: #228b22;">{max_label}</span></strong>.</p>
                  <p style="font-size:16px; margin-left:20px">iQC: {iQC_pred}, 2/1AC: {AC21_pred}, 1/1AC: {AC11_pred}, Others: {Others_pred}</p>
              </div>
              """
            elif max_label == 'Others':
              results_html += f"""
              <div style="border:1px solid #ccc; padding: 0 2px; margin-bottom:5px; margin-right: auto; width: 60%; min-width: 500px; max-width: 600px;">
                  <h3 style="margin: 10px 0;">File: {filename}</h3>
                  <p style="font-size:20px; margin-left:20px">Prediction: <strong><span style="font-size:25px; color: #b22222;">{max_label}</span></strong>.</p>
                  <p style="font-size:16px; margin-left:20px">iQC: {iQC_pred}, 2/1AC: {AC21_pred}, 1/1AC: {AC11_pred}, Others: {Others_pred}</p>
              </div>
              """
            elif max_label == 'error':
              results_html += f"""
              <div style="border:1px solid #ccc; padding: 0 2px; margin-bottom:5px; margin-right: auto; width: 60%; min-width: 500px; max-width: 600px;">
                  <h3 style="margin: 10px 0;">File: {filename}</h3>
                  <p style="color: #ff0000; font-size:20px; margin-left:20px">Prediction: <strong>Error: </strong>{result}.</p>
              </div>
              """

        # 結果を表示
        with output:
            display(HTML(classification_result))
            display(HTML(results_html))
    except Exception as e:
        with output:
            print(f'Error: {e}')


display(HTML(container))

# ファイルアップロードウィジェットを作成
upload = widgets.FileUpload(
    accept='.txt, .ras',  # 受け付けるファイル形式
    multiple=True  # 複数ファイルのアップロードを許可しない
)

upload.observe(on_button_click, names='value')

# 結果表示用のウィジェットを作成
output = widgets.Output()

# ウィジェットを表示
display(upload, output)