# GTCRN-ONNX-RealTime-Inference-Sample
[Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)のリアルタイム推論（チャンク毎処理）のサンプルです。

# Requirement
```
sounddevice 0.5.1 or later
soundfile   0.13.1 or later
onnx        1.17.0 or later
onnxruntime 1.17.0 or later
```

# Usage（Wav Sample）
Wavファイルでの動作デモは、以下のColaboratoryノートブックを参照ください。<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/GTCRN-ONNX-RealTime-Inference-Sample/blob/main/colab-wav-sample.ipynb)<br>
[Open In Colab]リンクからノートブックを開き、順に実行してください。

# Usage（Mic Sample）
マイクでの動作デモは、以下のPythonスクリプトを実行ください。
```
python mic-sample.py
```
マイクはデフォルトマイクで動作します。<br>
プログラム終了は「Ctrl＋C」です。停止までの録音ファイルが「output.wav」として保存されます。

# Reference
* [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)
* [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
 GTCRN-ONNX-RealTime-Inference-Sample is under [MIT License](LICENSE).
