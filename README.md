# bilstmcrf

运用BiLSTM + CRF 实现对中文进行分词

## Architecture

模型由三部分组成如下：

* Embedding: words embedding layer
* BiLSTM: a bidirectional LSTM layer
* CRF: a conditional random field layer

Segmentation 是一种标记，用B、M、E、S、0等记号标记一句话中的词:

* B: 开始记号
* M: 中间记号
* E: 结尾记号
* S: 单个字词记号
* O: 标记外

训练模型来标记每个输入序列，经过处理得到最终的分词。

## Training

配置文件为json格式，路径为 `bilstmcrf/config/example_params.json`:

```bash
python -m bilstmcrf.runner --params_file=bilstmcrf/config/example_params.json --mode=train
```

## Eval

```bash
python -m bilstmcrf.runner --params_file=bilstmcrf/config/example_params.json --mode=eval
```

## Predict

```bash
python -m bilstmcrf.runner --params_file=bilstmcrf/config/example_params.json --mode=predict
```

## Train and eval

```bash
python -m bilstmcrf.runner --params_file=bilstmcrf/config/example_params.json --mode=train_and_eval
```

## Export

导出模型

```bash
python -m bilstmcrf.runner --params_file=bilstmcrf/config/example_params.json --mode=export
```
