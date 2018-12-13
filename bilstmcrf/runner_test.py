import os

import tensorflow as tf

from . import utils
from .runner import Runner

tf.logging.set_verbosity(tf.logging.INFO)


class RunnerTest(tf.test.TestCase):

    @staticmethod
    def _buildParams():
        testdata_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../testdata"))
        params = {
            "model_dir": os.path.join(testdata_dir, "model"),
            "train_src_file": os.path.join(testdata_dir, "feature.txt"),
            "train_tag_file": os.path.join(testdata_dir, "label.txt"),
            "eval_src_file": os.path.join(testdata_dir, "feature.txt"),
            "eval_tag_file": os.path.join(testdata_dir, "label.txt"),
            "predict_src_file": os.path.join(testdata_dir, "feature.txt"),
            "src_vocab": os.path.join(testdata_dir, "vocab.feature.txt"),
            "tag_vocab": os.path.join(testdata_dir, "vocab.label.txt"),
            "pad": "<PAD>",
            "oov_tag": "O",
            "shuffle": True,
            "buff_size": 1000,
            "reshuffle_each_iteration": True,
            "repeat": 2,
            "batch_size": 2,
            "vocab_size": 18,
            "embedding_size": 256,
            "dropout": 0.5,
            "lstm_size": 256,
            "optimizer": "adam",
            "save_ckpt_steps": 50,
            "keep_ckpt_max": 5,
            "log_step_count_steps": 10,
            "num_tags": 5,
            "num_parallel_call": 4,
            "max_src_len": 40,
            "random_seed": 1000,
            "skip_count": 0
        }
        return params

    def testTrain(self):
        r = Runner(self._buildParams())
        r.train()

    def testEval(self):
        r = Runner(self._buildParams())
        r.eval()

    def testTrainAndEval(self):
        r = Runner(self._buildParams())
        r.train_and_eval()

    def testPredict(self):
        r = Runner(self._buildParams())
        predictions = r.predict()
        testdata_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../testdata"))
        tags_file = os.path.join(testdata_dir, "tags_output")
        with open(tags_file, mode="wt", encoding="utf8", buffering=8192) as f:
            for i, p in enumerate(predictions):
                print(i, p['predict_ids'])
                print(i, p['predict_tags'])
                f.write(
                    utils.convert_prediction_tags_to_string(p['predict_tags']))
                f.write("\n")

    def testExport(self):
        r = Runner(self._buildParams())
        r.export()


if __name__ == "__main__":
    tf.test.main()
