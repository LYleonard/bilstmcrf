import os

import tensorflow as tf

from . import utils


class UtilsTest(tf.test.TestCase):

    def getParams(self):
        testdata_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../testdata"))
        return {
            "src_vocab": os.path.join(testdata_dir, "vocab.feature.txt"),
            "tag_vocab": os.path.join(testdata_dir, "vocab.label.txt"),
            "unk": "<UNK>",
            "pad": "<PAD>",
            "oov_tag": "O"
        }

    def testCheckSrcVocab(self):
        params = self.getParams()
        utils.check_src_vocab_file(params)

    def testSegmentByTags(self):
        sequence = [
            ['贵', '州', '省', '贵', '阳', '市', '观', '山', '湖', '区', '长', '岭', '南', '路', '160', '号'],
            ['贵', '州', '省', '贵', '阳', '市', '花', '溪', '区', '花', '溪', '大', '道', '2708', '号']
        ]
        tags = [
            ['B', 'M', 'E', 'B', 'M', 'E', 'B', 'M', 'M', 'E', 'B', 'M', 'M', 'E', 'S', 'S'],
            ['B', 'M', 'E', 'B', 'M', 'E', 'B', 'M', 'E', 'B', 'M', 'M', 'E', 'S', 'S']
        ]

        for result in utils.segment_by_tag(sequence, tags):
            print("".join(result))


if __name__ == "__main__":
    tf.test.main()
