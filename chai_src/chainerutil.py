import chainer
from chainer import Link, Chain, ChainList
import chainer.functions as F#パラメータを持たない関数
import chainer.links as L#パラメータを持つ関数
import numpy as np
#chainはパラメータ層をもつ（link）をまとめて置くためのクラス。chain.params()メソッドで更新されるパラメータ一覧を取得


#参考URL：https://qiita.com/icoxfog417/items/96ecaff323434c8d677b
#http://ai-kenkyujo.com/2017/09/18/chainer-deeplearning/
#https://qiita.com/mitmul/items/1e35fba085eb07a92560

class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(784, 10)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return h
