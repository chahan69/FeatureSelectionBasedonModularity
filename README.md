# Feature Selection Based on Modularity

## 行列を受け取って特徴のスコアのリストを返します

卒業研究のアルゴリズムをPythonで実装したものになります．

Numpyまでしか使っていない+forループを$N_F$回回しているのでそこまで早くないです

'''
  obj = fsbm.fsbm(data) #コンストラクタ data -> list or np.array not including header
  ranking = obj.ranking #-> scores of each features
'''
