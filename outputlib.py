import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hparams import hparams, hparams_debug_string
def WriteConfusionSeaborn(m, labels, outpath):
    '''
    INPUT:
        m: confusion matrix (numpy array)
        labels: List of string, the category name of each entry in m
        name: Name for the output png plot
        m：混淆矩阵（numpy数组）
         标签：字符串列表，m中每个条目的类别名称
         name：输出 png 图的名称
    '''
    fig, ax = plt.subplots(dpi=600)  #本来的
    inn = m / m.sum(1, keepdims=True)
    ax = sns.heatmap(inn, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels, annot=True, annot_kws={"size": 6})
    for t in ax.texts:
        t.set_text(t.get_text()[:-1])

    fig.savefig(outpath)
    print(m)
    np.save("confmat.npy", m)
    print(f"Saved figure to {outpath}.")
    plt.close(fig)


# m = np.load("/home/lqb/PycharmProjects/test/olr2020cnn-test-ADI/confmat.npy")
# print(m)
#
# plt.figure(figsize=(15, 10))
#
# sns.heatmap(m,
#             cmap='Reds',
#             annot=True,
#             fmt='.2%'
#             ).get_figure().savefig("temp.png",dpi=500,bbox_inches = 'tight') # fmt显示完全，dpi显示清晰，bbox_inches保存完全
#
# plt.show()


# m = np.load("/home/lqb/PycharmProjects/test/olr2020cnn-test-ADI/confmat.npy")
# print(m)
# fig, ax = plt.subplots(dpi=200)  # 本来的
# inn = m / m.sum(1, keepdims=True)
# ax = sns.heatmap(inn, cmap='Blues', fmt='.2%', xticklabels=hparams.lang, yticklabels=hparams.lang, annot=True,
#                  annot_kws={"size": 6})
# for t in ax.texts:
#     t.set_text(t.get_text()[:-1])
#
# fig.savefig("confmat.png")
#
# plt.show()
