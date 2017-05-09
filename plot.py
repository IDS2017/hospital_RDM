import pickle
import matplotlib.pyplot as plt

# def boxPlot(data):

#     tmp = []
#     for v in data.values():
#         tmp.append(v)

#     plt.figure(figsize=(10, 6))

#     ax = plt.subplot(111)
#     for i in range(len(tmp)):
#         # ax.boxplot(tmp[i], positions = [i],widths = 0.35 ,showfliers=False, patch_artist=True)
#         ax.boxplot(tmp[i], positions=[i], widths=0.35, patch_artist=True)
#         ax.set_title('Comparison of ML models accuracy', fontsize=20)

#     plt.xticks(range(len(tmp)), data.keys())
#     ax.set_xlim(-1, len(tmp))
#     fig_name = str(int(time.time())) + '.png'
#     plt.savefig(fig_name)
#     plt.show()

def plotAUC():
    with open("scores.p", "rb") as f:
        scores = pickle.load(f)
    cs = [i for i in range(-5,5)]
    for m in scores:
        ss = [v2 for v1,v2 in scores[m]]
        plt.plot(cs, ss, label='%s' % (m))
    plt.xlim(cs)
    plt.ylim([0, 1])
    plt.xlabel('log10(C)')
    plt.ylabel('AUC')
    plt.legend(loc="lower right")
    plt.show()
