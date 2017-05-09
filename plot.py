# import matplotlib.pyplot as plt

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
