import numpy as np
import matplotlib.pyplot as plt


def GlafDraw(npy_pass):
    pos = np.load(npy_pass)
    rates = []
    for emp in list(pos):
        #ユーグリッド距離比率計算
        # 0鼻 1左目 2右目 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘
        # 9左手首 10右手首 11左腰 12右腰 13左膝 14右膝
        dis0 = np.linalg.norm(emp[0]-emp[1])
        dis1 = np.linalg.norm(emp[2]-emp[1])
        dis2 = np.linalg.norm(emp[4]-emp[3])
        rate = np.array([dis0, dis1, dis2]) / dis0
    
        #各距離比率の配列([[dis0][dis1]....])
        rates.append(list(rate))
    rates = np.array(rates).T
    print(rates)
    
    for i in rates:
        plt.plot(np.arange(len(i)), i)
    plt.xlim(0, len(rates[0]))
    plt.show()


npy_passes = ["C:/Users/Ono Hitoshi/output.npy"]
for npy_pass in npy_passes:
    GlafDraw(npy_pass)