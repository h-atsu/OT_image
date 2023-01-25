import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ot
from sklearn.cluster import MiniBatchKMeans


def resize_img(img):
    resolution = 480 * 320
    w, h = img.size
    r = np.sqrt(resolution / (w * h))
    img = img.resize((int(w * r), int(h * r)))
    return img


def transform(source_img, target_img):
    w1, h1 = source_img.size
    x1 = np.array(source_img).reshape(-1, 3)

    w2, h2 = target_img.size
    x2 = np.array(target_img).reshape(-1, 3)

    # クラスタリングの計算
    n = 500
    kmeans1 = MiniBatchKMeans(n, random_state=0)
    kmeans1.fit(x1)
    c1 = kmeans1.predict(x1)
    sx1 = kmeans1.cluster_centers_
    a = np.bincount(c1, minlength=n) / len(x1)  # 重みの計算

    kmeans2 = MiniBatchKMeans(n, random_state=0)
    kmeans2.fit(x2)
    c2 = kmeans2.predict(x2)
    sx2 = kmeans2.cluster_centers_
    b = np.bincount(c2, minlength=n) / len(x2)  # 重み

    # 最適輸送の計算
    C = np.linalg.norm(sx1.reshape(-1, 1, 3) -
                       sx2.reshape(1, -1, 3), axis=2) ** 2  # コスト行列の計算
    P = ot.emd(a, b, C)  # 最適輸送行列の計算

    csx1 = P @ sx2 / a.reshape(n, 1)  # 式 (2.52)

    cx1 = x1.copy()
    for i in range(len(x1)):
        j = c1[i]
        cx1[i] = np.maximum(0, np.minimum(
            csx1[j] + x1[i] - sx1[j], 255))  # 式 (2.51)

    res = Image.fromarray(cx1.reshape(h1, w1, 3))
    return res
