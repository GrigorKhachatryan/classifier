import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import fetch_olivetti_faces
import scipy
from scipy import fftpack
import time
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


df = fetch_olivetti_faces()


def plot_3(data, num_photo):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (15,6))
    ax4.imshow(data[num_photo[0]], cmap=plt.cm.gray)
    ax5.imshow(data[num_photo[1]], cmap=plt.cm.gray)
    ax1.imshow(df.images[num_photo[0]], cmap=plt.cm.gray)
    ax2.imshow(df.images[num_photo[1]], cmap=plt.cm.gray)

    plt.show()


def plot_3_hist(data, num_photo):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (15,6))
    ax4.hist(data[num_photo[0]])
    ax5.hist(data[num_photo[1]])
    ax1.imshow(df.images[num_photo[0]], cmap=plt.cm.gray)
    ax2.imshow(df.images[num_photo[1]], cmap=plt.cm.gray)
    plt.show()

def hist_data(data, count_col):
    histed_data = []
    for img in data:
        hist = np.histogram(img, bins=np.linspace(0, 1))
        histed_data.append(hist[0])
    return np.array(histed_data)

def scale_data(data, scale):
    scaled_data = []
    for img in data:
        shape = img.shape[0]
        width = int(shape * scale)
        dim = (width, width)
        scaled_data.append(cv2.resize(img, dim))
    return np.array(scaled_data)


def dft_data(data, matrix_size):
    dfted_data = []
    for img in data:
        dft = np.fft.fft2(img)
        dft = np.real(dft)
        dfted_data.append(dft[:matrix_size, :matrix_size])
    return np.abs(dfted_data)

def dct_data(data, matrix_size=8):
    result = []
    for img in data:
        dct = scipy.fftpack.dct(img, axis=1)
        dct = scipy.fftpack.dct(dct, axis=0)
        result.append(dct[:matrix_size,:matrix_size])
    return np.array(result)


def gradient_data(data, n):
    gradiented_data = []
    for img in data:

        shape = img.shape[0]
        i, l = 0, 0
        r = n
        result = []

        while r <= shape:
            window = img[l:r, :]
            result.append(np.sum(window))
            i += 1
            l = i * n
            r = (i + 1) * n
        gradiented_data.append(result)
    return np.array(gradiented_data)


new_show_df = scale_data(df.images, 0.4)
plot_3(new_show_df, [11, 30])
new_show_df = hist_data(df.images, 10)
plot_3_hist(new_show_df, [11, 30])
new_show_df = dct_data(df.images, 10)
plot_3(new_show_df, [11, 30])
new_show_df = gradient_data(df.images, 1)
plot_3_hist(new_show_df, [11,30])



from mpl_toolkits.mplot3d import Axes3D
def plot3d(array_acc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(array_acc)[:,1]*10, np.array(array_acc)[:,0], np.array(array_acc)[:,2], color='red', depthshade=True)
    ax.set_xlabel('Количество тестовых изображений')
    ax.set_ylabel('Параметры')
    ax.set_zlabel('Точность')
    fig.set_figwidth(9)
    fig.set_figheight(9)
    ax.view_init(0, -90)
    plt.show()


def get_best_param(method, params, test_sizes):
    best_score = 0
    params_acc = []
    for param in params:
        for size in test_sizes:
            neigh = KNeighborsClassifier(n_neighbors=1)
            X = method(df.images, param)
            y = df.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size,stratify=y, random_state=24)
            X_train = X_train.reshape(X_train.shape[0],-1)
            neigh.fit(X_train, y_train)
            X_test = X_test.reshape(X_test.shape[0],-1)
            y_predicted = neigh.predict(X_test)
            final_acc = accuracy_score(y_predicted, y_test)
            params_acc.append([param, size, final_acc])
            if final_acc > best_score:
                best_params = [param, size]
                best_score = final_acc
    plot3d(params_acc)
    return best_params, best_score, params_acc

test_sizes = [0.1*i for i in range(1, 10)]


params = [10*i for i in range(1, 11)]
best_params_hist, best_score, array_acc = get_best_param(hist_data, params, test_sizes)
print("Лучший параметр: {0}\nКоличество эталонов: {1}\nКоличество тестовых изображений: {2}\nЛучший результат:{3}".format(best_params_hist[0], best_params_hist[1]*10, 10 - best_params_hist[1]*10, best_score))


params = [0.1*i for i in range(5, 11)]
best_params_scale, best_score, array_acc = get_best_param(scale_data, params, test_sizes)
print("Лучший параметр: {0}\nКоличество эталонов: {1}\nКоличество тестовых изображений: {2}\nЛучший результат:{3}".format(best_params_scale[0], best_params_scale[1]*10, 10 - best_params_scale[1]*10, best_score))


params = [70*i for i in range(1, 6)]
best_params_dft, best_score, array_acc = get_best_param(dft_data, params, test_sizes)
print("Лучший параметр: {0}\nКоличество эталонов: {1}\nКоличество тестовых изображений: {2}\nЛучший результат:{3}".format(best_params_dft[0], best_params_dft[1]*10, 10 - best_params_dft[1]*10, best_score))


params = [10*i for i in range(1, 11)]
best_params_dct, best_score, array_acc = get_best_param(dct_data, params, test_sizes)
print("Лучший параметр: {0}\nКоличество эталонов: {1}\nКоличество тестовых изображений: {2}\nЛучший результат:{3}".format(best_params_dct[0], best_params_dct[1]*10, 10 - best_params_dct[1]*10, best_score))


params = [1*i for i in range(1, 11)]
best_params_gradient, best_score, array_acc = get_best_param(gradient_data, params, test_sizes)
print("Лучший параметр: {0}\nКоличество эталонов: {1}\nКоличество тестовых изображений: {2}\nЛучший результат:{3}".format(best_params_gradient[0], best_params_gradient[1]*10, 10 - best_params_gradient[1]*10, best_score))

import collections



def get_predict(method, param, size):
    neigh = KNeighborsClassifier(n_neighbors=1)
    X = method(df.images, param)
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, stratify=y, random_state=24)
    X_train = X_train.reshape(X_train.shape[0],-1)
    neigh.fit(X_train, y_train)
    X_test = X_test.reshape(X_test.shape[0],-1)
    y_predicted = neigh.predict(X_test)
    return y_predicted, y_test



def par_sistem(methods, test_sizes, params):
    full_acc = []
    for size in test_sizes:
        array_cls_acc = []
        array_y_pred = []
        par_sistem_y_pred = []
        for i in range(len(methods)):
            y_pred, y_test = get_predict(methods[i], params[i], size)
            array_cls_acc.append(accuracy_score(y_pred, y_test))
            array_y_pred.append(y_pred)
        array_y_pred = np.array(array_y_pred)
        for j in range(array_y_pred.shape[1]):
            par_sistem_y_pred.append(collections.Counter(array_y_pred[:,j]).most_common(1)[0][0])
        array_cls_acc.append(accuracy_score(par_sistem_y_pred, y_test))
        full_acc.append(array_cls_acc)
    return np.array(full_acc)



methods = [hist_data, scale_data, dft_data, dct_data, gradient_data]
test_sizes = [0.1*i for i in range(1, 10)]
params = [best_params_hist[0], best_params_scale[0], best_params_dft[0], best_params_dct[0], best_params_gradient[0]]



full_acc = par_sistem(methods, test_sizes, params)





x = np.array(test_sizes)*10

z = full_acc[:,0]
z1 = full_acc[:,1]
z2 =  full_acc[:, 2]
z3 = full_acc[:,3]
z4 = full_acc[:,4]
z5 = full_acc[:,5]

fig = plt.figure()


fig, (ax) = plt.subplots(figsize = (15,6))
ax.plot(x, z, label='Гистограмма яркости')
ax.plot(x, z1, label='Scale')
ax.plot(x, z2, label='DFT')
ax.plot(x, z3, label='DCT')
ax.plot(x, z4, label='Градиентный метод')
ax.plot(x, z5, label='Параллельная система')

ax.set_xlabel('Количество тестовых изображений')
ax.set_ylabel('Точность')

ax.legend()
plt.show()



import pandas as pd



table = pd.DataFrame(full_acc, columns=['Гистограмма яркости', 'Scale', 'DFT', 'DCT', 'Градиентный метод', 'Параллельная система'])



index = ['9/1 | 360/40','8/2 | 320/80','7/3 | 280/120','6/4 | 240/160','5/5 | 200/200','4/6 | 160/240','3/7 | 120/240','2/8 | 80/240','1/9 | 40/240' ]



table.index = index




# Сохраняем данные в таблицу
table.to_excel('path_to_file.xlsx')

table.index.name = 'Вариант разделения'

table = table.to_dict()
json = {}
for key,i in table.items():
    json[key] = [j for key,j in i.items()]
    json[key]

pprint(json)