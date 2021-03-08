import configparser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from math import *
from numpy import *
import numpy as np
from pathlib import Path
import os
import io
import glob
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm


def thomas(a, b, c, d, Z):
    x1 = a / 2
    y1 = b / 2
    x2 = c / 2
    y2 = d / 2

    x_sum = x1 + x2
    x_dif = x2 - x1
    y_sum = y1 + y2
    y_dif = y2 - y1

    s1_1 = (Z ** 2 + 2 * x_sum ** 2) / (2 * sqrt(Z ** 2 + x_sum ** 2))
    s1_2 = y_sum * atan(y_sum / (sqrt(Z ** 2 + x_sum ** 2))) - y_dif * atan(y_dif / (sqrt(Z ** 2 + x_sum ** 2)))
    s1 = s1_1 * s1_2

    s2_1 = (Z ** 2 + 2 * x_dif ** 2) / (2 * sqrt(Z ** 2 + x_dif ** 2))
    s2_2 = y_sum * atan(y_sum / (sqrt(Z ** 2 + x_dif ** 2))) - y_dif * atan(y_dif / (sqrt(Z ** 2 + x_dif ** 2)))
    s2 = s2_1 * s2_2

    s3_1 = (Z ** 2 + 2 * y_sum ** 2) / (2 * sqrt(Z ** 2 + y_sum ** 2))
    s3_2 = x_sum * atan(x_sum / (sqrt(Z ** 2 + y_sum ** 2))) - x_dif * atan(x_dif / (sqrt(Z ** 2 + y_sum ** 2)))
    s3 = s3_1 * s3_2

    s4_1 = (Z ** 2 + 2 * y_dif ** 2) / (2 * sqrt(Z ** 2 + y_dif ** 2))
    s4_2 = x_sum * atan(x_sum / (sqrt(Z ** 2 + y_dif ** 2))) - x_dif * atan(x_dif / (sqrt(Z ** 2 + y_dif ** 2)))
    s4 = s4_1 * s4_2

    G = s1 - s2 + s3 - s4

    return G


def loading_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    values = {'2X1': float(config['Data']['2X1']),
              '2Y1': float(config['Data']['2Y1']),
              '2X2': float(config['Data']['2X2']),
              '2Y2': float(config['Data']['2Y2']),
              'Z_m': float(config['Data']['Z_m']),
              'Z_c': float(config['Data']['Z_c']),
              'Z_ce': float(config['Data']['Z_ce']),
              'err_2X1': float(config['Data']['err_2X1']),
              'err_2Y1': float(config['Data']['err_2Y1']),
              'err_2X2': float(config['Data']['err_2X2']),
              'err_2Y2': float(config['Data']['err_2Y2']),
              'err_Z_m': float(config['Data']['err_Z_m']),
              'err_Z_c': float(config['Data']['err_Z_c']),
              'err_Z_ce': float(config['Data']['err_Z_ce'])
              }

    return values


def thomas_error_calculator(a, b, c, d, Z, e_x1, e_x2, e_y1, e_y2, e_Z):
    x1 = a / 2
    y1 = b / 2
    x2 = c / 2
    y2 = d / 2

    X = (x1 + x2) / 2
    Y = (y1 + y2) / 2

    sigma_X = 0.5 * sqrt((e_x1 ** 2 + e_x2 ** 2))
    sigma_Y = 0.5 * sqrt((e_y1 ** 2 + e_y2 ** 2))

    G_1_X = 4 * X * Y * (
            (8 * X ** 2 + 3 * Z ** 2) * atan(2 * Y / sqrt(4 * X ** 2 + Z ** 2)) / pow(4 * X ** 2 + Z ** 2, 3 / 2) -
            2 * Y * (8 * X ** 2 + Z ** 2) / ((4 * X ** 2 + Z ** 2) * (4 * X ** 2 + 4 * Y ** 2 + Z ** 2)))
    G_1_Y = (8 * X ** 2 + Z ** 2) * (
            2 * Y / (4 * X ** 2 + 4 * Y ** 2 + Z ** 2) + atan(2 * Y / sqrt(4 * X ** 2 + Z ** 2)) /
            sqrt(4 * X ** 2 + Z ** 2))
    G_1_Z = Y * Z * (Z ** 2 * atan(2 * Y / sqrt(4 * X ** 2 + Z ** 2)) / pow(4 * X ** 2 + Z ** 2, 3 / 2) -
                     2 * Y * (8 * X ** 2 + Z ** 2) / ((4 * X ** 2 + Z ** 2) * (4 * X ** 2 + 4 * Y ** 2 + Z ** 2)))

    G_2_X = 0
    G_2_Y = - 2 * Y / (4 * Y ** 2 / Z ** 2 + 1) - Z * atan(2 * Y / Z)
    G_2_Z = Y * (2 * Y * Z / (4 * Y ** 2 + Z ** 2) - atan(2 * Y / Z))

    G_3_X = (8 * Y ** 2 + Z ** 2) * (
            2 * X / (4 * Y ** 2 + 4 * X ** 2 + Z ** 2) + atan(2 * X / sqrt(4 * Y ** 2 + Z ** 2)) /
            sqrt(4 * Y ** 2 + Z ** 2))
    G_3_Y = 4 * X * Y * (
            (8 * Y ** 2 + 3 * Z ** 2) * atan(2 * X / sqrt(4 * Y ** 2 + Z ** 2)) / pow(4 * Y ** 2 + Z ** 2, 3 / 2) -
            2 * X * (8 * Y ** 2 + Z ** 2) / ((4 * Y ** 2 + Z ** 2) * (4 * Y ** 2 + 4 * X ** 2 + Z ** 2)))
    G_3_Z = X * Z * (Z ** 2 * atan(2 * X / sqrt(4 * Y ** 2 + Z ** 2)) / pow(4 * Y ** 2 + Z ** 2, 3 / 2) -
                     2 * X * (8 * Y ** 2 + Z ** 2) / ((4 * Y ** 2 + Z ** 2) * (4 * Y ** 2 + 4 * X ** 2 + Z ** 2)))

    G_4_X = - 2 * X / (4 * X ** 2 / Z ** 2 + 1) - Z * atan(2 * X / Z)
    G_4_Y = 0
    G_4_Z = X * (2 * X * Z / (4 * X ** 2 + Z ** 2) - atan(2 * X / Z))

    G_X = G_1_X + G_2_X + G_3_X + G_4_X
    G_Y = G_1_Y + G_2_Y + G_3_Y + G_4_Y
    G_Z = G_1_Z + G_2_Z + G_3_Z + G_4_Z

    errG = sqrt(pow(G_X * sigma_X, 2) + pow(G_Y * sigma_Y, 2) + pow(G_Z * e_Z, 2))

    dict = {'Dx': G_X,
            'Dy': G_Y,
            'Dz': G_Z,
            'eG': errG}

    return dict


def ratio_G(a, b, c, d, Z, e_x1, e_x2, e_y1, e_y2, e_Z):
    x1 = a / 2
    y1 = b / 2
    x2 = c / 2
    y2 = d / 2

    X = (x1 + x2) / 2
    Y = (y1 + y2) / 2

    sigma_X = 0.5 * sqrt((e_x1 ** 2 + e_x2 ** 2))
    sigma_Y = 0.5 * sqrt((e_y1 ** 2 + e_y2 ** 2))

    G_S = 2 * pi * X * Y

    G_t = thomas(a, b, c, d, Z)

    R_1_X = (-1 / (2 * pi)) * (8 * Y * (8 * X ** 2 + Z ** 2) / (
            (4 * X ** 2 + Z ** 2) * (4 * X ** 2 + 4 * Y ** 2 + Z ** 2)) + Z ** 4 * atan(
        2 * Y / sqrt(4 * X ** 2 + Z ** 2)) / (X ** 2 * pow(4 * X ** 2 + Z ** 2, 3 / 2)))
    R_1_Y = (8 * X ** 2 + Z ** 2) / (4 * pi * X ** 3 + 4 * pi * X * Y ** 2 + pi * X * Z ** 2)
    R_1_Z = (Z / (2 * pi * X)) * (Z ** 2 * atan(2 * Y / sqrt(4 * X ** 2 + Z ** 2)) / pow(4 * X ** 2 + Z ** 2, 3 / 2) - (
            2 * Y * (8 * X ** 2 + Z ** 2) / ((4 * X ** 2 + Z ** 2) * (4 * X ** 2 + 4 * Y ** 2 + Z ** 2))))

    R_2_X = Z * atan(2 * Y / Z) / (2 * pi * X ** 2)
    R_2_Y = -Z ** 2 / (4 * pi * X * Y ** 2 + pi * X * Z ** 2)
    R_2_Z = (atan(2 * Y / Z) / (2 * pi * X ** 2) - 2 * Y * Z / (4 * Y ** 2 + Z ** 2)) / (2 * pi * X)

    R_3_X = (8 * Y ** 2 + Z ** 2) / (4 * pi * Y ** 3 + 4 * pi * Y * X ** 2 + pi * Y * Z ** 2)
    R_3_Y = (-1 / (2 * pi)) * (8 * X * (8 * Y ** 2 + Z ** 2) / (
            (4 * Y ** 2 + Z ** 2) * (4 * Y ** 2 + 4 * X ** 2 + Z ** 2)) + Z ** 4 * atan(
        2 * X / sqrt(4 * Y ** 2 + Z ** 2)) / (Y ** 2 * pow(4 * Y ** 2 + Z ** 2, 3 / 2)))
    R_3_Z = (Z / (2 * pi * Y)) * (Z ** 2 * atan(2 * X / sqrt(4 * Y ** 2 + Z ** 2)) / pow(4 * Y ** 2 + Z ** 2, 3 / 2) - (
            2 * X * (8 * Y ** 2 + Z ** 2) / ((4 * Y ** 2 + Z ** 2) * (4 * Y ** 2 + 4 * X ** 2 + Z ** 2))))

    R_4_X = -Z ** 2 / (4 * pi * Y * X ** 2 + pi * Y * Z ** 2)
    R_4_Y = Z * atan(2 * X / Z) / (2 * pi * Y ** 2)
    R_4_Z = (atan(2 * X / Z) / (2 * pi * Y ** 2) - 2 * X * Z / (4 * X ** 2 + Z ** 2)) / (2 * pi * Y)

    R_X = R_1_X + R_2_X + R_3_X + R_4_X
    R_Y = R_1_Y + R_2_Y + R_3_Y + R_4_Y
    R_Z = R_1_Z + R_2_Z + R_3_Z + R_4_Z

    errR = sqrt(pow(R_X * sigma_X, 2) + pow(R_Y * sigma_Y, 2) + pow(R_Z * e_Z, 2))

    ratio = {'R': G_t / G_S,
             'eR': errR}

    return ratio


def intensity_error_calculator(C, eff1, eff2, G, e_C, e_eff1, e_eff2, e_G):
    I = C / (eff1 * eff2 * G)

    I_C = 1 / (eff1 * eff2 * G)
    I_eff1 = - C / (G * eff2 * eff1 ** 2)
    I_eff2 = - C / (G * eff1 * eff2 ** 2)
    I_G = - C / (eff1 * eff2 * G ** 2)
    e_I = sqrt(pow(I_C * e_C, 2) + pow(I_eff1 * e_eff1, 2) + pow(I_eff2 * e_eff2, 2) + pow(I_G * e_G, 2))

    dict = {'I': I,
            'eI': e_I}

    return dict


def rate_calculator(WorkDir, DetName):
    paths = glob.glob(os.path.join(WorkDir, DetName + "*.txt"))  # It stores every file that match in a list
    paths.sort()
    doppie = np.array([])
    for path in paths:
        with io.open(path, mode="r", encoding="utf-8") as fd:
            for line in fd:
                if not line.startswith('#'):
                    values = [float(s) for s in line.split()]
                    doppie = np.append(doppie, values[3] / 180)
    return doppie


def main():
    v = loading_config('configThomas.txt')
    G_m = thomas(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_m'])
    G_c = thomas(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_c'])
    G_ce = thomas(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_ce'])

    thom_m = thomas_error_calculator(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_m'], v['err_2X1'], v['err_2Y1'],
                                     v['err_2X2'], v['err_2Y2'], v['err_Z_m'])
    thom_c = thomas_error_calculator(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_c'], v['err_2X1'], v['err_2Y1'],
                                     v['err_2X2'], v['err_2Y2'], v['err_Z_c'])
    thom_ce = thomas_error_calculator(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_ce'], v['err_2X1'], v['err_2Y1'],
                                      v['err_2X2'], v['err_2Y2'], v['err_Z_ce'])

    # ratio_m = ratio_G(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_m'], v['err_2X1'], v['err_2Y1'],
    #                   v['err_2X2'], v['err_2Y2'], v['err_Z_m'])
    # ratio_c = ratio_G(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_c'], v['err_2X1'], v['err_2Y1'],
    #                   v['err_2X2'], v['err_2Y2'], v['err_Z_c'])
    # ratio_ce = ratio_G(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], v['Z_ce'], v['err_2X1'], v['err_2Y1'],
    #                    v['err_2X2'], v['err_2Y2'], v['err_Z_ce'])

    # rate_m = rate_calculator('./Sullivan_Thomas/Efficiency', 'Minosse')
    # mean_m = np.mean(rate_m)
    # devstd_m = np.std(rate_m)

    # rate_c = rate_calculator('./Sullivan_Thomas/Efficiency', 'Caronte')
    # mean_c = np.mean(rate_c)
    # devstd_c = np.std(rate_c)

    # rate_ce = rate_calculator('./Sullivan_Thomas/Efficiency', 'Cerbero')
    # mean_ce = np.mean(rate_ce)
    # devstd_ce = np.std(rate_ce)

    # i_m = intensity_error_calculator(mean_m, 0.9937, 0.9938, G_m, devstd_m, 0.0013, 0.0013, thom_m['eG'])
    # i_c = intensity_error_calculator(mean_c, 0.9938, 0.9948, G_c, devstd_c, 0.0013, 0.0011, thom_c['eG'])
    # i_ce = intensity_error_calculator(mean_ce, 0.9937, 0.9948, G_ce, devstd_ce, 0.0013, 0.0011, thom_ce['eG'])

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    x = np.arange(0.0001, 0.20, 0.0001)
    f2 = np.vectorize(thomas)

    sul = ax.plot(x, f2(v['2X1'], v['2Y1'], v['2X2'], v['2Y2'], x), label='G', linestyle='-')
    ax.set_title('Geometrical factor')
    ax.set_xlabel('Z $[m]$', color='black')
    ax.set_ylabel(r'G $[m^{2}\cdot sr]$', color='black')

    ax.vlines(x=v['Z_m'], ymin=0, ymax=G_m, color='red')
    ax.hlines(y=G_m, xmin=0, xmax=v['Z_m'], color='red')
    ax.vlines(x=v['Z_c'], ymin=0, ymax=G_c, color='mediumorchid')
    ax.hlines(y=G_c, xmin=0, xmax=v['Z_c'], color='mediumorchid')
    ax.vlines(x=v['Z_ce'], ymin=0, ymax=G_ce, color='darkorange')
    ax.hlines(y=G_ce, xmin=0, xmax=v['Z_ce'], color='darkorange')

    arrowprops_m = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05, 'color': 'red'}
    arrowprops_c = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05, 'color': 'mediumorchid'}
    arrowprops_ce = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05, 'color': 'darkorange'}
    ax.annotate('Z = ' + str(v['Z_m']), xy=(v['Z_m'], G_m), xytext=(50, 10), textcoords='offset points',
                va='bottom', ha='center', annotation_clip=False, arrowprops=arrowprops_m)
    ax.annotate('Z = ' + str(v['Z_c']), xy=(v['Z_c'], G_c), xytext=(50, 10), textcoords='offset points',
                va='bottom', ha='center', annotation_clip=False, arrowprops=arrowprops_c)
    ax.annotate('Z = ' + str(v['Z_ce']), xy=(v['Z_ce'], G_ce), xytext=(50, 10), textcoords='offset points',
                va='bottom', ha='center', annotation_clip=False, arrowprops=arrowprops_ce)
    legend = ax.legend(loc='best', shadow=True, fontsize='medium')

    # old_output = f"""
    #         ------------------------ LEGEND AND UNITS -----------------------
    #         |\t C is the coincidence rate expressed in [s^(-1)]\t\t\t|
    #         |\t G is the geometric factor expressed in [m^(-2)*sr^(-1)]\t|
    #         |\t I is the intensity expressed in [m^(-2)*s^(-1)*sr^(-1)]\t|
    #         -----------------------------------------------------------------
    #         RESULTS:
    #         Minosse (centrale)
    #         \t\t C = {mean_m} +/- {devstd_m}
    #         \t\t G = {G_m} +/- {thom_m['eG']}
    #         \t\t I = {i_m['I']} +/- {i_m['eI']}
    #         \t\t G_t/G_S =  + {ratio_m['R']} +/- {ratio_m['eR']}
    #         Caronte (centrale)
    #         \t\t C =  + {mean_c} +/- {devstd_c}
    #         \t\t G =  + {G_c} +/- {thom_c['eG']}
    #         \t\t I =  + {i_c['I']} +/- {i_c['eI']}
    #         \t\t G_t/G_S =  + {ratio_c['R']} +/- {ratio_ce['eR']}
    #         Cerbero (centrale)
    #         \t\t C =  + {mean_ce} +/- {devstd_ce}
    #         \t\t G =  + {G_ce} +/- {thom_ce['eG']}
    #         \t\t I =  + {i_ce['I']} +/- {i_ce['eI']}
    #         \t\t G_t/G_S =  + {ratio_ce['R']} +/- {ratio_ce['eR']}
    #     """
    output = f"""
Z = {v['Z_m']} [m] ----> G = {round(G_m, 6)} +/- {round(thom_m['eG'], 6)} ({round(thom_m['eG']/G_m*100, 2)} %)
Z = {v['Z_c']} [m] ----> G = {round(G_c, 6)} +/- {round(thom_c['eG'], 6)} ({round(thom_c['eG']/G_c*100, 2)} %)
Z = {v['Z_ce']}  [m] ----> G = {round(G_ce, 6)} +/- {round(thom_ce['eG'], 6)} ({round(thom_ce['eG']/G_ce*100, 2)} %)
"""
    print(output, file=open(os.path.abspath("Thomas_output.txt"), 'w'))

    # old_infoDict = {'C_m': round(mean_m, 2),
    #             'Err_C_m': round(devstd_m, 2),
    #             'G_m': round(G_m, 4),
    #             'Err_G_m': round(thom_m['eG'], 4),
    #             'i_m': round(i_m['I'], 2),
    #             'Err_i_m': round(i_m['eI'], 2),
    #             'R_m': round(ratio_m['R'], 4),
    #             'Err_R_m': round(ratio_m['eR'], 4),
    #             'C_c': round(mean_c, 2),
    #             'Err_C_c': round(devstd_c, 2),
    #             'G_c': round(G_c, 4),
    #             'Err_G_c': round(thom_c['eG'], 4),
    #             'i_c': round(i_c['I'], 2),
    #             'Err_i_c': round(i_c['eI'], 2),
    #             'R_c': round(ratio_c['R'], 4),
    #             'Err_R_c': round(ratio_c['eR'], 4),
    #             'C_ce': round(mean_ce, 2),
    #             'Err_C_ce': round(devstd_ce, 2),
    #             'G_ce': round(G_ce, 4),
    #             'Err_G_ce': round(thom_ce['eG'], 4),
    #             'i_ce': round(i_ce['I'], 2),
    #             'Err_i_ce': round(i_ce['eI'], 2),
    #             'R_ce': round(ratio_ce['R'], 4),
    #             'Err_R_ce': round(ratio_ce['eR'], 4)
    #             }
    infoDict = {'G_m': round(G_m, 4),
                'Err_G_m': round(thom_m['eG'], 4),
                'G_c': round(G_c, 4),
                'Err_G_c': round(thom_c['eG'], 4),
                'G_ce': round(G_ce, 4),
                'Err_G_ce': round(thom_ce['eG'], 4)
                }
    return infoDict


def show():
    main()
    plt.show()


if __name__ == '__main__':
    show()
