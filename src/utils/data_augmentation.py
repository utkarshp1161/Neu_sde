""" file to augment data"""

import numpy as np

def symmetry_augmentation(mx, my, angles=4, mirror=False):

    if mirror:
        mx = np.concatenate((mx, -mx, mx), axis = 0)
        my = np.concatenate((my, my, -my), axis = 0)

    mx_, my_ = np.empty(mx.size * angles), np.empty(my.size * angles)
    mx_[:mx.size] = mx
    my_[:my.size] = my

    phis = np.linspace(2 * np.pi / angles, (1 - 1 / angles) * 2 * np.pi, angles - 1)
    for i, phi in enumerate(phis):
        r = np.sqrt(mx ** 2 + my ** 2)
        theta = np.arctan2(my, mx)

        mx_[(mx.size * (i + 1)):(mx.size * (i + 2))] = r * np.cos(theta - phi)
        my_[(my.size * (i + 1)):(my.size * (i + 2))] = r * np.sin(theta - phi)

    return mx_, my_

def symmetry_augmentation_r(mx, my, rad, angles=4, mirror=False):

    if mirror:
        mx = np.concatenate((mx, -mx, mx), axis = 0)
        my = np.concatenate((my, my, -my), axis = 0)
        rad = np.concatenate((rad, rad, rad), axis = 0)

    mx_, my_, rad_ = np.empty(mx.size * angles), np.empty(my.size * angles), np.empty(rad.size * angles)
    mx_[:mx.size] = mx
    my_[:my.size] = my
    rad_[:rad.size] = rad

    phis = np.linspace(2 * np.pi / angles, (1 - 1 / angles) * 2 * np.pi, angles - 1)
    for i, phi in enumerate(phis):
        r = np.sqrt(mx ** 2 + my ** 2)
        theta = np.arctan2(my, mx)

        mx_[(mx.size * (i + 1)):(mx.size * (i + 2))] = r * np.cos(theta - phi)
        my_[(my.size * (i + 1)):(my.size * (i + 2))] = r * np.sin(theta - phi)
        rad_[(rad.size * (i + 1)):(rad.size * (i + 2))] = rad

    return mx_, my_, rad_
    