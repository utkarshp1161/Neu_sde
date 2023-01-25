from utils.data_augmentation import symmetry_augmentation
import pandas as pd
import numpy as np


def main(read_path_file = "", write_path_file = ""):
    data = pd.read_csv(read_path_file, sep ="\t", header=None)
    #data = np.load(read_path_file)
    mx = data.values[:,0]
    my = data.values[:,1]
    #mx = data[:,0]
    #my = data[:,1]
    #rad = data[:,2]
    print("old_data_len", len(mx))
    mxy = symmetry_augmentation(mx, my, angles=16, mirror=True)
    print("new_data_len", len(mxy))
    np.save(write_path_file, np.hstack((mxy[0].reshape(-1,1), mxy[1].reshape(-1,1))))


if __name__ == '__main__':
    main(read_path_file = "path_to_file", write_path_file = "path")
