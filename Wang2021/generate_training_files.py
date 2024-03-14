import napf
import numpy as np
import scipy
import pathlib
import os

def pixel_to_xy(image, normalized=True):
    image_width = image.shape[1]
    image_height = image.shape[2]
    coords = []
    filled = []
    for x_pixel in range(image_width):
        for y_pixel in range(image_height): 
            # Calculate the corresponding xy coordinates
            x_coord = x_pixel / (image_width) + 1/(2*image_width)
            y_coord = y_pixel / (image_height) + 1/(2*image_width)
            if normalized:
                x_coord = x_coord*2 - 1
                y_coord = y_coord*2 - 1
            coords.append(np.array([x_coord, y_coord]))
            filled.append(image[:,y_pixel, x_pixel])
    return np.array(coords), np.array(filled).astype(bool)

def get_SDF(image, normalized=True):
    coords, filled =  pixel_to_xy(image, normalized=normalized)

    inside = coords[np.where(filled), :][0]
    outside = coords[np.where(np.logical_not(filled)), :][0]

    kdt_out2in = napf.KDT(inside, metric=1)
    dist_out2in, ind = kdt_out2in.knn_search(outside, 1)

    kdt_in2out = napf.KDT(outside, metric=1)
    dist_in2out, ind = kdt_in2out.knn_search(inside, 1)
    pos = np.hstack([outside, dist_out2in])
    neg = np.hstack([inside, -dist_in2out])
    both = np.vstack([pos,neg])
    return {"pos": pos, "neg": neg, "both": both}

def write_npz_file(image, prop, index, path):
    SDF = get_SDF(image)
    pos = SDF["pos"]
    neg = SDF["neg"]
    print(f"Writing {pathlib.Path(path+str(index))}")
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savez(pathlib.Path(path+str(index)), pos=pos, neg=neg, prop=prop)

if __name__ == "__main__":
    # get the data into NumPy format
    mat_data = scipy.io.loadmat('./Wang2021/Geometries.mat')
    data = mat_data['Geometries'].T.reshape(-1,1, 50, 50)
    data = data.astype('float32')
    mat_prop = scipy.io.loadmat('./Wang2021/Properties.mat')
    prop = mat_prop['Properties']
    prop = prop.astype('float32')

    for i in range(data.shape[0]):
        path = "../data/SdfSamples/Wang2021/incl_properties/"
        write_npz_file(data[i], prop[i], i, path)