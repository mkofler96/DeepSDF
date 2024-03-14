from generate_training_files import pixel_to_xy, get_SDF
import splinepy as sp
import numpy as np
import scipy
from scipy.optimize import least_squares
import datetime

def evaluate_spline(spline, points, control_points_z):
    spline.control_points[:,0] = control_points_z
    # Calculate proximities between the spline and the given points
    sp_eval = spline.evaluate(points[:,:-1])
    pts = sp_eval[:,0] - points[:,2]
    return pts


def fit_SDF(SDF, size_u, size_v, degree_u, degree_v):
    
    sdf = SDF["both"]

    ind = np.lexsort((sdf[:,0],sdf[:,1])) 
    control_points = np.array([[0],
                                [0],
                                [0],
                                [0]])
    knot_vectors = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    spline = sp.BSpline(degrees=[1, 1], control_points=control_points, knot_vectors=knot_vectors)
    spline.insert_knots(0, np.linspace(0,1,size_u+1)[1:-1])
    spline.insert_knots(1, np.linspace(0,1,size_v+1)[1:-1])
    spline.elevate_degrees([0]*(degree_u-1))
    spline.elevate_degrees([1]*(degree_v-1))
    
    approx_spline = lambda control_points_z: evaluate_spline(spline, sdf[ind], control_points_z)

    init_guess = np.random.normal(size=spline.control_points[:,0].shape)
    result = least_squares(approx_spline, init_guess)
    spline.control_points[:,0] = result.x
    return spline

def have_equal_base(spline_a: sp.BSpline, spline_b: sp.BSpline):
    np.testing.assert_allclose(spline_a.knot_vectors, spline_b.knot_vectors)
    np.testing.assert_allclose(spline_a.degrees, spline_b.degrees)
    np.testing.assert_allclose(spline_a.dim, spline_b.dim)
    np.testing.assert_allclose(spline_a.para_dim, spline_b.para_dim)

def save_npz(fname, list_of_splines, list_of_names):
        # Initialize an empty dictionary to store the properties of each spline
    property_dicts = {}

    # set first spline to be master spline which all other splines are compared to
    common_spline = list_of_splines[0]

    # Add number of splines in list to dict
    property_dicts["number_of_splines"] = len(list_of_splines)
    property_dicts["degrees"] = list_of_splines[0].degrees

    property_dicts[f"knot_vectors"] = common_spline.knot_vectors

    # Loop through the list of splines and add their properties to the dictionary
    for name, spline in zip(list_of_names, list_of_splines):
        # Use a prefix to distinguish different splines

        # Add the common properties of all splines
        property_dicts[f"{name}_control_points"] = spline.control_points

        have_equal_base(common_spline, spline)

    # Save the dictionary as `.npz`
    np.savez(
        fname,
        **property_dicts,
    )



if __name__ == "__main__":
    # get the data into NumPy format
    mat_data = scipy.io.loadmat('Geometries.mat')
    data = mat_data['Geometries'].T.reshape(-1,1, 50, 50)
    data = data.astype('float32')
    splines = []
    names = []
    for i in range(data.shape[0]):
        image=data[i]
        SDF = get_SDF(image, normalized=False)
        coords, filled =  pixel_to_xy(data[i], normalized=False)
        fitted_spline = fit_SDF(SDF, 6, 6, 2, 2)
        splines.append(fitted_spline)
        names.append(i)

        # Get current time
        time_string = datetime.datetime.now().strftime("%H:%M:%S")

        print(f"{time_string} {i}/{data.shape[0]}")
    save_npz("Wang2021_splines.npz", splines, names)