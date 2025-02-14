from sdf_sampler.sdf_sampler import BoxSDF, SphereSDF, Blend
import numpy as np


def test_blend_box_sphere():
    box = BoxSDF(0.5)
    sphere = SphereSDF(0.5)
    blend = Blend(box, sphere)

    x = np.linspace(-1, 1, num=1000)
    y = np.linspace(-1, 1, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    sdf = blend(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T).reshape(X.shape)


if __name__ == "__main__":
    test_blend_box_sphere()



