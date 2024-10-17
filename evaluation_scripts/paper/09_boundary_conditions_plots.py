
import splinepy as sp
import gustaf as gus
import numpy as np
import vedo

rect = sp.helpme.create.box(2, 1, 1)
rect.show_options["control_points"] = False
_TUWIEN_COLOR_SCHEME = {
    "blue": (0, 102, 153),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue_1": (84, 133, 171),
    "blue_2": (114, 173, 213),
    "blue_3": (166, 213, 236),
    "blue_4": (223, 242, 253),
    "grey": (100, 99, 99),
    "grey_1": (157, 157, 156),
    "grey_2": (208, 208, 208),
    "grey_3": (237, 237, 237),
    "green": (0, 126, 113),
    "green_1": (106, 170, 165),
    "green_2": (162, 198, 194),
    "green_3": (233, 241, 240),
    "magenta": (186, 70, 130),
    "magenta_1": (205, 129, 168),
    "magenta_2": (223, 175, 202),
    "magenta_3": (245, 229, 239),
    "yellow": (225, 137, 34),
    "yellow_1": (238, 180, 115),
    "yellow_2": (245, 208, 168),
    "yellow_3": (153, 239, 225),
}
rect.show_options["c"] = _TUWIEN_COLOR_SCHEME["white"]
fix = sp.helpme.create.box(0, 1.5, 1.5)
fix.control_points -= np.array([0.001, 0.25, 0.25])
fix.show_options["control_points"] = False
fix.show_options["c"] = _TUWIEN_COLOR_SCHEME["black"]

n_arrows_x = 6
n_arrows_y = n_arrows_x/2
l_arrows = 0.3
area_of_application = 48/24

start_arrow = np.array([[2-area_of_application,0,1],[2,1,1]])
end_arrow = start_arrow + np.array([[0, 0, l_arrows]])
resolutions = np.array([n_arrows_x,n_arrows_x, n_arrows_y])
verts_start = gus.create.vertices.raster(bounds=start_arrow, resolutions=resolutions)
verts_end = gus.create.vertices.raster(bounds=end_arrow, resolutions=resolutions)

a_edges = []
for vr, vl in zip(verts_start.vertices, verts_end.vertices):
    e = gus.Edges([vl, vr], [[0,1]])
    a_edges.append(e)

d_F = gus.Edges.concat(a_edges)
d_F.show_options["as_arrows"] = True
d_F.show_options["c"] = _TUWIEN_COLOR_SCHEME["blue_1"]
# d_F.show_options["lw"] = 30
cam = dict(
    position=(3.73103, -4.35002, 1.65212),
    focal_point=(0.999999, 0.500001, 0.525011),
    viewup=(-0.0983954, 0.172364, 0.980107),
    distance=5.67905,
    clipping_range=(3.08500, 8.96935),
)
plt = gus.show([fix, rect, d_F], cam=cam, interactive=False) #axes=4, 
plt.add_global_axes(axtype=4)
plt.screenshot("screenshots/boundary_conditions.png").close()


