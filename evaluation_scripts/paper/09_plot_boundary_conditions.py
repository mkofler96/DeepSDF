import gustaf as gus

n_lines_for_bc = 26
offset_lines_for_bc = 0.05

bounds_right = np.array([[0,0],[0,1]])
bounds_left = np.array([[-offset_lines_for_bc,-offset_lines_for_bc],[-offset_lines_for_bc,1-offset_lines_for_bc]])
resolutions = np.array([n_lines_for_bc,n_lines_for_bc])
verts_right = gus.create.vertices.raster(bounds=bounds_right, resolutions=resolutions)
verts_left = gus.create.vertices.raster(bounds=bounds_left, resolutions=resolutions)
#gus.show([verts_right, verts_left])


### clamp bc
# fancy version
n_v = len(verts_right.vertices) * 2
verts_all = np.empty((n_v, verts_right.vertices.shape[1]))
verts_all[::2] = verts_right.vertices
verts_all[1::2] = verts_left.vertices
dirichlet_bcs = gus.Edges(verts_all, gus.utils.connec.range_to_edges(n_v, continuous=False))
#dirichlet_bcs.show()

# noob version
edges = []
for vr, vl in zip(verts_right.vertices, verts_left.vertices):
    e = gus.Edges([[vr, vl]], [[0,1]])
    e.show_options["as_arrows"] = True

    edges.append(e)

d_e = gus.Edges.concat(edges)
d_e.show_options["c"] = "black"
d_e.show_options["lw"] = 3

d_e.show()
