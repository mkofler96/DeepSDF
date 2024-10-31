import numpy as np
import argparse
import splinepy as sp
import gustaf as gus
import pathlib
import imageio.v3 as imageio
import json

def create_animation(surf_folder, add_boundary_conditions=False, show_latent_vector=False):
    mesh_files = []
    surf_folder = pathlib.Path(surf_folder)
    for surf_filename in surf_folder.iterdir():
        if ("surf" in str(surf_filename.stem)) and (surf_filename.suffix==".inp"):
            sim_index = surf_filename.stem.replace("surf", "")
            print(f"Surface filename: {surf_filename}, sim index: {sim_index}")
            mesh_files.append((int(sim_index), surf_filename))
        else:
            print(f"Not a surface mesh: {surf_filename}")
    if len(mesh_files) == 0:
        Warning("No mesh file found. Exiting.")
        return None
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

    if add_boundary_conditions:
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

    if show_latent_vector:
        latent_plot = sp.helpme.create.box(0, 1.5, 1.5)
        with open(surf_folder/'config.json', 'r') as file:
            spline_config = json.load(file)
        # knot_vectors = [[-1]*(n[0]+1)+[1]*(n[0]+1),
        #                 [-1]*(n[1]+1)+[1]*(n[1]+1),
        #                 [-1]*(n[2]+1)+[1]*(n[2]+1)]
        # n_initial control points = order + 1
        # n_initial_control_points = np.prod(n+1)
        latent_spline = sp.helpme.create.box(2,1,1).bspline
        degrees = np.array(spline_config["mesh"]["degrees"])
        while not np.all(latent_spline.degrees==degrees):
            new_deg = degrees - latent_spline.degrees
            latent_spline.elevate_degrees(np.argwhere(new_deg>0))

        n = np.array(spline_config["mesh"]["degrees"])
        knot_vectors = [[-1]*(n[0]+1)+[1]*(n[0]+1),
                        [-1]*(n[1]+1)+[1]*(n[1]+1),
                        [-1]*(n[2]+1)+[1]*(n[2]+1)]
        # n_initial control points = order + 1
        n_initial_control_points = np.prod(n+1)

        with open(surf_folder/'results.json', 'r') as file:
            results = json.load(file)
        initial_control_points = np.zeros((n_initial_control_points,len(results["design_vector"][0][0])))
        
        latent_vec_interpolation = sp.BSpline(
            degrees=n,
            knot_vectors=knot_vectors,
            control_points=initial_control_points,
        )
        latent_vec_interpolation.uniform_refine(spline_config["mesh"]["refinement"])
        latent_spline.uniform_refine(spline_config["mesh"]["refinement"])
        latent_spline.spline_data["me"] = latent_vec_interpolation
        latent_spline.show_options["data"] = "me"
        latent_spline.show_options["control_points"] = False
    images = []
    for index, mesh_file in sorted(mesh_files, 
                                    key=lambda surf_ind_tuple: surf_ind_tuple[0]):
        print(f"Creating sreenshot of {mesh_file}")
        mesh = gus.io.meshio.load(str(mesh_file))
        mesh.show_options["c"] = _TUWIEN_COLOR_SCHEME["grey_2"]
        cam = dict(
            position=(3.25705, -3.50828, 1.45651),
            focal_point=(1.00000, 0.500000, 0.525011),
            viewup=(-0.0983954, 0.172364, 0.980107),
            roll=-70.6513,
            distance=4.69343,
            clipping_range=(2.65216, 7.27428),
        )
        if add_boundary_conditions:
            shown_geom = [mesh, fix, d_F]
            name = "animtation_with_fix"
        else:
            shown_geom = [mesh]
            name = "animation"  

        if show_latent_vector:
            latent_vec_interpolation.control_points = results["design_vector"][index-1]
            latent_spline.spline_data["me"] = latent_vec_interpolation
            showable = gus.show([f"Iteration: {index:<3}", shown_geom], latent_spline, 
                                cam=cam, 
                                interactive=False, 
                                offscreen=True,
                                vmin=spline_config["optimization"]["bounds"][0],
                                vmax=spline_config["optimization"]["bounds"][1])
        else:    
            showable = gus.show(shown_geom, cam=cam, interactive=False, offscreen=True)


        showable.screenshot(mesh_file.with_suffix(".png").as_posix())
        image = imageio.imread(str(mesh_file.with_suffix(".png")))
        images.append(image)

    imageio.imwrite(surf_folder/f"{name}.gif", images, duration=300)    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run a DeepMS optimization")
    surface_dir = "surfs_final_optimization"
    create_animation(surface_dir, show_latent_vector=False)