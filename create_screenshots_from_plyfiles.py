import vedo
import plyfile
import pathlib
from natsort import natsorted
import imageio.v3 as imageio

cam = dict(
    position=(3.51484, 3.27242, 4.06787),
    focal_point=(-3.98290e-3, -2.36815e-3, 1.02887e-3),
    viewup=(-0.308731, 0.853375, -0.420043),
    roll=2.37119,
    distance=6.29647,
    clipping_range=(3.00412, 10.4561),
)

def main(ply_file_folder):
    images = []
    plt = vedo.Plotter(interactive=False)
    for ply_filename in natsorted(pathlib.Path(ply_file_folder).rglob("*.ply")):
        print(f"Processing {ply_filename}")
        mesh = vedo.load(str(ply_filename))
        plt.show(mesh, interactive=False, camera=vedo.camera_from_dict(cam))
        vedo.screenshot(ply_filename.with_suffix(".png").as_posix())
        image = imageio.imread(str(ply_filename.with_suffix(".png")))
        images.append(image)
        plt.clear()

    plt.close()        
    imageio.imwrite(ply_filename.parent/'reconstruction.gif', images, duration=300)

if __name__ == "__main__":
    # argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Create screenshots from ply files')
    parser.add_argument('ply_file_folder', type=str, help='Folder containing ply files')
    args = parser.parse_args()
    main(args.ply_file_folder)