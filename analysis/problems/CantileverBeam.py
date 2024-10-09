
import mfem.ser as mfem

from analysis.MFEMLinearElasticity import LinearElasticitySolver, VolumeForceCoefficient3D, SurfaceForceCoefficient3D

import gustaf as gus
import numpy as np
from typing import Union
# import logging

# logger = logging.getLogger("CantileverBeam")

class CantileverBeam:
    def __init__(self, simulation_folder):
        self.strain_energy_density_data = None
        self.u_data = None
        self.simulation_folder = simulation_folder

    def read_mesh(self, mesh_filename):
        self.mesh = mfem.Mesh.LoadFromFile(mesh_filename)
        # self.mesh = mfem.Mesh(mesh_filename, 1, 1)
        # self.mesh = mfem.Mesh.MakeCartesian2D(3, 1, mfem.Element.QUADRILATERAL,
        #                             True, 3.0, 1.0)

    def show_mesh(self):
        verts = self.mesh.GetVertexArray()
        elements = self.mesh.GetElementsArray()

        elements = np.array([self.mesh.GetElementVertices(i) for i in range(self.mesh.GetNE())])

        face_elements = gus.Volumes(verts, elements)
        face_elements.show_options["lw"] = 3
        gus.show(face_elements)

    def set_up(self, ref_levels=0, order=1):
                # u, llambda, mu, force, center, r
        mesh = self.mesh
        

        # 3. Refine the mesh.
        for lev in range(ref_levels):
            mesh.UniformRefinement()

        # define the linear elasticity solver
            # 6. Set-up the physics solver.
        maxat = mesh.bdr_attributes.Max()
        ess_bdr = mfem.intArray([0]*maxat)
        ess_bdr[0] = 1

        nm_bdr = mfem.intArray([0]*maxat)
        nm_bdr[1] = 1

        mfem.ConstantCoefficient(1.0)
        llambda = 0
        mu = 105
        lambda_cf = mfem.ConstantCoefficient(llambda)
        mu_cf = mfem.ConstantCoefficient(mu)

        self.ElasticitySolver = LinearElasticitySolver(lambda_cf, mu_cf)
        self.ElasticitySolver.mesh = mesh
        self.ElasticitySolver.order = order
        self.ElasticitySolver.SetupFEM()
        dim = self.mesh.Dimension()

        center = np.array((2.0, 0.5, 0.5))
        force = np.array((100.0, 0.0, 0.0))
        r = 0.1
        vforce_cf = VolumeForceCoefficient3D(r, center, force)
        xmin = 1.9
        xmax = 2.0
        zmin = 0.0
        sforce_df = SurfaceForceCoefficient3D(xmin, xmax, zmin, force)
        # self.ElasticitySolver.rhs_cf = vforce_cf
        force = np.array((1.0, 0.0, 0.0))
        const_vec = mfem.Vector(3)  # A 2D vector
        const_vec.Assign((0.0, 0.0, -0.01))  # Set the vector components, e.g., (1.0, 2.0)

        # Create a constant vector coefficient from the defined vector.
        sforce_df = mfem.VectorConstantCoefficient(const_vec)
        self.ElasticitySolver.surface_load = sforce_df
        self.ElasticitySolver.ess_bdr = ess_bdr
        self.ElasticitySolver.nm_bdr = nm_bdr

    def solve(self):
        self.ElasticitySolver.SolveState()
        # 7. Save the solution in a grid function.
        u = self.ElasticitySolver.u
        # self.u_data = u.GetDataArray().copy()
        self.u_data = u.GetDataArray()
        max_u = self.u_data.max()
        # logger.debug(f"Finished Solution. Max deflection: {max_u}")
        data_name = "paraview_output"
        paraview_dc = mfem.ParaViewDataCollection(data_name, self.mesh)

        paraview_dc.SetPrefixPath(str(self.simulation_folder))
        paraview_dc.SetLevelsOfDetail(self.ElasticitySolver.order)

        paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
        paraview_dc.SetHighOrderOutput(True)
        paraview_dc.SetCycle(0)
        paraview_dc.SetTime(0.0)
        paraview_dc.RegisterField("displacement", u)
        paraview_dc.Save()
        # logger.debug(f"Saving results to {data_name}")

    def show_solution(self, output: Union[str, list[str]], **kwargs):
        solutions = []
        if type(output) is str:
            output = [output]
        for op in output:
            u_data = self.u_data
            verts = self.mesh.GetVertexArray()
            elements = self.mesh.GetElementsArray()

            elements = np.array([self.mesh.GetElementVertices(i) for i in range(self.mesh.GetNE())])
            # vertices = gus.Vertices(verts)
            # vertices.vertex_data["u"] = u.reshape(-1,2)
            face_elements = gus.Volumes(verts, elements)
            if self.u_data is not None:
                face_elements.vertex_data["u_vec"] = u_data.reshape(-1, 3, order="F")
                face_elements.vertex_data["u_mag"] = np.linalg.norm(u_data.reshape(-1, 3, order="F"), axis=1)
            if self.strain_energy_density_data is not None:
                face_elements.vertex_data["strain_energy_density"] = self.strain_energy_density_data
            solution = face_elements

            available_options = solution.vertex_data.keys()
            if available_options is None:
                raise ValueError("No outputs available. Run simulation to create outputs.")
            if op not in available_options:
                raise ValueError(f"Desired output {op} not available. Available options are: {available_options}.")
            if op in ["u_vec"]:
                solution.show_options["arrow_data"] = op
            elif op in ["u_mag", "strain_energy_density"]:
                solution.show_options["data"] = op
            else:
                raise NotImplementedError(f"Plotting of {op} is not available yet.")
            solution.show_options["lw"] = 3
            solutions.append(solution)
        gus.show(*solutions, axes=1, **kwargs)


    def compute_shape_derivative(self):
        # 8. Compute the compliance.
        if self.u_data is None:
            raise ValueError("No solution found. Compute solution first to get Compliance.")
        compliance = self.ElasticitySolver.clcStrainEnergyDensity()
        derivative = self.ElasticitySolver.clcShapeDerivative(None)
        return derivative
    
    def compute_compliance(self, dTheta=None):
        # 8. Compute the compliance.
        if self.u_data is None:
            raise ValueError("No solution found. Compute solution first to get Compliance.")
        compliance = self.ElasticitySolver.clcStrainEnergyDensity()
        self.strain_energy_density_data = compliance.GetDataArray()
        tot_compliance = compliance.GetDataArray().sum()

        if dTheta is None:
            compl_der = None
        else:
            compl_der = []
            for i in range(dTheta.shape[2]):
                compl_der.append(self.ElasticitySolver.clcComplianceShapeDerivative(dTheta[:,:,i]))

        return tot_compliance, np.array(compl_der)

    def compute_volume(self, dTheta=None):
        vol = self.ElasticitySolver.clcVolume()
        if dTheta is None:
            vol_der = None
        else:
            vol_der = []
            for i in range(dTheta.shape[2]):
                vol_der.append(self.ElasticitySolver.clcVolumeShapeDerivative(dTheta[:,:,i]))
        return vol, np.array(vol_der)


