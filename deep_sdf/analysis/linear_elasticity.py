
import mfem.ser as mfem

from deep_sdf.analysis.mfem_prereq import LinearElasticitySolver, VolumeForceCoefficient3D

import gustaf as gus
import numpy as np

class LinearElasticityProblem:
    def __init__(self):
        self.strain_energy_density_data = None
        self.u_data = None

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

    def solve(self, ref_levels=1, order=1):
        # u, llambda, mu, force, center, r
        mesh = self.mesh
        dim = mesh.Dimension()

        # 3. Refine the mesh.
        for lev in range(ref_levels):
            mesh.UniformRefinement()

        # 4. Define the necessary finite element spaces on the mesh.
        state_fec = mfem.H1_FECollection(order, dim)    # space for u
        filter_fec = mfem.H1_FECollection(order, dim)   # space for ρ̃
        control_fec = mfem.L2_FECollection(order-1, dim,
                                        mfem.BasisType.GaussLobatto)  # space for ψ
        state_fes = mfem.FiniteElementSpace(mesh, state_fec, dim)
        filter_fes = mfem.FiniteElementSpace(mesh, filter_fec)
        control_fes = mfem.FiniteElementSpace(mesh, control_fec)

        state_size = state_fes.GetTrueVSize()
        control_size = control_fes.GetTrueVSize()
        filter_size = filter_fes.GetTrueVSize()

        print("Number of state unknowns: " + str(state_size))
        print("Number of filter unknowns: " + str(filter_size))
        print("Number of control unknowns: " + str(control_size))

        # define the linear elasticity solver
            # 6. Set-up the physics solver.
        maxat = mesh.bdr_attributes.Max()
        ess_bdr = mfem.intArray([0]*maxat)
        ess_bdr[0] = 1

        mfem.ConstantCoefficient(1.0)
        llambda = 0
        mu = 105
        lambda_cf = mfem.ConstantCoefficient(llambda)
        mu_cf = mfem.ConstantCoefficient(mu)

        self.ElasticitySolver = LinearElasticitySolver(lambda_cf, mu_cf)
        self.ElasticitySolver.mesh = mesh
        self.ElasticitySolver.order = state_fec.GetOrder()
        self.ElasticitySolver.SetupFEM()

        center = np.array((2.0, 0.5, 0.5))
        force = np.array((0.0, 0.0, -100.0))
        r = 0.1
        vforce_cf = VolumeForceCoefficient3D(r, center, force)
        self.ElasticitySolver.rhs_cf = vforce_cf
        self.ElasticitySolver.ess_bdr = ess_bdr
        self.ElasticitySolver.SolveState()

        # 7. Save the solution in a grid function.
        u = self.ElasticitySolver.u
        u = mfem.GridFunction(state_fes)
        u.Assign(self.ElasticitySolver.u)
        self.u_data = u.GetDataArray().copy()
        max_u = self.u_data.max()
        print(f"Finished Solution. Max deflection: {max_u}")

    def show_solution(self, output: str, **kwargs):
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
        self.solution = face_elements

        available_options = self.solution.vertex_data.keys()
        if available_options is None:
            raise ValueError("No outputs available. Run simulation to create outputs.")
        if output not in available_options:
            raise ValueError(f"Desired output {output} not available. Available options are: {available_options}.")
        if output in ["u_vec"]:
            self.solution.show_options["arrow_data"] = output
        elif output in ["u_mag", "strain_energy_density"]:
            self.solution.show_options["data"] = output
        else:
            raise NotImplementedError(f"Plotting of {output} is not available yet.")
        self.solution.show_options["lw"] = 3
        gus.show(self.solution, axes=1, **kwargs)

        # show_solution = False
        # if show_solution:
        #     # 7. Save the solution in a grid function.
        #     u = self.ElasticitySolver.u
        #     u = mfem.GridFunction(state_fes)
        #     u.Assign(self.ElasticitySolver.u)
        #     u_data = u.GetDataArray()
        #     verts = mesh.GetVertexArray()
        #     elements = mesh.GetElementsArray()

        #     elements = np.array([mesh.GetElementVertices(i) for i in range(mesh.GetNE())])
        #     # vertices = gus.Vertices(verts)
        #     # vertices.vertex_data["u"] = u.reshape(-1,2)
        #     face_elements = gus.Faces(verts, elements)
        #     face_elements.vertex_data["my_data"] = u_data.reshape(-1, 2, order="F")*1000
        #     face_elements.show_options["arrow_data"] = "my_data"
        #     face_elements.show_options["lw"] = 3
        #     gus.show(face_elements)

    def compute_compliance(self):
        # 8. Compute the compliance.
        if self.u_data is None:
            raise ValueError("No solution found. Compute solution first to get Compliance.")
        compliance = self.ElasticitySolver.clcStrainEnergyDensity() 
        self.strain_energy_density_data = compliance.GetDataArray()