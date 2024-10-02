
import mfem.ser as mfem

from deep_sdf.analysis.mfem_prereq import LinearElasticitySolver, VolumeForceCoefficient3D

import gustaf as gus
import numpy as np

class LinearElasticityProblem:
    def __init__(self):
        pass

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

    def get_volume_and_compliance(self, ref_levels=1):
        # u, llambda, mu, force, center, r
        order = 1
        mesh = self.mesh
        dim = mesh.Dimension()

        # 2. Set BCs.
        for i in range(mesh.GetNBE()):
            be = mesh.GetBdrElement(i)
            vertices = mesh.GetBdrElementVertices(i)   # this method returns list

            coords1 = mesh.GetVertexArray(vertices[0])   # this returns numpy array
            coords2 = mesh.GetVertexArray(vertices[1])

            center = (coords1 + coords2)/2

            # if abs(center[0] - 0.0) < 1e-10:
            #     # the left edge
            #     be.SetAttribute(1)
            # else:
            #     # all other boundaries
            #     be.SetAttribute(2)
        # mesh.SetAttributes()
        # print(mesh.GetAttributeArray())

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

        show_mesh = False
        if show_mesh:
            verts = mesh.GetVertexArray()
            elements = mesh.GetElementsArray()

            elements = np.array([mesh.GetElementVertices(i) for i in range(mesh.GetNE())])

            face_elements = gus.Faces(verts, elements)
            face_elements.show_options["lw"] = 3
            gus.show(face_elements)


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

        ElasticitySolver = LinearElasticitySolver(lambda_cf, mu_cf)
        ElasticitySolver.mesh = mesh
        ElasticitySolver.order = state_fec.GetOrder()
        ElasticitySolver.SetupFEM()

        center = np.array((2.0, 0.5, 0.5))
        force = np.array((0.0, -1.0, 0.0))
        r = 0.05
        vforce_cf = VolumeForceCoefficient3D(r, center, force)
        ElasticitySolver.rhs_cf = vforce_cf
        ElasticitySolver.ess_bdr = ess_bdr
        ElasticitySolver.SolveState()

        # 7. Save the solution in a grid function.
        u = ElasticitySolver.u
        u = mfem.GridFunction(state_fes)
        u.Assign(ElasticitySolver.u)
        u_data = u.GetDataArray()
        verts = mesh.GetVertexArray()
        elements = mesh.GetElementsArray()

        elements = np.array([mesh.GetElementVertices(i) for i in range(mesh.GetNE())])
        # vertices = gus.Vertices(verts)
        # vertices.vertex_data["u"] = u.reshape(-1,2)
        face_elements = gus.Volumes(verts, elements)
        face_elements.vertex_data["my_data"] = u_data.reshape(-1, 3, order="F")*1000
        face_elements.show_options["arrow_data"] = "my_data"
        face_elements.show_options["lw"] = 3
        self.solution = face_elements

    def show_solution(self):
        gus.show(self.solution)

        # show_solution = False
        # if show_solution:
        #     # 7. Save the solution in a grid function.
        #     u = ElasticitySolver.u
        #     u = mfem.GridFunction(state_fes)
        #     u.Assign(ElasticitySolver.u)
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

        # # 8. Compute the compliance.
        # compliance = ElasticitySolver.clcStrainEnergyDensity()    
        # show_solution = True
        # if show_solution:
        #     SE_data = compliance.GetDataArray()
        #     verts = mesh.GetVertexArray()
        #     elements = mesh.GetElementsArray()

        #     elements = np.array([mesh.GetElementVertices(i) for i in range(mesh.GetNE())])
        #     # vertices = gus.Vertices(verts)
        #     # vertices.vertex_data["u"] = u.reshape(-1,2)
        #     face_elements = gus.Faces(verts, elements)
        #     face_elements.vertex_data["my_data"] = SE_data
        #     face_elements.show_options["data"] = "my_data"
        #     face_elements.show_options["lw"] = 3
        #     face_elements.show_options["cmap"] = "jet"
        #     gus.show(face_elements)
        # raise ValueError("stop here")