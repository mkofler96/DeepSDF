import mfem.ser as mfem

mesh = mfem.Mesh.MakeCartesian3D(10, 10, 10, mfem.Element.HEXAHEDRON)
order = 1
fec = mfem.H1_FECollection(order,  mesh.Dimension())
fespace = mfem.FiniteElementSpace(mesh, fec, mesh.Dimension())
fespace1 = mfem.FiniteElementSpace(mesh, fec, 1)
b = mfem.LinearForm(fespace1)

class displacement_vector(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        x[1] = 0
        x[2] = 0
        return x

theta = displacement_vector(3)
bdn_integrator = mfem.BoundaryNormalLFIntegrator(theta)
b.AddBoundaryIntegrator(bdn_integrator)
b.Assemble()
print(b.GetDataArray())
print(b.Sum())

theta_gf = mfem.GridFunction()