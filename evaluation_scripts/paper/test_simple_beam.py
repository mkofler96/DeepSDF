import mfem.ser as mfem
import numpy as np
n=20
mesh = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON, sz=2)
mfem.Mesh.Save(mesh, "mfem_export.mesh")
order = 1
fec = mfem.H1_FECollection(order,  mesh.Dimension())
fespace = mfem.FiniteElementSpace(mesh, fec, mesh.Dimension())
fespace1 = mfem.FiniteElementSpace(mesh, fec, 1)
b = mfem.LinearForm(fespace)
u = mfem.GridFunction(fespace)
u.Assign(0.0)

def SurfaceForceCoefficient3D(xmin, xmax, zmin, force):
    @mfem.jit.vector(shape=(3,))
    def coeff(ptx):
        if (ptx[0] > xmin) and (ptx[0] < xmax) and (ptx[2] > zmin):
            return np.array((force[0], force[1], force[2]))
        else:
            return np.array((0.0, 0.0, 0.0))
    return coeff

def LinearSurfaceForceCoefficient3D():
    @mfem.jit.vector(shape=(3,))
    def coeff(ptx):
        # ptx[0] = x
        # ptx[1] = y
        # ptx[2] = z
        k = 10
        return np.array((0.0, 0.0, k*ptx[0]))
    return coeff

A = mfem.OperatorPtr()
B = mfem.Vector()
X = mfem.Vector()

maxat = mesh.bdr_attributes.Max()
print(maxat)
ess_bdr = mfem.intArray([0]*maxat)
ess_bdr[0] = 1

nm_bdr = mfem.intArray([0]*maxat)
nm_bdr[5] = 1
ess_tdof_list = mfem.intArray()
fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

x = mfem.GridFunction(fespace)
x .Assign(0.0)

xmin = 0.9
xmax = 1.0
zmin = 0.0
force = np.array((1.0, 0.0, 0.0))
const_vec = mfem.Vector(3)  # A 2D vector
const_vec.Assign((0.0, 0.0, 1.0))  # Set the vector components, e.g., (1.0, 2.0)

# Create a constant vector coefficient from the defined vector.
surface_load = mfem.VectorConstantCoefficient(const_vec)
# surface_load = SurfaceForceCoefficient3D(xmin, xmax, zmin, force)
b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(surface_load), nm_bdr)
b.Assemble()
print("B after Assembly")
print(b.GetDataArray())
a = mfem.BilinearForm(fespace)
llambda = 0
mu = 1
lambda_cf = mfem.ConstantCoefficient(llambda)
mu_cf = mfem.ConstantCoefficient(mu)
a.AddDomainIntegrator(mfem.ElasticityIntegrator(lambda_cf, mu_cf))
a.Assemble()
# if self.essbdr_cf is not None:
#     u.ProjectBdrCoefficient(self.essbdr_cf, self.ess_bdr)

a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)


AA = A.AsSparseMatrix()
AA_np = AA.GetDataArray()
M = mfem.GSSmoother(AA)
cg = mfem.CGSolver()

cg.SetRelTol(1e-10)
cg.SetMaxIter(10000)
cg.SetPrintLevel(1)
cg.SetPreconditioner(M)
cg.SetOperator(A)
cg.Mult(B, X)
a.RecoverFEMSolution(X, b, x)

u = x

# calculate strain density
class StrainEnergyDensityCoefficient(mfem.PyCoefficientBase):
    def __init__(self, lambda_, mu_, u):
        super(StrainEnergyDensityCoefficient, self).__init__(0)
        self.lam = lambda_   # coefficient
        self.mu = mu_       # coefficient
        self.u = u   # displacement GridFunction
        self.grad = mfem.DenseMatrix()

    def SetComponent(self, i, j):
        self.si = i
        self.sj = j

    def SetDisplacement(self, u):
        self.u = u

    def Eval(self, T, ip):
        L = self.lam.Eval(T, ip)
        M = self.mu.Eval(T, ip)
        self.u.GetVectorGradient(T, self.grad)
        div_u = self.grad.Trace()
        density = L*div_u*div_u
        dim = T.GetSpaceDim()
        for i in range(dim):
            for j in range(dim):
                density += M*self.grad[i,j]*(self.grad[i,j]+self.grad[j,i]);

        return density


# SE_coeff = StrainEnergyDensityCoefficient(lambda_cf, mu_cf, u)
# strain_energy = mfem.GridFunction(fespace1)
# strain_energy.ProjectCoefficient(SE_coeff.coeff)
# print(strain_energy.GetDataArray())
SE_coeff = StrainEnergyDensityCoefficient(lambda_cf, mu_cf, u)
strain_energy = mfem.GridFunction(fespace1)
strain_energy.ProjectCoefficient(SE_coeff)
print(strain_energy.GetDataArray())
# print(strain_energy.Get)

# raise NotImplementedError("stop it")
paraview_dc = mfem.ParaViewDataCollection("linear_elasticity", mesh)

paraview_dc.SetPrefixPath("ParaView")
paraview_dc.SetLevelsOfDetail(order)

paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
paraview_dc.SetHighOrderOutput(True)
paraview_dc.SetCycle(0)
paraview_dc.SetTime(0.0)
paraview_dc.RegisterField("displacement", u)
# paraview_dc.RegisterField("strain_energey", strain_energy)
paraview_dc.RegisterField("strain_energy", strain_energy)
paraview_dc.Save()

# print(x.GetDataArray())

u_data_single_vec = u.GetDataArray().copy()
u_data = u_data_single_vec.reshape(-1, 3, order="F")
# print(u_data)
