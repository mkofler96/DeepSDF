'''
   PyMFEM example 37

   See c++ version in the MFEM library for more detail

   Sample runs:
       python ex37.py -alpha 10
       python ex37.py -alpha 10 -pv
       python ex37.py -lambda 0.1 -mu 0.1
       python ex37.py -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
       python ex37.py -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5


   Description: This example code demonstrates the use of MFEM to solve a
                density-filtered [3] topology optimization problem. The
                objective is to minimize the compliance

                    minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)

                    subject to

                      -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
                      -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
                      0 ≤ ρ ≤ 1                 in Ω
                      ∫_Ω ρ dx = θ vol(Ω)

                Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
                penalization (SIMP) law, C is the elasticity tensor for an
                isotropic linearly elastic material, ϵ > 0 is the design
                length scale, and 0 < θ < 1 is the volume fraction.

                The problem is discretized and gradients are computing using
                finite elements [1]. The design is optimized using an entropic
                mirror descent algorithm introduced by Keith and Surowiec [2]
                that is tailored to the bound constraint 0 ≤ ρ ≤ 1.

                This example highlights the ability of MFEM to deliver high-
                order solutions to inverse design problems and showcases how
                to set up and solve PDE-constrained optimization problems
                using the so-called reduced space approach.

   [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
       (2011). Efficient topology optimization in MATLAB using 88 lines of
       code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
   [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
       preserving finite element method for pointwise bound constraints.
       arXiv:2307.12444 [math.NA]
   [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
       based on Helmholtz‐type differential equations. International Journal
       for Numerical Methods in Engineering, 86(6), 765-781.

'''
import mfem.ser as mfem
from numpy import sqrt, log, exp
import numpy as np
from numba import njit
from numba.types import float64

from mfem import mfem_mode

'''
   PyMFEM example 37 - Serial/Parallel Shared code
'''



if mfem_mode == 'serial':
    import mfem.ser as mfem
    use_parallel = False
else:
    import mfem.par as mfem
    use_parallel = True


@njit(float64(float64))
def inv_sigmoid(x):
    '''
    Inverse sigmoid function
    '''
    tol = 1e-12
    x = min(max(tol, x), 1.0-tol)
    return log(x/(1.0-x))


@njit(float64(float64))
def sigmoid(x):
    '''
    Sigmoid function
    '''
    if x >= 0:
        return 1.0/(1.0 + exp(-x))
    else:
        return exp(x)/(1.0 + exp(x))


@njit(float64(float64))
def der_sigmoid(x):
    '''
    Derivative of sigmoid function    
    '''
    tmp = sigmoid(-x)
    return tmp - tmp**2


def MappedGridFunctionCoefficient(gf, func):

    c_gf = mfem.GridFunctionCoefficient(gf)

    @mfem.jit.scalar(dependency=(c_gf,))
    def coeff(ptx, c_gf):
        return func(c_gf)
    return coeff


def DiffMappedGridFunctionCoefficient(gf, other_gf, func, comp=1):
    c_gf = mfem.GridFunctionCoefficient(gf)
    c_ogf = mfem.GridFunctionCoefficient(other_gf)

    @mfem.jit.scalar(dependency=(c_gf, c_ogf))
    def coeff(ptx, c_gf, c_ogf):
        return func(c_gf) - func(c_ogf)
    return coeff


class SIMPInterpolationCoefficient():
    '''
    Python Note: 
       Approach here is to replace Eval in C++ example using the dependency 
       feature of mfem.jit.

       In order to avoid repeating Numba-Jitting in iteration loop, we use
       SetGridFunction to update the GridFunction referred from 
       GridFunctionCoefficient.
    '''

    def __init__(self, rho_filter, min_val=1e-6, max_val=1.0, exponent=3):
        val = mfem.GridFunctionCoefficient(rho_filter)

        @mfem.jit.scalar(dependency=(val,))
        def coeff(ptx, val):
            coeff = min_val + val**exponent*(max_val-min_val)
            return coeff

        self.c_gf = val
        self.coeff = coeff

    def Update(self, rho_filter):
        self.c_gf.SetGridFunction(rho_filter)

def VolumeForceCoefficient(r, center, force):

    @mfem.jit.vector(shape=(len(center),))
    def coeff(ptx):
        cr = sqrt(sum((ptx - center)**2))
        if cr < r:
            return np.array((force[0], force[1]))
        else:
            return np.array((0.0, 0.0))
    return coeff

def VolumeForceCoefficient3D(r, center, force):

    @mfem.jit.vector(shape=(len(center),))
    def coeff(ptx):
        cr = sqrt(sum((ptx - center)**2))
        if cr < r:
            return np.array((force[0], force[1], force[2]))
        else:
            return np.array((0.0, 0.0, 0.0))
    return coeff

def SurfaceForceCoefficient3D(xmin, xmax, zmin, force):

    @mfem.jit.vector(shape=(3,))
    def coeff(ptx):
        if (ptx[0] > xmin) and (ptx[0] < xmax) and (ptx[2] > zmin):
            return np.array((force[0], force[1], force[2]))
        else:
            return np.array((0.0, 0.0, 0.0))
    return coeff

class StrainEnergyDensityCoefficient2D():
    '''
    Python Note: 
       Approach here is to replace Eval and GetVectorGradient method call in C++
       using the dependency feature of mfem.jit.

       GetVectorGradient is mimiced by creating GradientGridFunctionCoefficient
       for each component of u vector. Note GridFunction(fes, u.GetDataArray())
       reuses the data array from u.
    '''

    def __init__(self, llambda, mu, u):

        fes = u.FESpace()
        assert fes.GetOrdering() == mfem.Ordering.byNODES, "u has to use byNODES ordering"

        mesh = fes.GetMesh()
        dim = mesh.Dimension()
        assert dim == 2, "dim must be two."

        fec = fes.FEColl()
        fes = mfem.FiniteElementSpace(mesh, fec)
        size = len(u.GetDataArray())

        u1 = mfem.GridFunction(fes, mfem.Vector(
            u.GetDataArray()))   # first component
        u2 = mfem.GridFunction(fes, mfem.Vector(
            u.GetDataArray()), size//2)  # second component


        c_gradu1 = mfem.GradientGridFunctionCoefficient(u1)
        c_gradu2 = mfem.GradientGridFunctionCoefficient(u2)


        @mfem.jit.scalar(dependency=(llambda, mu, c_gradu1, c_gradu2))
        def coeff(ptx, L, M, grad1, grad2):
            div_u = grad1[0] + grad2[1]
            density = L*div_u*div_u

            grad = np.zeros(shape=(2, 2), dtype=np.float64)
            grad[0, 0] = grad1[0]
            grad[0, 1] = grad1[1]
            grad[1, 0] = grad2[0]
            grad[1, 1] = grad2[1]

            for i in range(2):
                for j in range(2):
                    density += M*grad[i, j]*(grad[i, j] + grad[j, i])
            return density

        self.fes = fes
        self.size = size
        self.u1u2 = (u1, u2)
        self.dependency = (c_gradu1, c_gradu2)
        self.coeff = coeff


class StrainEnergyDensityCoefficient():
    '''
    Python Note: 
       Approach here is to replace Eval and GetVectorGradient method call in C++
       using the dependency feature of mfem.jit.

       GetVectorGradient is mimiced by creating GradientGridFunctionCoefficient
       for each component of u vector. Note GridFunction(fes, u.GetDataArray())
       reuses the data array from u.
    '''

    def __init__(self, llambda, mu, u):

        fes = u.FESpace()
        assert fes.GetOrdering() == mfem.Ordering.byNODES, "u has to use byNODES ordering"

        mesh = fes.GetMesh()
        
        fec = fes.FEColl()
        fes = mfem.FiniteElementSpace(mesh, fec)
        size = len(u.GetDataArray())
        u_data_single_vec = u.GetDataArray().copy()
        u_data = u_data_single_vec.reshape(-1, 3, order="F")
        u1 = mfem.GridFunction(fes, mfem.Vector(
            u_data[:,0]))   # first component
        u2 = mfem.GridFunction(fes, mfem.Vector(
            u_data[:,1]))   # second component
        u3 = mfem.GridFunction(fes, mfem.Vector(
            u_data[:,2]))   # third component


        c_gradu1 = mfem.GradientGridFunctionCoefficient(u1)
        c_gradu2 = mfem.GradientGridFunctionCoefficient(u2)
        c_gradu3 = mfem.GradientGridFunctionCoefficient(u3)

        @mfem.jit.scalar(dependency=(llambda, mu, c_gradu1, c_gradu2, c_gradu3))
        def coeff(ptx, L, M, grad1, grad2, grad3):
            div_u = grad1[0] + grad2[1] + grad3[2]
            density = L*div_u*div_u

            grad = np.zeros(shape=(3, 3), dtype=np.float64)
            grad[0, 0] = grad1[0]
            grad[0, 1] = grad1[1]
            grad[0, 2] = grad1[2]
            grad[1, 0] = grad2[0]
            grad[1, 1] = grad2[1]
            grad[1, 2] = grad2[2]
            grad[2, 0] = grad3[0]
            grad[2, 1] = grad3[1]
            grad[2, 2] = grad3[2]

            for i in range(3):
                for j in range(3):
                    density += M*grad[i, j]*(grad[i, j] + grad[j, i])
            return density

        self.fes = fes
        self.size = size
        self.u1u2u3 = (u1, u2, u3)
        self.dependency = (c_gradu1, c_gradu2, c_gradu3)
        self.coeff = coeff

class LinearElasticitySolver():
    def __init__(self, lambda_cf, mu_cf):
        self.lambda_cf = lambda_cf
        self.mu_cf = mu_cf
        self.rhs_cf = None
        self.surface_load = None
        self.essbdr_cf = None
        self.nm_bdr = None
        self.ess_bdr = None

    def SetupFEM(self):
        dim = self.mesh.Dimension()
        self.fec = mfem.H1_FECollection(
            self.order, dim, mfem.BasisType.Positive)
        self.fes = mfem.FiniteElementSpace(self.mesh, self.fec, dim)
        self.SE_fes = mfem.FiniteElementSpace(self.mesh, self.fec, 1)
        self.SE = mfem.GridFunction(self.SE_fes)
        self.u = mfem.GridFunction(self.fes)
        self.u.Assign(0.0)

    def SolveState(self):
        A = mfem.OperatorPtr()
        B = mfem.Vector()
        X = mfem.Vector()

        ess_tdof_list = mfem.intArray()
        self.fes.GetEssentialTrueDofs(self.ess_bdr, ess_tdof_list)

        x = mfem.GridFunction(self.fes)
        x .Assign(0.0)

        self.u.Assign(0.0)
        b = mfem.LinearForm(self.fes)

        if self.rhs_cf is not None:
            b.AddDomainIntegrator(mfem.VectorDomainLFIntegrator(self.rhs_cf))
        if self.surface_load is not None:
            b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(self.surface_load), self.nm_bdr)
        b.Assemble()

        a = mfem.BilinearForm(self.fes)
        a.AddDomainIntegrator(
            mfem.ElasticityIntegrator(self.lambda_cf, self.mu_cf))
        a.Assemble()
        # if self.essbdr_cf is not None:
        #     u.ProjectBdrCoefficient(self.essbdr_cf, self.ess_bdr)

        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        AA = A.AsSparseMatrix()
        M = mfem.GSSmoother(AA)
        cg = mfem.CGSolver()

        cg.SetRelTol(1e-10)
        cg.SetMaxIter(10000)
        cg.SetPrintLevel(0)
        cg.SetPreconditioner(M)
        cg.SetOperator(A)
        cg.Mult(B, X)
        a.RecoverFEMSolution(X, b, x)

        self.u += x
        self.b = b

    def GetFEMSolution(self):
        return self.u

    def GetLinearForm(self):
        return self.b

    def clcStrainEnergyDensity(self):
        self.StrainEnergyDensity = StrainEnergyDensityCoefficient(self.lambda_cf, self.mu_cf, self.u)
        self.SE.ProjectCoefficient(self.StrainEnergyDensity.coeff)
        return self.SE

    def clcComplianceShapeDerivative(self, theta_discrete):
        """
        calculates the shape derivative according to section 6.3 in 
        Allaire, G., Dapogny, C. & Jouve, F. Shape and topology optimization. 
        in Geometric partial differential equations, part II (eds. Bonito, A. & Nochetto, R. H.) vol. 22 (2021).
        """
        if self.StrainEnergyDensity is None:
            self.clcStrainEnergyDensity()
        fec = mfem.H1_FECollection(self.order, 3)
        fe_scalar_space = mfem.FiniteElementSpace(self.mesh, fec, 1)
        fe_physical_space = mfem.FiniteElementSpace(self.mesh, fec, 3)
        
        # coeff_one = mfem.ConstantCoefficient(1.0)
        # vol_integrator = mfem.DomainLFIntegrator(coeff_one)
        # theta_gf = mfem.GridFunction(fe_physical_space)
        SE_theta_discrete = -self.SE.GetDataArray().reshape(-1,1)*theta_discrete
        SE_theta_gf = mfem.GridFunction(fe_physical_space, mfem.Vector(
            SE_theta_discrete.reshape(-1, order="F")))
        # print(theta_gf.GetDataArray())
        SE_theta = mfem.VectorGridFunctionCoefficient(SE_theta_gf)
        sed_integrator = mfem.BoundaryNormalLFIntegrator(SE_theta)
        
        b = mfem.LinearForm(fe_scalar_space)
        b.AddBoundaryIntegrator(sed_integrator)
        b.Assemble()
        # print(b.GetDataArray())
        return b.Sum()

    def clcVolume(self):
        b = mfem.LinearForm(self.fes)
        # print mesh stats
        mesh = self.fes.mesh

        # coeff_one = mfem.ConstantCoefficient(1.0)
        # vol_integrator = mfem.DomainLFIntegrator(coeff_one)
        # # oder: sed_integrator = mfem.BoundaryNormalLFIntegrator(self.StrainEnergyDensity.coeff, theta)
        # b.AddDomainIntegrator(vol_integrator)
        # b.Assemble()
        # vol = b.Sum()
        # del b
        # for i in range(len(mesh.Get))
        vol = 0
        for i in range(mesh.GetNE()):
            vol += mesh.GetElementVolume(i)
        return vol

    def clcVolumeShapeDerivative(self, theta_discrete):
        fec = mfem.H1_FECollection(self.order, 3)
        fe_scalar_space = mfem.FiniteElementSpace(self.mesh, fec, 1)
        fe_physical_space = mfem.FiniteElementSpace(self.mesh, fec, 3)
        
        # coeff_one = mfem.ConstantCoefficient(1.0)
        # vol_integrator = mfem.DomainLFIntegrator(coeff_one)
        # theta_gf = mfem.GridFunction(fe_physical_space)
        theta_gf = mfem.GridFunction(fe_physical_space, mfem.Vector(
            theta_discrete.reshape(-1, order="F")))
        # print(theta_gf.GetDataArray())
        theta = mfem.VectorGridFunctionCoefficient(theta_gf)
        
        sed_integrator = mfem.BoundaryNormalLFIntegrator(theta)

        b = mfem.LinearForm(fe_scalar_space)
        b.AddBoundaryIntegrator(sed_integrator)
        b.Assemble()
        # print(b.GetDataArray())
        return b.Sum()

class Proj():
    '''

    @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
          ∫_Ω ρ dx = θ vol(Ω) as follows:

          1. Compute the root of the R → R function
              f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
          2. Set ψ ← ψ + c.

    @param psi a GridFunction to be updated
    @param target_volume θ vol(Ω)
    @param tol Newton iteration tolerance
    @param max_its Newton maximum iteration number
    @return double Final volume, ∫_Ω sigmoid(ψ)

    '''

    def __init__(self, psi):
        self.psi = psi
        self.sigmoid_psi = MappedGridFunctionCoefficient(psi, sigmoid)
        self.der_sigmoid_psi = MappedGridFunctionCoefficient(psi, der_sigmoid)

    def __call__(self, target_volume, tol=1e-12, max_its=10):
        psi = self.psi
        int_sigmoid_psi = mfem.LinearForm(psi.FESpace())
        int_sigmoid_psi.AddDomainIntegrator(
            mfem.DomainLFIntegrator(self.sigmoid_psi))
        int_der_sigmoid_psi = mfem.LinearForm(psi.FESpace())
        int_der_sigmoid_psi.AddDomainIntegrator(mfem.DomainLFIntegrator(
            self.der_sigmoid_psi))
        done = False
        for k in range(max_its):  # Newton iteration
            int_sigmoid_psi.Assemble()         # Recompute f(c) with updated ψ
            f = int_sigmoid_psi.Sum() - target_volume

            int_der_sigmoid_psi.Assemble()      # Recompute df(c) with updated ψ
            df = int_der_sigmoid_psi.Sum()

            dc = -f/df
            psi += dc
            if abs(dc) < tol:
                done = True
                break

        if not done:
            message = ("Projection reached maximum iteration without converging. " +
                       "Result may not be accurate.")
            import warnings
            warnings.warn(message, RuntimeWarning)

        int_sigmoid_psi.Assemble()
        return int_sigmoid_psi.Sum()