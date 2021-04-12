"""
Spectral methods in MATLAB. Lloyd
Program 29
"""

# Solve Poisson equation on the unit disk (compare program 16 and 28)

from numpy import *
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from cheb import cheb
import numpy.polynomial.chebyshev as n_cheb
from mpl_toolkits.mplot3d import axes3d

def matrix2D(N, M):
    'Laplacian in polar coordinates:'

    # N = 31
    N2 = (N - 1) // 2
    [D,x] = cheb(N)

    # annulus
    r_1 = 1 /4
    r_2 = 1

    a = 2 * r_1 / (r_2 - r_1) + 1
    b = 2 / (r_2 - r_1)
    r = (x + a) / b

    D2 = D @ D
    D1 = D2[1:N2 + 1, 1:N2 + 1]
    D2 = D2[1:N2 + 1, N - 1:N2:-1]
    E1 = D[1:N2 + 1, 1:N2 + 1]
    E2 = D[1:N2 + 1, N - 1:N2:-1]

    # \theta<TAB> = θ coordinate, ranging from 0 to 2π (M must be even):
    # M = 40
    dθ = 2 * pi / M
    θ = dθ * arange(1, M + 1)
    M2 = M // 2
    col = hstack([
        -pi**2 / (3 * dθ**2) - 1 / 6,
        0.5 * (-1)**arange(2, M + 1) / sin(dθ * arange(1, M) / 2)**2
    ])
    D2θ = toeplitz(col, col)

    # Laplacian in polar coordinates:
    R = diag(1 / r[1:N2 + 1])
    Z = zeros((M2, M2))
    I = identity(M2)
    L = kron(D1 + R @ E1, identity(M)) + kron(
        D2 + R @ E2, vstack(
            (hstack([Z, I]), hstack([I, Z])))) + kron(R @ R, D2θ)
    return r, θ, L


def time_step(vvold, L, vv, dt, method):
    # Δvv = L @ vv
    dt_im = dt * 60
    dt_ex = dt * 2e-4

    if method == "leap frog":
        vvnew = vvold + (L @ vv) * dt_ex * 2
        vvold = vv
        vv = vvnew

    if method == "forward Euler":
        vv += dt_ex * (L @ vv)

    if method == "backward Euler":
        vv = linalg.solve(1 - dt_im * L, vv)

    if method == "Crank-Nicolson":
        _ = (dt_im / 2) * L  # not sure what to name it
        vv = linalg.solve(1 - _, (1 + _) @ vv)

    return vvold, vv


def solve_and_plot(sp_diff, N, M, partial_t):
    N2 = (N - 1) // 2
    r, θ, L = sp_diff(N, M)

    # Right-hand side and solution for u:
    [rr, θθ] = meshgrid(r[1:N2 + 1], θ)
    rr = hstack(stack(rr[:], axis=-1))
    θθ = hstack(stack(θθ[:], axis=-1))
    f = exp(-rr ** 2)
    u = f
    uold = u

    dt = 6.0/N**2
    plotgap = int(round((1./3)/dt))
    dt = (1./3)/plotgap

    # Time-stepping by leap frog formula:
    fig = plt.figure()
    plt.title(partial_t+f" dt_im = {dt * 60:.2e}, dt_ex = {dt * 2e-4:.2e}")
    
    for n in range(0,3*plotgap+1):
        t = n*dt
        if (((n+0.5)%plotgap) < 1):        #plots at multiplies of t = 1/3
            i = int(float(n)/plotgap) + 1
    # Reshape results onto 2D grid and plot them:
    uu = reshape(u, (N2, M)).T
    u2 = vstack((uu[M - 1, :], uu[0:M - 1, :]))
    uu = hstack([zeros((M, 1)), u2])
    [rr, θθ] = meshgrid(r[0:N2 + 1], hstack([θ[M - 1], θ[0:M - 1]]))
    xx = rr * cos(θθ)
    yy = rr * sin(θθ)
    ax = fig.add_subplot(2,2,i, projection='3d')
    ax.text2D(0.05, 0.95, f'{t = :.2e}', transform=ax.transAxes)
    ax.plot_surface(xx, yy, uu, rstride=1, cstride=1, cmap='coolwarm', alpha=.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim(-.7, .7)

        # Δvv = sp_diff(N, vv, x, y)
        uold, u = time_step(uold, L, u, dt, partial_t)
    plt.savefig("figs-matrix/"+partial_t+".jpg")
    plt.show()
    

def main():
    N = 31
    M = 40
    %time solve_and_plot(matrix2D, *(N, M), "forward Euler")
    %time solve_and_plot(matrix2D, *(N, M), "leap frog")
    %time solve_and_plot(matrix2D, *(N, M), "backward Euler")
    %time solve_and_plot(matrix2D, *(N, M), "Crank-Nicolson")
    # %time solve_and_plot(fft2D, *(N, M), "forward Euler")
    # %time solve_and_plot(fft2D, *(N, M), "leap frog")

    
%time main()

