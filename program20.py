"""
Spectral methods in MATLAB. Lloyd
Program 20
"""

# 2nd order wave eq. in 2D via FFT (compare program 19)

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import interp2d
from numpy.linalg import matrix_power

def solve_and_plot(sp_diff, N, partial_t):
    # Grid and initital data
    # N = 24
    x = cos(pi*arange(0,N+1)/N)
    y = x.T
    dt = 6.0/N**2
    [xx,yy] = meshgrid(x,y)
    plotgap = int(round((1./3)/dt))
    dt = (1/3)/plotgap * 1e-15
    dt *= .6
    vv = exp(-xx ** 2)
    vvold = vv

    # Time-stepping by leap frog formula:
    fig = plt.figure()
    plt.title(partial_t+f" dt_ex = {dt:.2e}")
    for n in range(0,3*plotgap+1):
        t = n*dt
        if (((n+0.5)%plotgap) < 1):        #plots at multiplies of t = 1/3
            i = int(float(n)/plotgap) + 1
            ar = arange(-1,1+1./16,1./16)
            [rr, θθ] = meshgrid(ar,ar)
            xxx = rr * cos(θθ)
            yyy = rr * sin(θθ)
            vvv = interp2d(x, y, vv, kind='cubic')
            ax = fig.add_subplot(2,2,i, projection='3d')
            ax.plot_surface(xxx, yyy, vvv(ar,ar), rstride=1, cstride=1, cmap="coolwarm", alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_zlim(-.7, .7)
            ax.text2D(0.05, 0.95, f'{t = :.2e}', transform=ax.transAxes)



        Δvv = sp_diff(N, vv, x, y)
        vvold, vv = time_step(vvold, vv, Δvv, dt, partial_t)
    plt.savefig("figs-fft/"+partial_t+".jpg")
    plt.show()


def fft2D(N, vv, x, y):
    urr = zeros((N+1,N+1))
    r_1ur = zeros((N+1,N+1))
    r_2uθθ = zeros((N+1,N+1))
    ii = arange(1,N)
    for i in range(1,N+1):          # 2nd derivs wrt x in each row
        v = vv[i,:]
        V = hstack([v,flipud(v[ii])])
        U = real(fft.fft(V))
        W1 = real(fft.ifft(1j*hstack([arange(0,N), 0, arange(1-N,0)]).T*U))     # diff wrt theta
        W2 = real(fft.ifft(-hstack([arange(0,N+1), arange(1-N,0)]).T**2*U))       # diff**2 wrt theta
        urr[i,ii] = W2[ii]/(1-x[ii]**2) - x[ii]*W1[ii]/(1-x[ii]**2)**(3./2)
        r_1ur[i,ii] = (1/x[ii])*-W1[ii]/(1-x[ii]**2)**(1/2)
    for k in range(1,N+1):
        v = vv[:,k]
        V = hstack([v,flipud(v[ii])])
        U = real(fft.fft(V))
        W1 = real(fft.ifft(1j*hstack([arange(0,N), 0, arange(1-N,0)]).T*U))     # diff wrt theta
        W2 = real(fft.ifft(-hstack([arange(0,N+1), arange(1-N,0)]).T**2*U))       # diff**2 wrt theta
        r_2uθθ[ii,k] = (1/y[ii]**2)*W2[ii]
    return urr + r_1ur + r_2uθθ

def time_step(vvold, vv, Δvv, dt, method):
    # dt_im = dt * 60
    dt_ex = dt
    # dt_ex = dt * 1e-15
    # dt_ex *= .6

    if method == "leap frog":
        vvnew = vvold + Δvv * dt_ex * 2
        vvold = vv
        vv = vvnew

    if method == "forward Euler":
        vv += dt_ex * Δvv

    return vvold,vv


def main():
    N = 24
    %time solve_and_plot(fft2D, N, "forward Euler")
    %time solve_and_plot(fft2D, N, "leap frog")

    
%time main()

