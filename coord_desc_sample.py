#
import numpy as np 




def f(x,y):
    return 5*x**2 -6*x*y+5*y**2


def grad_f(x, y):
    dfx = 10*x - 6*y 
    dfy = -6*x + 10*y 

    return np.array([dfx, dfy])
def grad_f(x,y):
    eps = 0.001
    print("T", f(x, y)-f(x+eps, y))
    print("T", (f(x, y)-f(x+eps, y))/eps)
    gx = (f(x, y)-f(x+eps, y))/eps
    gy = (f(x, y+eps)-f(x, y))/eps
    return np.array([gx, gy])

x0 = np.array([-.5, -1])
print("init : ", f(x0[0],x0[1]))
for i in range(20):
    for i in range(len(x0)):
        x0[i] -= 0.01*grad_f(x0[0],x0[1])[i]
        # print(x0)
        print(f(x0[0],x0[1]), "\n","="*10)
print(x0)
