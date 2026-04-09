import cv2
import torch
import numpy as np

device = torch.device("xpu")
x_steps = 1500
y_steps = 1000
iterations = 100

windows = [[-2, 1, -1, 1]]

def mandelbrot(x, c):
    return x ** 2 + c

def getImage():
    x0, x1, y0, y1 = windows[-1]
    x_stepsize = (x1-x0)/x_steps
    y_stepsize = (y1-y0)/y_steps

    real = torch.arange(x0, x1, x_stepsize)
    imag = torch.arange(y0, y1, y_stepsize)

    imag, real = torch.meshgrid(imag, real, indexing="ij")
    base = torch.complex(real, imag).to(device, dtype=torch.complex128)
    x = torch.zeros(base.shape, device=device)
    img = torch.zeros_like(x, device=device)

    i = 0
    while int(img.max() - img.min()) < iterations and i < 1000:
        x = mandelbrot(x, base)
        img += (torch.abs(x) > 2).to(dtype=torch.int8)
        i+=1

    #img = iterations - img
    img_norm = cv2.normalize(img.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)

    colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
    #colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_OCEAN)

    return colored

def mouse_callback(event, x, y, flags, param):
    x0, x1, y0, y1 = windows[-1]

    if event == cv2.EVENT_LBUTTONDOWN:
        x /= x_steps
        y /= y_steps

        x_range = x1 - x0
        y_range = y1 - y0

        x *= x_range
        y *= y_range

        x += x0
        y += y0

        x_range /= 6
        y_range /= 6

        windows.append([x - x_range, x + x_range, y - y_range, y + y_range])
        cv2.imshow("mandelbrot", getImage())

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(windows) == 1:
            return
        windows.remove(windows[-1])
        cv2.imshow("mandelbrot", getImage())


cv2.imshow("mandelbrot", getImage())
cv2.setMouseCallback("mandelbrot", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
