#!/usr/bin/python3
'''
AUTHOR  :   Zhengyang Zhong
DATE    :   2023-03-16
PAPER   :   Fractional Differentiation-Based Variational Level Set Model for Noisy Image Segmentation without Contour Initialization
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from skimage import measure
from tqdm import tqdm


# The fractional calculus in frequence domain
# Ref: 2007, Jian Bai, Fractional-order Anisotropic Diffusion for Image Denoising
def freq_fractional_calculus(img, alpha=0.5):
    '''
    The data type of the input image should be uint8
    The data type of the returned FOD feature map is also uint8, and the maximum and minimum of the FOD are 255 and 0 respectively.
    '''

    img = img.astype(np.float64)
    height, width = img.shape
    displace1 = (height-1)/2
    displace2 = (width-1)/2
    Fimg = fft.fft2(img)
    fshift = fft.fftshift(Fimg)
    print("fshift real min = ", np.min(np.real(fshift)), ", max = ", np.max(np.real(fshift)))
    print("fshift imge min = ", np.min(np.imag(fshift)), ", max = ", np.max(np.imag(fshift)))

    # fractional derivative terms with centeral difference
    # derivative on y-direction
    pw1 = np.zeros_like(fshift)
    for i in range(height):
        w1 = i - displace1
        pw1[i, :] = np.round((1 - np.exp(-2j * np.pi * w1 / height))**alpha * np.exp( 1j * np.pi * alpha * w1 / height), 5)
    FourierDy = pw1 * fshift
    Dy = np.abs( fft.ifft2( fft.ifftshift( FourierDy ) ) )
    print("Dy min = ", np.min(Dy), ", max = ", np.max(Dy))

    # derivative on x-direction
    pw2 = np.zeros_like(fshift)
    for i in range(width):
        w2 = i - displace2
        pw2[:, i] = np.round((1 - np.exp(-2j * np.pi * w2 / width))**alpha * np.exp( 1j * np.pi * alpha * w2 / width), 5)
    FourierDx = pw2 * fshift
    Dx = np.abs( fft.ifft2( fft.ifftshift( FourierDx ) ) )

    DI = img - np.sqrt(np.square(np.abs(Dx)) + np.square(np.abs(Dy)))
    # DI = np.interp(DI, [np.min(DI), np.max(DI)], [0, 255]).astype(np.uint8)
    return DI


def FVLSmodel(img, fractional_order=0.8, c0=2, beta=0.5, mu=1.0, alpha=1.0, itermax=300, tolerance=1e-4):
    '''
    The data type of the input image should be uint8
    The objects in the image should be brighter than background
    '''

    # Generate Fractional Order Differentiation(FOD) feature map
    DI = freq_fractional_calculus(img, fractional_order).astype(np.float64)

    phi = c0 * np.ones_like(img, dtype=np.float64)
    psi = c0 * np.ones_like(img, dtype=np.float64)
    himg, wimg = img.shape

    img = img.astype(np.float64)
    laplace = np.array([[0, 1, 0], 
                        [1, -8, 1],
                        [0, 1, 0]], dtype=np.float32)

    lapF = fft.fft2(laplace, (himg, wimg))
    print("lap FFT real min", np.min(np.real(lapF)), ", max = ", np.max(np.real(lapF)))
    print("lap FFT imag min", np.min(np.imag(lapF)), ", max = ", np.max(np.imag(lapF)))

    for it in tqdm(range(20)):

        old_Fnorm = np.sqrt(np.sum(phi**2))

        # Neumann condition
        phi[np.ix_([0, -1]), np.ix_([0, -1])]   =   phi[np.ix_([2, -3]), np.ix_([2, -3])]  
        phi[np.ix_([0, -1]), 1:-1]              =   phi[np.ix_([2, -3]), 1:-1]
        phi[1:-1, np.ix_([0, -1])]              =   phi[1:-1, np.ix_([2, -3])]

        phi_p1q = (phi + 1)**2
        phi_n1q = (phi - 1)**2

        # c1 = np.sum(img * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        # c2 = np.sum(img * phi_n1q) / (np.sum(phi_n1q) + 1e-10)
        # c12 = c1 + c2

        m1 = np.sum(DI * phi_p1q) / (np.sum(phi_p1q) + 1e-10)
        m2 = np.sum(DI * phi_n1q) / (np.sum(phi_n1q) + 1e-10)
        m12 = m1 + m2
        print("m1 = ", m1, ", m2 = ", m2)

        # update phi
        # phi_A = mu * psi + beta * (2*img - c12)*(c1 - c2) + (2*DI - m12)*(m1 - m2)
        # phi_B = mu + beta * (2*img**2 - 2*img*c12 + c1**2 + c2**2) + 2*DI**2 - 2*DI*m12 + m1**2 + m2**2
        phi_A = mu * psi + (2*DI - m12)*(m1 - m2)
        phi_B = mu + 2*DI**2 - 2*DI*m12 + m1**2 + m2**2
        phi = phi_A / (phi_B + 1e-10)
        print("phi_A min = ", np.min(phi_A), ", max = ", np.max(phi_A))
        print("phi_B min = ", np.min(phi_B), ", max = ", np.max(phi_B))
        print("phi min = ", np.min(phi), ", max = ", np.max(phi))
        
        # update psi
        phiF = fft.fft2(phi)
        # print("\n")
        # print("\nphi min = ", np.min(phi), ", max = ", np.max(phi))
        # print("\nphiF FFT real min", np.min(np.real(phiF)), ", max = ", np.max(np.real(phiF)))
        # print("\nphiF FFT imag min", np.min(np.imag(phiF)), ", max = ", np.max(np.imag(phiF)))
        psi_A = mu * phiF
        psi_B = mu - alpha * lapF
        # print("\npsi_A FFT real min", np.min(np.real(psi_A)), ", max = ", np.max(np.real(psi_A)))
        # print("\npsi_A FFT imag min", np.min(np.imag(psi_A)), ", max = ", np.max(np.imag(psi_A)))
        # print("\npsi_B FFT real min", np.min(np.real(psi_B)), ", max = ", np.max(np.real(psi_B)))
        # print("\npsi_B FFT imag min", np.min(np.imag(psi_B)), ", max = ", np.max(np.imag(psi_B)))
        tmp = psi_A / (psi_B + 1e-10)
        psii = fft.ifft2(tmp)
        # print("\nIFFT psi_AdB real part, min = ", np.min(np.real(psii)), ", max = ", np.max(np.real(psii)))
        # print("\nIFFT psi_AdB image part, min = ", np.min(np.imag(psii)), ", max = ", np.max(np.imag(psii)))
        psi = np.abs(psii)
        # print("\nIFFT abs min = ", np.min(psi), ", max = ", np.max(psi))

        # terimination
        new_Fnorm = np.sqrt(np.sum(phi**2))
        criteria_Fnorm = np.abs(old_Fnorm - new_Fnorm) / (old_Fnorm + 1e-10)
        if criteria_Fnorm < tolerance:
            break

        if it % 2 == 0:
            img0 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            tmp = np.zeros_like(phi, np.uint8)
            tmp[phi < 0] = 255
            cnts, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img0, cnts, -1, (0, 0, 255), 1)
            cv2.imwrite("./iterprocess/"+str(it+1)+".png", img0)

    # phi = drlse(img, phi)
    return phi, it+1


if __name__ == "__main__":
    path = "D:/pyAC/data/filtered_data0005.png"
    img = cv2.imread(path, 0)
    # img = img.astype(float)
    # img = 255*(img-img.min())/(img.max()-img.min())
    img.astype(np.uint8)
    h, w = img.shape
    phi = 2 * np.ones((h, w))
    phi, iterations = FVLSmodel(img,beta=1,alpha=4,mu=0.2,tolerance=1e-8)
    print(iterations)
    contours = measure.find_contours(phi, 0)
    plt.imshow(phi,cmap='gray')
    for cnt in contours:
        plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
    plt.show(block=True)
