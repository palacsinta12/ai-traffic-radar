import cv2
img = cv2.imread("measurements/20251021-103512-R_4646-I_700-W_0_ped2/bev_images/2667.png")
print(img.shape) # MUST print: (640, 192, 3)