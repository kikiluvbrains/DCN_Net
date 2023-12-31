# importing opencv CV2 module 
import cv2 

# fill this with the name of the img you want
img = cv2.imread('house.jpg') 

# make sure that you have saved it in the same folder 
# Averaging 
# You can change the kernel size as you want
kernel = 30
avging = cv2.blur(img,(kernel,kernel)) 
cv2.imwrite("blurred_house" + str(kernel) + ".jpg", avging)

#cv2.imshow('Averaging',avging)
#cv2.waitKey(0) 

# Gaussian Blurring 
# Again, you can change the kernel size 
gausBlur = cv2.GaussianBlur(img, (5,5),0) 
#cv2.imshow('Gaussian Blurring', gausBlur) 
#cv2.waitKey(0) 

# Median blurring 
medBlur = cv2.medianBlur(img,5) 
#cv2.imshow('Media Blurring', medBlur) 
#cv2.waitKey(0) 

# Bilateral Filtering 
bilFilter = cv2.bilateralFilter(img,9,75,75) 
cv2.imshow('Bilateral Filtering', bilFilter) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
