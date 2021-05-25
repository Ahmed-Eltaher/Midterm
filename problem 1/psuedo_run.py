import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename # Open dialog box
import sys
import ccluster

try:
                folder = askopenfilename(filetypes=[("images","*.*")]) 
                img= cv2.imread(folder,cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255
            
                #--------------Lord image file-------------- 

                h,i = img.shape
            
                #--------------Clustering--------------  
                ## Take a cluster object from class ccluster
                cluster = ccluster.ccluster(img,image_bit=h,noclusters=3,fuzziness=2,max_iterations=80,epsilon=sys.float_info.epsilon)
                ## Return the result of the fuzzy clustering on the image to result
                result=cluster.form_clusters()
            
                print("result is",result)
                        
                #-------------------Plot and save result------------------------

                ## Show the result of clustering the chosen picture

                plt.imshow(result,cmap='gray')
                plt.show()


        
except IOError:
    print("Error")
