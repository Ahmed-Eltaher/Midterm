import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename # Open dialog box
import sys
import ccluster
from gui import Ui_MainWindow
from PyQt5 import QtWidgets
from PIL import Image as pi
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
       
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.img = []
        self.ui.browse.clicked.connect(self.browse)
        self.ui.fuzzy.clicked.connect(self.p1)
        
        
    def browse(self):
                folder = askopenfilename(filetypes=[("images","*.*")]) 
                self.img= cv2.imread(folder,cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255
    def p1(self):
        try:

            
                #--------------Lord image file-------------- 

                h,i = self.img.shape
            
                #--------------Clustering--------------  
                ## Take a cluster object from class ccluster
                cluster = ccluster.ccluster(self.img,image_bit=h,noclusters=3,fuzziness=2,max_iterations=80,epsilon=sys.float_info.epsilon)
                ## Return the result of the fuzzy clustering on the image to result
                result=cluster.form_clusters()
            
                print("result is",result)
                        
                #-------------------Plot and save result------------------------

                ## Show the result of clustering the chosen picture

                plt.imshow(result,cmap='gray')
                plt.show()


                
        except IOError:
            print("Error")

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()