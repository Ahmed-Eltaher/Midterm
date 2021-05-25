import sys
import numpy as np


class ccluster:
    def __init__(self,image,image_bit,noclusters=2,fuzziness=2,max_iterations=150,epsilon=sys.float_info.epsilon):
        ##Give initial values to the parameters needed for the clustering

        self.noclus=noclusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations        
        self.image_bit = image_bit
        self.result=image
        self.shape = image.shape 
        self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1) 
        self.numPixels = image.size
        self.epsilon=epsilon
       
        
        #-------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if noclusters <= 0 or noclusters != int(noclusters):
            raise Exception("<noclusters> needs to be positive integer.")
        if fuzziness < 1:
            raise Exception("<m> needs to be >= 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")


    def initializeMembershipMatrix(self):
        ## function that itializes the memberships
        n=self.numPixels
        nn=self.noclus
        membership_mat = np.random.random((nn, n))
        ## Intializes the omegas with a random values of random floats 
        value = sum(membership_mat)
        membership_mat=np.divide(membership_mat,np.dot(np.ones((nn,1)),np.reshape(value,(1,n))))
        ## Normalize the omega values

        
        

        return membership_mat

    def update_membership(self):
        ## Function that updates the membership weights based on the update of the clustering centeroids
        # function will be called iteratively
        '''Compute weights'''
        c_mesh,idx_mesh = np.meshgrid(self.noclus,self.X)
        m=self.fuzziness
        power = 2./(m-1) ## Fuzzy value power factor
        distance = abs(idx_mesh-c_mesh) ##compute the distance between the centers and the data values
        p1 = distance**power ## computation of the numerator of the weight update using the fuzzy c-means formula 
        p2 = np.sum((1./distance)**power,axis=1) ## computation of the denomantor ....
        
        return 1./(p1*p2[:,None]) ##return updated value

    def update_clusters(self):
        
        ## function that computes the centroids of the clusters
        denominator = np.sum(self.U**self.fuzziness,axis=0)## .... ^fuzzy number
        numerator = np.dot(self.X,self.U**self.fuzziness) ## the summation of weights[i,j]^fuzzy number * xi 
        
        return numerator/denominator

    def form_clusters(self):      
    # Function that loops and form clusters at each iteration while optimizing the error function

        d = 100
        self.U = np.transpose(self.initializeMembershipMatrix())
        ## We want to make the rows as the data and columns as number of clusters
        if self.max_iterations != -1:
            i = 0
            while True: 
            #Loop that updates the clusters and the membership values untill the optimization function is minimum or the iterations has rached a certain maximum value            
                self.noclus = self.update_clusters()
                old_u = np.copy(self.U)
                self.U = self.update_membership()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iterations:
                    break
                i+=1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_clusters()
                old_u = np.copy(self.U)
                self.U = self.update_membership()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iterations:
                    break
                i+=1
        return self.segmentImage()


    def deFuzzify(self):
        ## to turn the float of the cluster into binary numbers with number of clusters categorization 
        return np.argmax(self.U, axis = 1)

    def segmentImage(self):
        '''Segment image based on max weights'''
        ## defuzzify the weights and reshape for visualization
        result = self.deFuzzify()
        self.result = result.reshape(self.shape).astype('float')

        return self.result
