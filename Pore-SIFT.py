#coding: utf-8

#user:rain
#function:

import numpy as np
from scipy import misc
from scipy import ndimage
import pylab as pl
import cv2
from sklearn.decomposition import PCA
from sklearn import preprocessing
class Psift:
    def __init__(self, im, octave_num, sigma1=0.8,conthreshold=1,upSampling = True):
        """

        :param im: image
        :param octave_num: pyramis`s num
        :param sigma: source sigma
        :param contrast: center_value`s threashold
        """
        self.octave_num = octave_num
        self.conthreshold = conthreshold
        self.im = im
        self.sigma = sigma1
        self.octave_im = {}
        self.octave_gauss = {}
        self.cov_im = {}
        self.center_sigma = {}
        self.k = 2.0 ** (1.0 / 6);
        if upSampling == True:
            self.octave_im[0]=misc.imresize(im,[im.shape[0]*2,im.shape[1]*2],interp='bicubic')/255.0
        else:
            self.octave_im[0] = im /255.0
        print ('Image`s shape :',im.shape)

    def CreatePyramid(self):
        """
        Create the HOD Pyramid
        """
        sigma = self.sigma/2.0
        for i in range(self.octave_num):
            sigma = sigma * 2.0
            imt = self.octave_im[i]
            blur_a = ndimage.gaussian_filter(imt, sigma=sigma)
            blur_b = ndimage.gaussian_filter(imt, sigma=self.k * sigma)
            blur_c = ndimage.gaussian_filter(imt, sigma=(self.k ** 2) * sigma)
            blur_d = ndimage.gaussian_filter(imt, sigma=(self.k ** 3) * sigma)
            blur_e = ndimage.gaussian_filter(imt, sigma=(self.k ** 4) * sigma)
            blur_f = ndimage.gaussian_filter(imt, sigma=(self.k ** 5) * sigma)
            blur_g = ndimage.gaussian_filter(imt, sigma=(self.k ** 6) * sigma)


            temp = np.zeros([6, imt.shape[0], imt.shape[1]])
            temp[0] =abs(blur_b - blur_a)
            temp[1] = abs(blur_c - blur_b)
            temp[2] =abs(blur_d - blur_c)
            temp[3] = abs(blur_e - blur_d)
            temp[4] = abs(blur_f - blur_e)
            temp[5] = abs(blur_g - blur_f)

            self.octave_gauss[i, 0] = blur_b
            self.octave_gauss[i, 1] = blur_c
            self.octave_gauss[i, 2] = blur_d
            self.octave_gauss[i, 3] = blur_e


            self.octave_im[i + 1] = cv2.resize(blur_c, (int(blur_c.shape[1] / 2), int(blur_c.shape[0] / 2)))
            self.cov_im[i] = temp
        print ('Create image`s Pyramid successful')

    def ScolePoint(self, rmedge=False, specofic_hd=0, curvature=10.0):

        """
        Finding scole points 
        :param rmedge: remove the boreder point
        :param specofic_hd: first min / second min `s threashold
        :param curvature: eliminate border point
        :return: x,y
        """
        print ('Start find scole points')
        point_list = []
        find_point = 0
        after_re_point = 0
        R = (curvature + 1) ** 2 / (curvature * 1.0)
        for i in range(self.octave_num):
            for j in np.arange(1,5):
                imt = self.cov_im[i][j-1:j+2]
                [l, m, n] = imt.shape
                lessen = 1.0 / (imt.shape[1] * 1.0 / self.im.shape[0])
                for x in np.arange(10, m - 10):
                    for y in np.arange(10, n - 10):
                        tmp_patch = np.array(imt[:, (x - 1):(x + 2), (y - 1):(y + 2)])  # patch
                        center_value = tmp_patch[1, 1, 1]  # center value
                        # Find fist
                        first_max = np.max(tmp_patch.ravel())
                            # threashold
                        if center_value == first_max and center_value >specofic_hd:
                            find_point += 1
                            #detect the curvature whether bigger than (r+1)^2/r^2
                            if rmedge == True and center_value < self.conthreshold:

                                center_im = np.array(tmp_patch[1, :, :])

                                imx = np.zeros(center_im.shape)
                                ndimage.sobel(center_im, axis=1, output=imx)
                                imy = np.zeros(center_im.shape)
                                ndimage.sobel(center_im, axis=0, output=imy)

                                Wxx = ndimage.gaussian_filter(imx * imx, 1)
                                Wxy = ndimage.gaussian_filter(imx * imy, 1)
                                Wyy = ndimage.gaussian_filter(imy * imy, 1)
                                Wdet = Wxx * Wyy - Wxy ** 2
                                Wtr = Wxx + Wyy
                                det = (Wtr ** 2 / (Wdet+1e-8))[1, 1]
                                # print det ,R
                                # 用曲率和比值消除边缘点
                                if det < R :
                                    after_re_point += 1  # 计数
                                    point_list.append([int(x), int(y), i,j-1, lessen])
                            else:
                                point_list.append([int(x), int(y), i,j-1, lessen])

        print ('Find source point:', find_point,'   after eliminate edge point:', after_re_point,'\n')
        return np.array(point_list)

    def GetPCAFeature(self,point_list):
        print('Start get feature(vector with size (512))')


        x = point_list[:, 0]
        y = point_list[:, 1]
        octave_num = point_list[:, 2]
        inv_octave_num = point_list[:, 3]
        lessen = point_list[:, 4]
        feature_matrix = []
        point_weight = []
        point_list = []
        lens = np.size(x, axis=0)
        pi2 = 2 * 3.1415
        li = np.int(20)

        # gauss = (np.dot(cv2.getGaussianKernel(li * 2 + 1, 10), cv2.getGaussianKernel(li * 2 + 1, 10).T) * 1e4).reshape(-1)/10.0
        for i in range(lens):
            if i % 1000 == 0:
                print('Has got', i, 'features')
            row = np.int(x[i])
            col = np.int(y[i])
            try:
                gradient_im = np.array(
                    self.octave_gauss[octave_num[i], inv_octave_num[i]][row - li:row + li + 1, col - li:col + li + 1])
                assert gradient_im.shape[0] == li * 2 + 1
                assert gradient_im.shape[1] == li * 2 + 1
            except:
                # print 'Border point'
                continue
            y_gradient = ndimage.sobel(gradient_im, axis=0).reshape(-1)
            x_gradient = ndimage.sobel(gradient_im, axis=1).reshape(-1)
            tan_garadient = np.arctan2(y_gradient, x_gradient)
            tan_weight = np.hypot(x_gradient, y_gradient)  # * gauss  # 根据距离中心位置分权重
            tan_garadient[tan_garadient < 0] += pi2

            his = self.__getHis(tan_garadient, tan_weight)
            mean_his = np.argmax(his)
            gradient_im = ndimage.rotate(gradient_im, -(mean_his * 10))  # 旋转至主方向
            center = gradient_im.shape[0] / 2
            new_len = 13
            weight = gradient_im[int(center - new_len):int(center + new_len + 1),
                     int(center - new_len):int(center + new_len + 1)]

            feature_y_gradient = ndimage.sobel(weight, axis=0)
            feature_x_gradient = ndimage.sobel(weight, axis=1)
            feature_tan = np.arctan2(feature_y_gradient, feature_x_gradient)
            feature_tan[feature_tan < 0] += pi2
            tan_weight = np.hypot(feature_x_gradient, feature_y_gradient)

            xy_gradient = np.dot(tan_weight,feature_tan).reshape(-1)
            feature_matrix.append(preprocessing.scale( xy_gradient))
            point_list.append([int(x[i] * lessen[i]), int(y[i] * lessen[i])])
        pca = PCA(n_components=256)
        pca.fit(feature_matrix)
        feature = pca.transform(feature_matrix)
        print('stop get feature \n')
        return np.array(feature), np.array(point_list)

    def GetFeature(self, point_list):
        '''
        extracting the pore-SIFT feature from point_list
        :param point_list:A list with feature points
        :return:feature , x and y
        '''
        print ('Start get feature(vector with size (512))')

        x = point_list[:, 0]
        y = point_list[:, 1]
        octave_num = point_list[:, 2]
        inv_octave_num = point_list[:,3]
        lessen = point_list[:, 4]

        point_weight = []
        point_list = []
        lens = np.size(x, axis=0)
        pi2 = 2 * 3.1415
        li = np.int(20)
        #gauss = (np.dot(cv2.getGaussianKernel(li * 2 + 1, 10), cv2.getGaussianKernel(li * 2 + 1, 10).T) * 1e4).reshape(-1)/10.0
        for i in range(lens):
            if i%1000 == 0:
                print ('Has got',i,'features')
            row = np.int(x[i])
            col = np.int(y[i])
            try:
                gradient_im = np.array(self.octave_gauss[octave_num[i],inv_octave_num[i]][row - li:row + li + 1, col - li:col + li + 1])
                assert gradient_im.shape[0] == li * 2 + 1
                assert gradient_im.shape[1] == li * 2 + 1
            except:
                # print 'Border point'
                continue
            y_gradient = ndimage.sobel(gradient_im, axis=0).reshape(-1)
            x_gradient = ndimage.sobel(gradient_im, axis=1).reshape(-1)
            tan_garadient = np.arctan2(y_gradient, x_gradient)
            tan_weight = np.hypot(x_gradient, y_gradient)#* gauss  # 根据距离中心位置分权重
            tan_garadient[tan_garadient < 0] += pi2

            his = self.__getHis(tan_garadient, tan_weight)
            mean_his = np.argmax(his)
            gradient_im = ndimage.rotate(gradient_im, -(mean_his * 10))  # 旋转至主方向
            center = gradient_im.shape[0]/2
            new_len = 13
            weight = gradient_im[int(center - new_len):int(center + new_len+1), int(center - new_len):int(center + new_len+1)]
            #weight = np.delete(weight, new_len, axis=0)
            #weight = np.delete(weight, new_len, axis=1)

            feature_y_gradient = ndimage.sobel(weight, axis=0)
            feature_x_gradient = ndimage.sobel(weight, axis=1)
            feature_tan = np.arctan2(feature_y_gradient, feature_x_gradient)
            feature_tan[feature_tan < 0] += pi2
            feature_weight = np.hypot(feature_x_gradient, feature_y_gradient)
            his_feature = []
            border_len = 3
            border_size = 6
            #get 512 vector/128vector
            for t in range(64):
                tx = t / 8
                ty = t - tx * 8
                weight_value = self.__getHis(feature_tan[int(tx * border_len):int(tx * border_len + border_size), int(ty * border_len):int(ty * border_len + border_size)],
                feature_weight[int(tx * border_len):int(tx * border_len + border_size), int(ty * border_len):int(ty * border_len + border_size)], bin=8)
                his_feature.extend(weight_value)
            point_weight.append(preprocessing.scale(his_feature))
            point_list.append([int(x[i] * lessen[i]), int(y[i] * lessen[i])])
        print ('stop get feature \n')
        return np.array(point_weight), np.array(point_list)

    def GetPPCASIFTFeature(self):
        self.CreatePyramid()
        point_list = self.ScolePoint()
        feature, points = self.GetPCAFeature(point_list=point_list)
        return np.array([feature, points, self.im])

    def GetPSIFTFeature(self):
        """
        This function is used to get picture`s pore-SIFT feature using approve  param
        :return: image`s matrix ; feature ; feature points `s list
        """
        self.CreatePyramid()
        point_list = self.ScolePoint()
        feature , points = self.GetFeature(point_list=point_list)
        return np.array([feature,points,self.im])

    def DisPlayPyramid(self):
        """
         This function is used to display the image`s pyramid
         :return:None 
        """
        n = self.octave_num*10+100
        for i in range(self.octave_num):
            temp_im = self.cov_im[i][1,:,:]*255
            pl.subplot(n+i+1)
            pl.imshow(temp_im)
            pl.gray()
        pl.show()

    def DisPlaySource(self):
        """
        This function is used to display the Scoure image
        :return:None 
        """
        n = self.octave_num * 100 + 10
        for i in range(self.octave_num):
            temp_im = self.octave_im[i]
            pl.subplot(n + i + 1)
            pl.imshow(temp_im)
            pl.gray()
        pl.show()

    def __getHis(self,tan_garadient,tan_weight,bin = 36):
        """
        This function is used to get a histogram
        :param tan_garadient: gradient Matrix
        :param tan_weight:  weight Matrix
        :param bin: the bin what histogram needs
        :return: histogram
        """
        his = np.zeros(bin)
        tmp_grad = np.array(tan_garadient).reshape([-1])
        tmp_weight = np.array(tan_weight).reshape([-1])
        for i in np.arange(tmp_grad.shape[-1]):
            index = int (tmp_grad[i] / (2*3.1416/bin))
            his[index]+=tmp_weight[i]
        return his

    def SavePlayPyramid(self,file):
        """
            This function is used to save and display the Picture`s Pyramid 
        :param file: file path
        :return: None
        """
        n = self.octave_num * 10 + 100
        pl.figure()
        for i in range(self.octave_num):
            temp_im = self.cov_im[i][1, :, :]
            pl.subplot(n + i + 1)
            pl.imshow(temp_im)
            pl.gray()
        pl.savefig(file+'LOG.png')

    @staticmethod
    def SaveFeatureFile(feature,path):
        """
        Saving feature include feature ,points and image
        :param feature: Pore-SIFT feature
        :param point: The point whitch reflect feature
        :param im: imput image
        :param path: file `s name
        :return: None
        """
        np.save(path,feature)
    @staticmethod
    def GetFeatureFile(path):
        """
        Geting Pore-SIFT feature from file
        :param path: file name
        :return: Pore-SIFT feature
        """
        serialization =  np.load(path)
        return serialization

    @staticmethod
    def match(featureA,featureB,file,ratio=0.6,RANSAC = False,kmeans= False,dispaly=True,save = False,plot_match=True):
        im1 = featureA[2]
        im2 = featureB[2]
        feature_a=featureA[0]
        feature_b=featureB[0]
        a_pts=featureA[1]
        b_pts=featureB[1]
        print ('Start match:',feature_a.shape[0],feature_b.shape[0])
        if kmeans==True:
            a_len = feature_a.shape[0]
            allfeature = np.row_stack((feature_a,feature_b))
            temp, classified_points, means = cv2.kmeans(data=np.float32(allfeature),K=2,bestLabels=None,criteria=(cv2.TERM_CRITERIA_MAX_ITER , 120, 10), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
            class_a = classified_points[0:a_len]
            class_b = classified_points[a_len-1:-1]
            one_value = np.sum(classified_points)
            if one_value> len(classified_points)/2:
                ture_value = 1
            else:
                ture_value = 0
            feature_a = feature_a[class_a.ravel() == ture_value,:]
            feature_b = feature_b[class_b.ravel() == ture_value,:]
            a_pts = a_pts[class_a.ravel() == ture_value,:]
            b_pts = b_pts[class_b.ravel() == ture_value,:]
            if save==True:
                Psift.SaveFeaturePoint(im1, a_pts, im2, b_pts, file + 'detected_point')
            if dispaly==True:
                Psift.DisFeaturePoint(im1, a_pts, im2, b_pts)
        #Ratio:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(np.float32(feature_a), np.float32(feature_b), k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        print ('good', len(good))
        #RANSAC
        src_pts = np.float32([ a_pts[m.queryIdx] for m in good ])
        dst_pts = np.float32([ b_pts[m.trainIdx] for m in good ])
        if save == True:
            Psift.SaveFeaturePoint(im1, src_pts, im2, dst_pts, file + 'favourable_point')
        if RANSAC == True:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,mask=8)# eight-point algorithm
            matchesMask = mask.ravel()
            src_pts = src_pts[matchesMask>0]
            dst_pts = dst_pts[matchesMask>0]
            if dispaly == True:
                Psift.DisFeaturePoint(im1, src_pts, im2, dst_pts)
            if save ==True:
                Psift.SaveFeaturePoint(im1, src_pts, im2, dst_pts, file + 'RANSAC_point')
        print ('Detect' ,len(src_pts))
        if plot_match==True:
            Psift.plot_match(im1,im2,src_pts,dst_pts,file,gray=True)
        return src_pts,dst_pts

    @staticmethod
    def appendimages(im1,im2):
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]

        if rows1 <rows2:
            im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
        elif rows1 > rows2:
            im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
        return np.concatenate((im1,im2),axis=1)

    @staticmethod
    def plot_match(im1,im2,src_point,dst_point,file=None,gray = True):
        pl.close()
        x1= src_point[:,0]
        y1 = src_point[:,1]
        x2 = dst_point[:,0]
        y2 = dst_point[:,1]
        im3 = Psift.appendimages(im1,im2)
        pl.imshow(im3)
        if gray:
          pl.gray()
        cols1 = im1.shape[1]
        for i in range(x1.shape[0]):
            pl.plot([y1[i],y2[i]+cols1],[x1[i],x2[i]],'c')
        pl.axis('off')
        if file != None:
            pl.savefig(file+'mtach_result.png')
        pl.show()

    @staticmethod
    def DisFeaturePoint(im1,new_point1,im2,new_point2):
        pl.subplot(121)
        pl.imshow(im1)
        pl.gray()
        pl.axis('off')
        pl.plot(new_point1[:, 1], new_point1[:, 0], '.')
        pl.subplot(122)
        pl.imshow(im2)
        pl.gray()
        pl.plot(new_point2[:, 1], new_point2[:, 0], '.')
        pl.axis('off')
        pl.show()

    @staticmethod
    def SaveFeaturePoint(im1, new_point1, im2, new_point2,name):
        pl.figure()
        pl.subplot(121)
        pl.imshow(im1)
        pl.gray()
        pl.plot(new_point1[:, 1], new_point1[:, 0], '.')
        pl.axis('off')
        pl.subplot(122)
        pl.imshow(im2)
        pl.gray()
        pl.plot(new_point2[:, 1], new_point2[:, 0], '.')
        pl.axis('off')
        pl.savefig(name+'.png')

