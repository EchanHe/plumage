"""
按不同类型读图片，
并生成为：
x (m, 宽*高*3)
y (m,关键点*3)
"""

import numpy as np
import pandas as pd


from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2

import os, errno
from random import randint,uniform,choice,random
import math
from PIL import ImageEnhance,ImageChops,ImageOps,ImageFilter
import sys

        
class plumage_data_input:
    """
    读数据的类。
    一次读入所有图片的索引表
    然后随机或者按顺序返回batch大小的数据
    """
    file_name_col = 'file.vis'
    file_info_cols = ['file.vis', 'file.uv', 'view' , 'img.missing']
    bounding_cols = ['outline.bb' , 'poly.outline']
    patches_cols = ['poly.crown', 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail'
    , 'poly.wing.coverts',   'poly.wing.primaries.secondaries',
     'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside']
    coords_cols = ['s02.standard_x', 's02.standard_y', 's20.standard_x', 's20.standard_y',
       's40.standard_x', 's40.standard_y', 's80.standard_x', 's80.standard_y',
       's99.standard_x', 's99.standard_y','crown_x', 'crown_y', 'nape_x',
       'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
       'tail_y', 'throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
       'belly_y', 'tail.underside_x', 'tail.underside_y', 'wing.coverts_x',
       'wing.coverts_y', 'wing.primaries.secondaries_x',
       'wing.primaries.secondaries_y']
    contour_col = ["poly.outline" ]
    cols_num_per_coord = 2
    lm_cnt = int(len(coords_cols) /  cols_num_per_coord)
    
    aug_option = {'trans' :False , 'rot' :True , 'scale' :True}
    def __init__(self,df,batch_size,is_train,pre_path, state,scale=1 ,is_aug = True):

        self.df  =df# "train_pad/Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
        self.pre_path = pre_path
        self.scale = scale
        # self.X = X
        self.df_size = df.shape[0]
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_aug = is_aug
        self.state = state

        self.start_idx =0
        self.indices = np.arange(self.df_size)
        np.random.shuffle(self.indices)

        filepath_test = pre_path+df.iloc[0,0]
        img = Image.open(filepath_test)
        self.img_width= img.size[0]
        self.img_height = img.size[1]

        print("Init data class...")
        print("\tData shape: {}\n\tbatch_size:{}\n\tImage resolution: {}*{}\n\tImage Augmentation:{}"\
            .format(self.df_size, self.batch_size ,self.img_width , self.img_height, self.is_aug))

        
        
        

    def get_next_batch(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
            self.indices = np.arange(self.df_size)
            np.random.shuffle(self.indices)

        excerpt = self.indices[self.start_idx:self.start_idx + batch_size]
        df_mini = self.df.iloc[excerpt].copy()
        
        self.start_idx += batch_size
        #returning X and y 
        if self.is_aug:
            x_mini, df_mini = self.get_x_df_aug(df_mini)
        else:
            x_mini = self.get_x_img(df_mini)
        # print(df_mini)
        if is_train:
            # print(df_mini)
            if self.state =='coords':
                y_mini = self.get_y_map(df_mini)
                coords_mini = self.get_y_coord(df_mini , 1 , True)
                vis_mini = self.get_y_vis_map(df_mini)
                return x_mini , y_mini, coords_mini,vis_mini
            elif self.state =='patches':
                y_mini = self.get_y_patches(df_mini)
                return x_mini , y_mini
            elif self.state =='contour':
                y_mini = self.get_y_contour(df_mini)
                return x_mini , y_mini
        else:
            return x_mini

    def get_next_batch_no_random(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
        df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]
        self.start_idx += batch_size
        # print(df_mini.image_id)
        x_mini = self.get_x_img(df_mini)
        
        if is_train:
            # print(df_mini)
            if self.state =='coords':
                y_mini = self.get_y_map(df_mini)
                coords_mini = self.get_y_coord(df_mini , 1 , True)
                vis_mini = self.get_y_vis_map(df_mini)
                return x_mini , y_mini, coords_mini,vis_mini
            elif self.state =='patches':
                y_mini = self.get_y_patches(df_mini)
                return x_mini , y_mini
            elif self.state =='contour':
                y_mini = self.get_y_contour(df_mini)
                return x_mini , y_mini
        else:
            return x_mini
        

    def get_next_batch_no_random_all(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]

        # print(df_mini.image_id)
        x_mini = self.get_x_img(df_mini)
        
        if is_train:
            # print(df_mini)
            if self.state =='coords':
                y_mini = self.get_y_map(df_mini)
                coords_mini = self.get_y_coord(df_mini , 1 , True)
                vis_mini = self.get_y_vis_map(df_mini)
                return x_mini , y_mini, coords_mini,vis_mini
            elif self.state =='patches':
                y_mini = self.get_y_patches(df_mini)
                return x_mini , y_mini
            elif self.state =='contour':
                y_mini = self.get_y_contour(df_mini)
                return x_mini , y_mini
        else:
            return x_mini
        
    ######get data#############
    
    #Get image in [batch,width,height,3]
    def get_x_img(self ,df):
        scale = self.scale
        folder = self.pre_path

        size= df.shape[0]  
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)

        x_all = np.zeros((size, height , width,3))

        i=0
        for idx,row in df.iterrows():

            filename =folder+row[self.file_name_col]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)

            x_all[i,:,:,:] = img
            i+=1
            x_all

        return x_all.astype('uint8')
    
    #Get patches in [batch, width, height, patches]
    def get_y_patches(self, df):
        scale = self.scale
        cols = self.patches_cols
        
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)
        
        output_map = np.zeros((self.batch_size , height,width ,  len(cols)+1))

        patches = df[cols].as_matrix()
        for row in np.arange(patches.shape[0]):
            for col in np.arange(patches.shape[1]):
                patch = patches[row,col]
                mask = np.zeros((height, width))
                if patch is not np.nan and patch!="-1":
                    patch_int = np.array([int(s) for s in patch.split(",")])
                    coords = np.reshape((patch_int/scale).astype(int).tolist(), (-1, 2))
                    cv2.fillConvexPoly(mask, coords , 1)
                    output_map[row, :,:,col+1] = mask
            label_row = np.argmax(output_map[row, ...],axis=2)
            output_map[row,label_row==0,0] =1        
        return output_map

    def get_y_contour(self, df):
        scale = self.scale
        cols = self.contour_col
        
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)
        
        output_map = np.zeros((self.batch_size , height,width ,  len(cols)+1))

        patches = df[cols].as_matrix()
        for row in np.arange(patches.shape[0]):
            for col in np.arange(patches.shape[1]):
#                 img = Image.new('1', (width, height), 0)
                mask = np.zeros((height, width))
                patch = patches[row,col]
                if patch is not np.nan and patch!="-1":
                    patch_int = np.array([int(s) for s in patch.split(",")])
    
                    coords = np.reshape((patch_int/scale).astype(int).tolist(), (-1, 2))
                    coords = np.expand_dims(coords , axis = 0)
                    cv2.fillPoly(mask, coords, 1)
                    output_map[row, :,:,col+1] = mask
            label_row = np.argmax(output_map[row, ...],axis=2)
            output_map[row,label_row==0,0] =1  
        return output_map
    
    def get_y_coord(self ,df,scale = 1,coord_only = False):
        """
        返回[m,2*landmark]的关键点坐标数组
        """
        l_m_columns = self.coords_cols
        cols_num_per_coord = self.cols_num_per_coord

        y_coord = df[l_m_columns].as_matrix()
        return y_coord

        y_coord[:,np.arange(0,y_coord.shape[1],3)] = y_coord[:,np.arange(0,y_coord.shape[1],3)]/scale
        y_coord[:,np.arange(1,y_coord.shape[1],3)] = y_coord[:,np.arange(1,y_coord.shape[1],3)]/scale
        
        l_m_index = np.append(np.arange(0,y_coord.shape[1],3), np.arange(1,y_coord.shape[1],3) )

        l_m_index = np.sort(l_m_index)
        vis_index = np.arange(2,y_coord.shape[1],3)

        
        # Whether has landmark point
        has_lm_data = y_coord[:,vis_index]
        has_lm_data[has_lm_data==0]=1
        has_lm_data[has_lm_data==-1]=0

        # Whether is visible
        is_vis_data = y_coord[:,vis_index]
        is_vis_data[np.logical_or( is_vis_data ==-1 , is_vis_data==0 )]=0
        
        if coord_only:
            return y_coord[:,l_m_index]
        return_array = np.concatenate((y_coord[:,l_m_index],has_lm_data , is_vis_data),axis=1)
        return return_array
        #x1,y1 ... xn, yn
        #lm_1 ... lm_n
        #vis_1 ... vis_n


    def get_y_map(self ,df):
        """
        返回[m,64,64,landmark]的关键点热点图
        """
        l_m_columns = self.coords_cols
        cols_num_per_coord = self.cols_num_per_coord

        y_coord = df[l_m_columns].as_matrix()


        lm_cnt = self.lm_cnt

        df_size = y_coord.shape[0]

        real_scale = self.scale * 4
        scaled_width = int(self.img_width/real_scale)
        scaled_height = int(self.img_height/real_scale)

        pad = int(scaled_width/2)
        pad_half = int(pad/2)

        
        y_map = np.zeros((df_size,scaled_height,scaled_width,lm_cnt))

        for j in range(df_size):
            for i in range(lm_cnt):
                x = y_coord[j,i*cols_num_per_coord]
                y = y_coord[j,i*cols_num_per_coord+1]
                if x!=-1 and y!=-1:
                    scale_x = int((x-1)/real_scale)
                    scale_y = int((y-1)/real_scale)

                    y_map_pad = np.zeros((scaled_height+pad,  scaled_width+pad))
                    y_map_pad[scale_y + pad_half,  scale_x+pad_half] = 200
                    y_map_pad = gaussian_filter(y_map_pad,sigma=2)
                    y_map[j,:,:,i] = y_map_pad[pad_half:scaled_height+pad_half, pad_half:scaled_width+pad_half]
                    # y_map[j,y,x,i]=1
        y_map = np.round(y_map,8)
        return y_map


    def get_y_map_ori(self , df):
        """
        返回[m,64,64,landmark]的关键点热点图
        """
        scale = self.scale
        l_m_columns = self.coords_cols
        cols_num_per_coord = self.cols_num_per_coord

        pad = int(self.img_width/5)
        pad_half = int(pad/2)
        gaus_sigma = 20
        
        y_coord = df[l_m_columns].as_matrix().astype(int)
        lm_cnt = int(y_coord.shape[1]/cols_num_per_coord)
        df_size = y_coord.shape[0]


        height = int(self.img_height/(scale*8))
        width = int(self.img_width/(scale*8))

        y_map = np.zeros((df_size,height,width,lm_cnt))

        for j in range(df_size):
            for i in range(lm_cnt):

                y_map_ori = np.zeros((self.img_height+pad,  self.img_width+pad))
                x = y_coord[j,i*cols_num_per_coord]
                y = y_coord[j,i*cols_num_per_coord+1]
                # print(y_map_ori.shape , x, y)

                if x!=-1 and y!=-1:
                    y_map_ori[y + pad_half , x + pad_half] = 20000
                    y_map_ori =  gaussian_filter(y_map_ori,sigma=gaus_sigma)
                    y_map_ori = y_map_ori[pad_half:self.img_height+pad_half, pad_half:self.img_width+pad_half]
                    y_map[j,:,:,i] = cv2.resize(y_map_ori, dsize=(width, height),interpolation = cv2.INTER_NEAREST)
                    # y_map[j,y,x,i]=1
        y_map = np.round(y_map,8)
        print(y_map.shape)
        return y_map
    def get_y_vis_map(self ,df):
        """
        返回[m,64,64,landmark]的关键点是否能看到热点图
        """
        l_m_columns = self.coords_cols
        y_coord = df[l_m_columns].as_matrix()
        lm_cnt = self.lm_cnt
        df_size = y_coord.shape[0]


        real_scale = self.scale * 4
        scaled_width = int(self.img_width/real_scale)
        scaled_height = int(self.img_height/real_scale)

        y_map = np.ones((df_size,scaled_height,scaled_width,lm_cnt))

        for j in range(df_size):
            for i in range(lm_cnt):

                is_lm = y_coord[j,i*self.cols_num_per_coord]
                if is_lm==-1:
                    y_map[j,:,:,i] = np.zeros((scaled_height,scaled_width))
        return y_map




    def get_x_df_aug(self ,df):
        folder = self.pre_path
        l_m_columns = self.coords_cols
        scale = self.scale

        size= df.shape[0]  
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)

        x_all = np.zeros((size, height , width,3))

        img_id=0

        for idx,row in df.iterrows(): 
            filename = folder+row[self.file_name_col]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hei = img.shape[0]
            img_wid = img.shape[1]
#             img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
            aug_prob = 1
            if aug_prob > 0.5:
                
                #-----平移----
                min_offset = 0
                max_offset = 1000
                trans_lr = choice([randint(min_offset,max_offset) , randint(-max_offset,-min_offset)] ) #left/right (i.e. 5/-5)

                trans_ud = choice([randint(min_offset,max_offset) , randint(-max_offset,-min_offset)] ) #up/down (i.e. 5/-5)

                
                old_row = row.copy()
                for i in np.arange(0,len(l_m_columns),self.cols_num_per_coord):
                    id_x = l_m_columns[i]
                    id_y = l_m_columns[i+1]
                    id_vis = l_m_columns[i]
                    if row[id_vis] !=-1:
                        row[id_x] = row[id_x] + trans_lr
                        row[id_y] = row[id_y] + trans_ud
                        inside = row[id_x]>self.img_width or row[id_x]<0 or row[id_y]>self.img_height or row[id_y]<0 
                        if inside:
                            trans_lr=0
                            trans_ud = 0
                            row= old_row
                            break
                M = np.float32([[1,0,trans_lr],[0,1,trans_ud]])
                
                img = cv2.warpAffine(img,M,(img_wid,img_hei))
#                 img = img.transform(img.size, Image.AFFINE, (1, 0, trans_lr, 0, 1, trans_ud))
                

                
                
                # ---------Rotate--------
                if self.aug_option['rot']:
                    angle_bound = 30
                    angle = randint(-angle_bound,angle_bound)
                    # angle =15
                    radian = math.pi/180*angle

                    old_row = row.copy()
                    for i in np.arange(0,len(l_m_columns),self.cols_num_per_coord):
                        id_x = l_m_columns[i]
                        id_y = l_m_columns[i+1]
                        id_vis = l_m_columns[i]
                        if row[id_vis] !=-1:
                            x = row[id_x] - self.img_width/2
                            y = (self.img_height- row[id_y])
                            y=y- self.img_height/2 
                            row[id_x] = x*math.cos(radian) - ((y)*math.sin(radian))
                            row[id_x] =int((row[id_x] + self.img_width/2))
                            row[id_y] =(x*math.sin(radian) + ((y)*math.cos(radian)))
                            row[id_y] =int ((self.img_height-(row[id_y]+self.img_height/2 )))
                            inside = row[id_x]>self.img_width or row[id_x]<0 or row[id_y]>self.img_height or row[id_y]<0 
                            if inside:
                                angle=0
                                row= old_row
                                break
                    M = cv2.getRotationMatrix2D((img_wid/2,img_hei/2),angle,1)
                    img = cv2.warpAffine(img,M,(img_wid,img_hei))
#                     img = img.rotate(angle)
                
                
                ##-----Scale------
                if self.aug_option['scale']:
                    #scale value:
                    scale_ratio = randint(10,20)/10
#                     scale_ratio = 1.5
                    new_scaled_width = (int(self.img_width/scale_ratio))
                    new_scaled_height = (int(self.img_height/scale_ratio))
                    
                    img = cv2.resize(img, dsize=(new_scaled_width,new_scaled_height),
                                     interpolation=cv2.INTER_CUBIC)
                    delta_w = self.img_width - new_scaled_width
                    delta_h = self.img_height - new_scaled_height
                    
                    img= cv2.copyMakeBorder(img,delta_h//2,delta_h-(delta_h//2),
                                            delta_w//2,delta_w-(delta_w//2),
                                            cv2.BORDER_CONSTANT,value=[0,0,0])
                    
                    for i in np.arange(0,len(l_m_columns),self.cols_num_per_coord):
                        id_x = l_m_columns[i]
                        id_y = l_m_columns[i+1]
                        id_vis = l_m_columns[i]
                        if row[id_vis] !=-1:
                            row[id_x] = (row[id_x] )//scale_ratio+ delta_w//2
                            row[id_y] = (row[id_y])//scale_ratio + delta_h//2


#                 img[...,2]+100
                    
                #---Intensity and enhance---#
                if False:
                    aug_type = randint(0,3)
        #             print(aug_type)
                    if random()>0.7:
                        value =choice([uniform(0.3,0.7) , uniform(1.5,2.0)] )
                        value = round(value,1)

                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(value)
                    if random()>0.7:   
                        value =choice([uniform(0.3,0.7) , uniform(1.5,2.0)] )
                        value = round(value,1)
                        enhancer = ImageEnhance.Brightness(img)
                        img = enhancer.enhance(value)

                    if random() > 0.7:
                        value =choice([uniform(0.3,0.7) , uniform(1.5,2.0)] )
                        value = round(value,1)                
                        enhancer = ImageEnhance.Color(img)
                        img = enhancer.enhance(value)

                    if random() > 0.9:
                        img = ImageOps.invert(img)

                    sharp_blur = random()
                    if sharp_blur>0.7:
                        img = img.filter(ImageFilter.UnsharpMask)
                    elif sharp_blur<0.3:
                        img = img.filter(ImageFilter.BLUR)

            df.loc[idx,:] = row      
            img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
            x_all[img_id,:,:,:] = img
            img_id+=1
            
        return x_all.astype('uint8'),df
