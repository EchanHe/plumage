"""
Data input class
generate image and labels
"""

import numpy as np
import pandas as pd


from scipy.ndimage import gaussian_filter
import cv2

import os, errno
import random as rand_lib
from random import randint,uniform,choice,random
import math
from PIL import ImageEnhance,ImageChops,ImageOps,ImageFilter
import sys

from config_data import return_patches_cols , return_coords_cols


def check_masks(masks):
    """
    Check masks are Mutually Exclusive
    By going through every mask
    """
    for mask in masks:
        if not check_mask(mask):
            return False
    return True

def check_mask(mask):
    """
    Check mask is mutually exclusive
    By checking the sum of every class is equal to the total pixels
    """
    img_wid = mask.shape[1]
    img_hei = mask.shape[0]
    sum_pixel = 0 
    for c in range(mask.shape[2]):
        sum_pixel += np.sum(mask[...,c])

    return  sum_pixel == (img_wid*img_hei)

# Change 
def segs_to_masks(segs , n_cl):
    """
    Transfer segmentaions to masks
    
    params
        segs: [batch_size, height, width]
        n_cl: total classes number
    
    return masks [batch_size, height, width, n_cl]
    """
    
    assert len(segs.shape) ==3 , "Make sure input is [batch_size , height, width]"
    df_size = segs.shape[0]

    masks = np.zeros((segs.shape[0] , segs.shape[1] , segs.shape[2] , n_cl))
    for i in np.arange(df_size):
        cl, _ = extract_classes(segs[i,...] )
        masks[i,...] = extract_masks(segs[i,...] , cl, n_cl)

    return masks
#-----helper functions of segs_to_masks----

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((h, w , n_cl))

    for i, c in enumerate(cl):
        masks[:, : , c] = segm == c

    return masks
def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise
    return height, width

#-----helper functions of segs_to_masks----


# Data class
class plumage_data_input:
    """
    Class that read a dataframe
    and output the images and labels for the network
    """
    
    #---- basic config ----
    file_name_col = 'file.vis'
    file_info_cols = ['file.vis', 'file.uv', 'view' , 'img.missing']
    contour_col = ["poly.outline" ]
    # patches_cols = ['poly.crown' , 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail',
    # 'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside',
    #  'poly.wing.coverts',   'poly.wing.primaries.secondaries']

    # coords_cols = ['s02.standard_x', 's02.standard_y', 's20.standard_x', 's20.standard_y',
    #    's40.standard_x', 's40.standard_y', 's80.standard_x', 's80.standard_y',
    #    's99.standard_x', 's99.standard_y','crown_x', 'crown_y', 'nape_x',
    #    'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
    #    'tail_y', 'throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
    #    'belly_y', 'tail.underside_x', 'tail.underside_y', 'wing.coverts_x',
    #    'wing.coverts_y', 'wing.primaries.secondaries_x',
    #    'wing.primaries.secondaries_y']
    # contour_col = ["poly.outline" ]
    # cols_num_per_coord = 2
    # lm_cnt = int(len(coords_cols) /  cols_num_per_coord)
    
    aug_option = {'trans' :True , 'rot' :True , 'scale' :True}
    STATE_OPTIONS = ['coords', 'contour', 'patches']
    
    #---basic config ---
    def __init__(self,df,batch_size,is_train,pre_path, state,
        scale=1 ,is_aug = True , heatmap_scale = 4,
         view = 'all', no_standard = False , train_ids = None):
        """
        init function for the data
        
        params:
            df: a dataframe with image names and markups(training and validation set)
            batch_size: size of a batch
            is_train: The input data in training or not? False means only read image names
            pre_path: The image folder
            State: The labels to read with giving state
            scale: The scale for downsampling the images
            is_aug: Whether augment the data
            
        """
        self.df  = df.copy()
        self.pre_path = pre_path
        self.scale = scale
        self.heatmap_scale = heatmap_scale    
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_aug = is_aug
        #
        if self.is_aug:
            # make data frame:
            df2 = self.df.copy()
            self.df.loc[:,'aug'] =False
            df2.loc[:,'aug'] = True
            self.df = pd.concat([self.df, df2]).reset_index(drop=True)
        self.df_size = self.df.shape[0]
        
        self.all_columns = df.columns
        self.coords_cols = return_coords_cols(view = view , no_standard = no_standard, train_ids = train_ids)
        self.patches_cols = return_patches_cols(view = view , train_ids = train_ids)
        self.cols_num_per_coord = 2
        self.lm_cnt = int(len(self.coords_cols) /  self.cols_num_per_coord)
        
        self.start_idx =0
        self.indices = np.arange(self.df_size)
        np.random.shuffle(self.indices)
        filepath_test = pre_path+df.loc[df.index[0],self.file_name_col]
        print(filepath_test)
        img = cv2.imread(filepath_test)
#         img = Image.open(filepath_test)
        self.img_width= img.shape[1]
        self.img_height = img.shape[0]

        print("Init data class...")
        print("\tOrignal Data shape: {}\n\tData shape: {}\n\tbatch_size:{}\n\tscale:{}\n\tImage original resolution: {}*{}\n\tscaled resolution: {}*{}"\
            .format(df.shape[0],self.df_size, self.batch_size,self.scale ,self.img_width , self.img_height, self.img_width//self.scale , self.img_height//self.scale))

        print("Image Augmentation:{}\n\tAugmentation option:".format(self.is_aug, self.aug_option))
      
        
        assert state in self.STATE_OPTIONS, "State is not in {}".format(self.STATE_OPTIONS)
        self.state = state

        print("Data output type: {}".format(self.state))
        if self.state =='coords':
            print("\tOutput stride: {}\n\tThe output heatmap shape: {}*{}*{}".format(self.heatmap_scale,
             self.img_width//(self.scale*self.heatmap_scale),  self.img_height//(self.scale*self.heatmap_scale), self.lm_cnt ))
            
            self.output_channel = int(len(self.coords_cols) /  self.cols_num_per_coord)
        elif self.state =='patches':
            self.output_channel = len(self.patches_cols) + 1
        elif self.state =='contour':
            self.output_channel = len(self.contour_col) + 1
        
        
        
    
    def get_next_batch(self):
        """
        Return the images and different labels 
        in batch and random order
        
        Options:
            Can return augmentation images and labels
            randomly Return coordiantion or patches or outline labels.
        
        """
        #Augmentation is only in this next batch.
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

        if is_train:
            # Return a batch of coords
            if self.state =='coords':
                if self.is_aug:
                    x_mini, df_mini = self.get_x_df_aug(df_mini)
                else:
                    x_mini = self.get_x_img(df_mini)
                
                y_mini = self.get_y_map(df_mini)
                coords_mini = self.get_y_coord(df_mini , 1 , True)
                vis_mini = self.get_y_vis_map(df_mini)
                return x_mini , y_mini, coords_mini,vis_mini
            else:
                x_mini = self.get_x_img(df_mini)
                y_mini = None
                if self.state =='patches':
                    y_mini = self.get_y_patches(df_mini)
                elif self.state =='contour':
                    y_mini = self.get_y_contour(df_mini)
                if self.is_aug:
                    x_mini , y_mini = self.get_x_masks_aug(x_mini , y_mini)
                #print("The mask :" ,check_masks(y_mini))
                if check_masks(y_mini) ==False:
                    y_mini = segs_to_masks(np.argmax(y_mini , 3) , self.output_channel)
                return x_mini , y_mini

        else:
            return x_mini

    def get_next_batch_no_random(self):
        """
        Return the images and different labels 
        in batch and in the dataframe order.

        If the index slice reach the end, wrap the slice.
        e.g. total length 10. [8:12] [8,9,0,1] 
        
        Options:
            Return coordiantion or patches or outline labels.
        
        """
        #Augmentation is only in this next batch.
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        # Index part
        int_index = np.arange(df_size)
        if self.start_idx >= df_size:
            self.start_idx = 0 
        # Wrap the index and use iloc on the wrapped index.
        wrapped_index = int_index.take(range(self.start_idx, self.start_idx+batch_size), axis = 0, mode = 'wrap')
        df_mini = self.df.iloc[wrapped_index,:]
        self.start_idx += batch_size

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
        """
        Return the images and different labels 
        in batch and in the dataframe order.
        and return last batch in restshape than restart.
        
        Options:

            Return coordiantion or patches or outline labels.
        
        """
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        if self.start_idx >= df_size:
            self.start_idx = 0

        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]
        self.start_idx += batch_size


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
   

    def get_next_batch_no_random_all_labels(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        if self.start_idx>=df_size:
            self.start_idx = 0
        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]
        x_mini = self.get_x_img(df_mini)
        coords_mini = self.get_y_coord(df_mini , self.scale , True)
        p_coords_mini = self.get_contour_patches_coords(df_mini , mode = 'patches' )
        c_coords_mini = self.get_contour_patches_coords(df_mini , mode = 'contour')
        self.start_idx += batch_size
        return x_mini, coords_mini , p_coords_mini, c_coords_mini
        
    def get_next_batch_no_random_all_labels_without_img(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        if self.start_idx+1>=df_size:
            self.start_idx = 0
        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]

        coords_mini = self.get_y_coord(df_mini , self.scale , True)
        p_coords_mini = self.get_contour_patches_coords(df_mini , mode = 'patches' )
        c_coords_mini = self.get_contour_patches_coords(df_mini , mode = 'contour')
        self.start_idx += batch_size
        return coords_mini , p_coords_mini, c_coords_mini
    
    def get_next_batch_no_random_all_point_mask_without_img(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        if self.start_idx+1>=df_size:
            self.start_idx = 0
        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]

        coords_mini = self.get_y_coord(df_mini , self.scale , True)
        p_coords_mini = self.get_y_patches(df_mini)

        self.start_idx += batch_size
        return coords_mini , p_coords_mini  
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
    
   
    def get_y_patches(self, df):
         #Get patches in [batch, width, height, patches]
        scale = self.scale
        cols = self.patches_cols
        
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)
        
        output_map = np.zeros((df.shape[0], height,width ,  len(cols)+1))

        # PD future warning
        # patches = df[cols].as_matrix()
        patches = df[cols].values
        for row in np.arange(patches.shape[0]):
            for col in np.arange(patches.shape[1]):
                patch = patches[row,col]
                mask = np.zeros((height, width))
                if patch is not np.nan and patch!="-1" and patch !=-1:
                    patch_int = np.array([int(s) for s in patch.split(",")])
                    coords = np.reshape((patch_int/scale).astype(int).tolist(), (-1, 2))
                    cv2.fillConvexPoly(mask, coords , 1)
                    output_map[row, :,:,col+1] = mask
            label_row = np.argmax(output_map[row, ...],axis=2)
            output_map[row,label_row==0,0] =1      
        return output_map

    
    def get_y_contour(self, df):
        #Get the contour
        scale = self.scale
        cols = self.contour_col
        
        width = int(self.img_width/scale)
        height = int(self.img_height/scale)
        
        output_map = np.zeros((df.shape[0] , height,width ,  len(cols)+1))

        #pd future warning
        # patches = df[cols].as_matrix()
        patches = df[cols].values
        for row in np.arange(patches.shape[0]):
            for col in np.arange(patches.shape[1]):
#                 img = Image.new('1', (width, height), 0)
                mask = np.zeros((height, width))
                patch = patches[row,col]
                if patch is not np.nan and patch!="-1" and patch !=-1:
                    patch_int = np.array([int(s) for s in patch.split(",")])
    
                    coords = np.reshape((patch_int/scale).astype(int).tolist(), (-1, 2))
                    coords = np.expand_dims(coords , axis = 0)
                    cv2.fillPoly(mask, coords, 1)
                    output_map[row, :,:,col+1] = mask
            label_row = np.argmax(output_map[row, ...],axis=2)
            output_map[row,label_row==0,0] =1  
        return output_map
    def get_contour_patches_coords(self, df ,mode):
        #Get the contour
        scale = self.scale
        patches_coords = None

        if mode == "patches":
            cols = self.patches_cols 
        elif mode == "contour":
            cols = self.contour_col
        else:
            return None 

        if _check_exist_cols(cols , self.all_columns):
            #pd future warning
            # patches = df[cols].as_matrix()
            patches = df[cols].values
            patches_coords = [0]* patches.shape[0]
            for row in np.arange(patches.shape[0]):
                patch_coords = []
                for col in np.arange(patches.shape[1]):
                    patch = patches[row,col]
                    if patch is not np.nan and patch!="-1" and patch !=-1:
                        patch_int = np.array([int(s) for s in patch.split(",")]) //scale
                    else:
                        patch_int = np.array([-1])
                    patch_coords.append(patch_int)
                patches_coords[row] = patch_coords
        #             print(patch_int)
        return patches_coords 
    def get_y_coord(self ,df,scale = 1,coord_only = False):
        """
        Goal: return [batch size, 2 * landmark count]
        """
        l_m_columns = self.coords_cols
        cols_num_per_coord = self.cols_num_per_coord

        #pd future warning
        # y_coord = df[l_m_columns].as_matrix()
        y_coord = df[l_m_columns].values

        return y_coord//scale

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
        Goal: return a series of heatmaps for each key points
        shape:  [batchsize, heatmap_height, heatmap_width, landmark count]
        """
        l_m_columns = self.coords_cols
        cols_num_per_coord = self.cols_num_per_coord

        #pd future warnings
        # y_coord = df[l_m_columns].as_matrix()
        y_coord = df[l_m_columns].values

        lm_cnt = self.lm_cnt

        df_size = y_coord.shape[0]

        real_scale = self.scale * self.heatmap_scale
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

    def get_y_vis_map(self ,df):
        """
        Goal: return a series of masks for each key points
        if there is a key point, the mask is one matrix
        if there is not a key point, the mask is zero matrix
        shape:  [batchsize, heatmap_height, heatmap_width, landmark count]
        """
        l_m_columns = self.coords_cols

        #pd future warnings
        # y_coord = df[l_m_columns].as_matrix()
        y_coord = df[l_m_columns].values
        
        lm_cnt = self.lm_cnt
        df_size = y_coord.shape[0]


        real_scale = self.scale * self.heatmap_scale
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
            
            
            ## Read images
            filename = folder+row[self.file_name_col]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hei = img.shape[0]
            img_wid = img.shape[1]
            if row['aug']==True:    
                # Make sure the transformation for each is same.
                rand_lib.seed(idx)            
                #-----Translate----
                if self.aug_option['trans']:
    #                 if False:
                    min_offset = 100
                    max_offset = 800
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
    #                                 print(idx, "outside")
                                trans_lr=0
                                trans_ud = 0
                                row= old_row
                                break
                    M = np.float32([[1,0,trans_lr],[0,1,trans_ud]])

                    img = cv2.warpAffine(img,M,(img_wid,img_hei))

                ##-----Scale------
                if self.aug_option['scale']:
    #                 if False:
                    #scale value:
                    scale_ratio = randint(10,15)/10
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
                # ---------Rotate--------
                if self.aug_option['rot']:
    #                 if False:    
                    angle_bound = 30
                    angle = randint(-angle_bound,angle_bound)
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
    #                                 print(idx, "outside")
                                angle=0

                                row= old_row
                                break
                    M = cv2.getRotationMatrix2D((img_wid/2,img_hei/2),angle,1)
                    img = cv2.warpAffine(img,M,(img_wid,img_hei))
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
#             print("new row", row[self.coords_cols].values)
#             print("old rows", old_row[self.coords_cols].values)
            df.loc[idx,:] = row      
            img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
            x_all[img_id,:,:,:] = img
            img_id+=1
            
        return x_all.astype('uint8'),df

    def get_x_masks_aug(self,x ,masks):
        # x [batch_size ,height, width, 3]
        # mask [batch_size ,height, width, 3]
        
        df_size =  x.shape[0]
        img_wid = x.shape[2]
        img_hei =x.shape[1]
        n_cl = masks.shape[-1]
        for idx in range(df_size):
            img = x[idx,...].copy()
            mask = masks[idx,...].copy()
            
            aug_prob = 1# random()  
            if aug_prob > 0.5:
                
                #-----Translate----
                if self.aug_option['trans']:
                    min_offset = 0
                    max_offset = 1000//self.scale
                    trans_lr = choice([randint(min_offset,max_offset) , randint(-max_offset,-min_offset)] ) #left/right (i.e. 5/-5)

                    trans_ud = choice([randint(min_offset,max_offset) , randint(-max_offset,-min_offset)] ) #up/down (i.e. 5/-5)
                    
                    M = np.float32([[1,0,trans_lr],[0,1,trans_ud]])
        
                    img = cv2.warpAffine(img,M,(img_wid,img_hei))
                    mask  = cv2.warpAffine(mask,M,(img_wid,img_hei))
    #                 img = img.transform(img.size, Image.AFFINE, (1, 0, trans_lr, 0, 1, trans_ud))
                           
                # ---------Rotate--------
                if self.aug_option['rot']:
                    angle_bound = 20
                    angle = randint(-angle_bound,angle_bound)

#                     print(angle)
                    radian = math.pi/180*angle

                    M = cv2.getRotationMatrix2D((img_wid/2,img_hei/2),angle,1)
                    img = cv2.warpAffine(img,M,(img_wid,img_hei))
#                     img = img.rotate(angle)
                    mask  = cv2.warpAffine(mask,M,(img_wid,img_hei))
            
                        ##-----Scale------
                if self.aug_option['scale']:
                    #scale value:
                    scale_ratio = randint(5,15)/10
#                     scale_ratio = 2.0
#                     print(scale_ratio )
                    new_scaled_width = (int(img_wid *scale_ratio))
                    new_scaled_height = (int(img_hei * scale_ratio))
                    
                    img = cv2.resize(img, dsize=(new_scaled_width,new_scaled_height),
                                     interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, dsize=(new_scaled_width,new_scaled_height),
                                     interpolation=cv2.INTER_LINEAR)
                    # shrinking and padding
                    if scale_ratio<1:
                        delta_w = img_wid - new_scaled_width
                        delta_h = img_hei - new_scaled_height

                        img= cv2.copyMakeBorder(img,delta_h//2,delta_h-(delta_h//2),
                                                delta_w//2,delta_w-(delta_w//2),
                                                cv2.BORDER_CONSTANT,value=[0,0,0])
                        mask = np.pad(mask ,((delta_h//2,delta_h-(delta_h//2)) ,
                                             (delta_w//2,delta_w-(delta_w//2)),
                                             (0,0)),
                                            'constant' , constant_values=0)
#                         mask= cv2.copyMakeBorder(mask,delta_h//2,delta_h-(delta_h//2),
#                                                 delta_w//2,delta_w-(delta_w//2),
#                                                 cv2.BORDER_CONSTANT,value=[0]*n_cl)
                    elif scale_ratio>1:
                        delta_w =  new_scaled_width - img_wid
                        delta_h =  new_scaled_height - img_hei
                        img = img[delta_h//2:delta_h//2+img_hei ,  delta_w//2: delta_w//2+img_wid,:]
                        mask = mask[delta_h//2:delta_h//2+img_hei ,  delta_w//2: delta_w//2+img_wid,:]
        
            seg_aug = np.argmax(mask , axis=2)
            cl_aug, n_cl_aug = extract_classes(seg_aug)
            
            seg_old = np.argmax(masks[idx,...], axis = -1)
            cl_old, n_cl_old = extract_classes(seg_old)
#             print(cl_aug, cl_old , n_cl)
#             print(np.array_equal(cl_aug, cl_old))
            #After augmentation, check the classes if is equal to original classes
        
            if mask.shape[-1]== n_cl and np.array_equal(cl_aug, cl_old):
                x[idx,...] = img
                masks[idx,...] = mask
        
        return x, masks

def _check_exist_cols(cols , total_cols):
    return set(cols).issubset(set(total_cols))

'''
Exceptions
'''
class DataInputErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)