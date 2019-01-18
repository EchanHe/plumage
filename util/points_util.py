import numpy as np
import cv2

def heatmap_to_coord(heatmaps , ori_width , ori_height):
    """
    # Goals: transfer heatmaps to the size given

    Params:
        heatmaps: heatmaps from pose esitmation method, 
            Shape(df_size, heatmap_height,heatmap_width, heatmap channel)
        ori_width: The width you want to transfer back
        ori_height: The height you want to transfer back

    return:
        The coordinates found on heatmap, eg [x1,y1,x2,y2,...,xn,yn]
        Shape: (df_size,cnt_size*2)
    """
    df_size = heatmaps.shape[0]
    cnt_size = heatmaps.shape[3]
    output_result = np.ones((df_size,cnt_size*2))
    for i in range(df_size):
        for j in range(cnt_size):
            heat_map = heatmaps[i,:,:,j]
            ori_heatmap = cv2.resize(heat_map, dsize=(ori_width, ori_height),interpolation = cv2.INTER_NEAREST)
            map_shape = np.unravel_index(np.argmax(ori_heatmap, axis=None), ori_heatmap.shape)
            output_result[i,j*2+0] = map_shape[1] + 1
            output_result[i,j*2+1] = map_shape[0] + 1
    return output_result

# Goals: transfer the coords into a rectangle coords

#return a matrix of rectangle x,y strings in [batch, n_patches]
def pred_coords_to_patches(pred_coords , half_width =10, half_height=10 , ignore_coords =10):

    pred_coords = pred_coords[:,ignore_coords:]

    total_patches = pred_coords.shape[-1]//2
    x_coords = pred_coords[:,range(0,total_patches*2,2)]
    y_coords = pred_coords[:,range(1,total_patches*2,2)]

    upper_x_coords = x_coords + half_width
    upper_y_coords = y_coords + half_height
    lower_x_coords = x_coords - half_width
    lower_y_coords = y_coords - half_height

    pred_patches = np.empty((pred_coords.shape[0] , total_patches) , dtype = object)

    # [pred_coords+4000 , pred_coords-4000 ]
    # y_coords.astype(str) 
    ## Make 4 verticies of a rectangle near the centric.
    for i in range(pred_coords.shape[0]):
        for j in range(total_patches):
            u_x = upper_x_coords[i,j]
            u_y = upper_y_coords[i,j]
            l_x = lower_x_coords[i,j]
            l_y = lower_y_coords[i,j]
            patch_coords = [l_x,u_y,u_x,u_y,u_x,l_y,l_x,l_y]
            pred_patches[i, j] = ','.join(str(int(c)) for c in patch_coords )
    return pred_patches



#return a matrix of rectangle x,y strings in [batch, n_patches]
def create_rect_on_coords(coords , width =10, height=10 , ignore_coords =10):
    """
    Goal: a N_D array of rectangle using coords as center 

    params:
        coords: [batch, x1,y1,..., xn,yn]

    return:
        x,y strings in [batch, n_patches]
    """

    # coords = coords[:,ignore_coords:]

    total_patches = coords.shape[-1]//2
    x_coords = coords[:,range(0,total_patches*2,2)]
    y_coords = coords[:,range(1,total_patches*2,2)]

    half_width = width//2
    half_height = height //2

    upper_x_coords = x_coords + half_width
    upper_y_coords = y_coords + half_height
    lower_x_coords = x_coords - half_width
    lower_y_coords = y_coords - half_height

    pred_patches = np.empty((coords.shape[0] , total_patches) , dtype = object)

    # [coords+4000 , coords-4000 ]
    # y_coords.astype(str) 
    ## Make 4 verticies of a rectangle near the centric.
    for i in range(coords.shape[0]):
        for j in range(total_patches):
            u_x = upper_x_coords[i,j]
            u_y = upper_y_coords[i,j]
            l_x = lower_x_coords[i,j]
            l_y = lower_y_coords[i,j]
            patch_coords = [l_x,u_x,l_y,u_y]
            pred_patches[i, j] = ','.join(str(int(c)) for c in patch_coords )
    pred_patches[x_coords == -1] =-1
    return pred_patches