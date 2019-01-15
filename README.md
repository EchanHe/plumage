

# Project plumage
A labelling project that collect colour information across all birds.
The website of [project plumage](https://www.zooniverse.org/projects/ghthomas/project-plumage)
People can help to label images on the websit

## Data and Labels
### Points:
**Input data format:**
file_name|info_col_1| ... | info_col_n | point_1_x | point_1_y| ... | point_n_x |point_n_y
---|------|---|-----------|-------------|----------|-----|-------|----
`<filename>`.jpg|`<info_1>`|...|`<info_n>`|`x_pt1`|`y_pt1`|...|`x_ptn`|`y_ptn`
**Steps:**
1. Download the raw data from Project plumage
2. use the pre_process.ipynb to process the raw data into the input data format file `input.csv`.
3. If the quality check hasn't been done, Then visualize labels on photos and do quality check.
4. Note down the unqualified photos with their labels into a csv `update_file.csv`.
5.  Re-label these data again (e.g. using Fiji or project plumage) and update new labels into the `update_file.csv`.
6. Update the data file with `update_file.csv` and get `input_updated.csv`

**Files and Usage:**
`./input/process_raw_data.ipynb` is used to process raw data into usable input format data and update the input data using the updated labels.
`./input/Find_data_exception/process_raw_data.ipynb` :
1. Copy the error labelled image into certain folders
2. Find the incorrect entries and copy them into a wait for update csv.
3. Find out which files hasn't been labelled in this dataset
### Outlines
TODO
## File Hierarchy
`Baseline.ipynb` : Create mean or mode for baseline model
`colour_extraction.ipynb` : Extract colour features from pixels
`grid_search_result_analysis.ipynb`: Plot and stats models on the metrics of grid search results.
`Write_labels_images.ipynb`: Save the images and labels by reading the data csv
## Points part:
### Metrics
##### 

### Grid search
Read and analyse metrics from different configurations or hyperparameters.
**Input**: Dataframe format for metric evaluation
row|index | numeric metric | List metric | config_1 | ... | config_n
---|------|----------------|-------------|----------|-----|-------
0|config_1-value_1;<br>config_n-value_n|value|[V1,V2,...,Vn]|config_value|...|config_value

* Visualize the metrics on different configurations. Use bar plot and radar plot.
* Which factor is most important for the accuracy? Stats analysis. e.g. Anova on different configuration.

 ### Result
 Read labels of GT and prediction, then do different polygons and metrics.
**Input** : Dataframe of orignal format of image labels 

* Output metrics between prediction and ground truth value
* Extract colour from the labels
* Compare the extracted colour information. Pearson correlate between prediction and GT
* Generate polygons from predicted centroids. Use different methods

### Result visualization

* Write images and different combinations of GT and prediction labels.

## Network Workflow
### Training or grid search training
[train](https://drive.google.com/uc?export=view&id=18kiBBb5nmqb1JiUUMHTEcsvM9EvYU8LE)

<!---
 1. Read Configuration file.
 2. Read and process the images and labels for later input.
 3. Select model and initialize model with given configurations
 4. If the configurations has grid search options  <br>Loop through all combinations for grid search options
 6. Training
     1. Execute training operation
     2. Save training log
     3. Save Validation log and calculate validation results.
     4. Save trained parameters
5. Create polygon patches with
6. Save the results of metrics and labels.
-->

### Validation and Prediction
 [pred](https://drive.google.com/uc?export=view&id=1K8CY57AXGtP53baeQh5x5h91h-kFDhE5)


## TODO list
### Data
shift the std 2,3,4 to left for a certain calibrator.

### Points
* [ ] seperating training. <br> robust indexing name for `data_input.py, point_io.py, visualize`<br>Divide and conquer.
* [x] Update augmentation in datainput.
* [x] Update the naming system for grid search result.
* [x] The easy version of writing summary into log, encapsulate summary writing in `network.py`
* [x] Put pose estimation methods into `network` class in `network.py`
* [x] Bug for images size can not divided by batch size. [solution](https://stackoverflow.com/questions/53326656/good-ways-to-wrap-around-the-indices-for-slicing-in-pandas-data-frame/53327125#53327125)
* [x] Batch implementation for validation, because of larger data.
* [x] Save log to the correct folder.
* [x] Add weight to histogram
* [x] Select different optimizers (SGD, ADAM) and different decay methods (exponential and cosine restart)
* [ ] Time calculating features
* [ ] Weighted loss
* [ ] learning rate decay on loss plateau, time series problem.t
* [ ] insert rows, for the filter and midlayer in the image log.
* [ ] augmentation pipeline
* [ ] better split method
* [ ] test visualize lib
* [x] Visualization the mid layers _partlly finished, wait for stackoverflow 


### Network
* [ ] Better of use backbone.
r
### Data input
* [x] Wrap the index slice on `get_next_batch_no_random(self)`
* [ ] Check the augmentation quality in `data_input.py`
* [ ] gaussian peak according to different scale
### Analyse result
* [ ] Find better way to generate predicted patches.
* [ ] Different extract colour information methods. (cluster or sampling)
* [ ] IOU calculate as 80 problem.
* [ ] Find the point to line segment.
### Area or polygons
* [ ] Add weight loss for deep lab
* [ ] Try different methods such as mask rcnn.
* [ ] Add Grid search on best hyperparameters
* [ ] Add code Prediction training between different class
* [ ] Read the json data.

Segs to masks() has problem 
AREA to mask index is not same.

### Training
* [ ] Train different resolution, from high to low.
* [ ] Use different size of training set.
* [ ] Find the most effected factor for improving the result.

### Project
* [ ] Calculate the time for training and predicting images on cpu and gpu.
* [ ] Find the best guest polygon and update the centroid method.
* [ ] Add Comments
* [ ] Add robust file dir create (eg pred dir)
* [ ] Program robust column names for generalize in data_input.py
* [ ] write code to check pred and validation are in the same index names
* [ ] polygon patches are too small or too similar for semantic segmentation.


<!---
Done:
TICK Metrics in whether in polygon. (mask correspond to points)
TICK develop read rgb or grey value. in network and data_input
TICK develop the grid search
TICK write the grid search result in csv.

####
work flow of 
The segmentation results are saved in a python dictionary format.

1 layer: filenames of index
2 layer: The ID of the different class (outline, or different colour patches)
3 layer: cols and rows. Each entry contains a list of the column and row index of the segmentation.

Hourglass workflow

train.py
valid.py
prediction.py
grid_search.py  Output the metrics on different configs of hyperparameters

Display functions:

Draw one plot
Draw for many plots from a dataframe
-->
