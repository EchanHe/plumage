

# Project plumage
A labelling project that collect colour information across all birds.
The website of [project plumage](https://www.zooniverse.org/projects/ghthomas/project-plumage)
People can help to label images on the websit

## Data and Labels
TODO
## File Hierarchy
TODO
## Points features:
### Metrics
Read and analyse metrics from different configurations or hyperparameters.
**Input**: Dataframe format for metric evaluation

row|index | numeric metric | List metric | config_1 | ... | config_n
---|------|----------------|-------------|----------|-----|-------
0|config_1-value_1;<br/>config_n-value_n|value|V1,V2,...,Vn|config_value|...|config_value

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
### Training
 1. Read Configuration file.
 2. Read and process the images and labels for later input.
 3. Select model and initialize model with given configurations
 4. Training
     1. Execute training operation
     2. Save training log
     3. Save Validation log and calculate validation results.
     4. Save trained parameters
5. Create polygon patches with
6. Save the results of metrics and labels.

### Grid Search
1. Train the networks with different combination of hyperparameters.
2. Evaluate and trained networks on validation set with different metrics
3. Write the evaluated metric into file.
### Validation

### Prediction

## TODO list
### Points
* [x] Update augmentation in datainput.
* [x] Update the naming system for grid search result.
* [ ] Check the augmentation quality
* [ ] Find better way to generate predicted patches.
* [ ] Different extract colour information methods. (cluster or sampling)
* [ ] Bug for images size can not divided by batch size.
* [ ] The easy version of writing summary into log.
* [ ] Batch implementation for validation, because of larger data.
* [ ] Time calculating features
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
