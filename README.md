

# Project plumage
A labelling project that collect colour information across all birds.
The website of [project plumage](https://www.zooniverse.org/projects/ghthomas/project-plumage)
People can help to label images on the websit

## Data and Labels
TODO

## Points features:
### Metrics
Read and analyse metrics from different configurations or hyperparameters.
**Input**: Dataframe format for metric evaluation
row|index | numeric metric | List metric | config_1 | ... | config_n
---|------|----------------|-------------|----------|-----|-------
0|config_1-value_1;<br>config_n-value_n|value|[V1,V2,...,Vn]|config_value|...|config_value

* Visualize the metrics on different configurations. Use bar plot and radar plot.
* Which factor is most important for the accuracy? Stats analysis. e.g. Anova on different configuration.

 ### Result
**Input** : Dataframe of orignal format of image labels 

* Output metrics between prediction and ground truth value
* Extract colour from the labels
* Compare the extracted colour information. Pearson correlate between prediction and GT
* Generate polygons from predicted centroids. Use different methods

### Result visualization

* Write images with GT labels, prediction labels, and GT and prediction labels.

## TODO list
### Points
* [x] Update augmentation in datainput.
* [ ] Check the augmentation quality
* [ ] Find better way to generate predicted patches.
* [ ] Different extract colour information methods. (cluster or sampling)

### Training
* [ ] Train different resolution, from high to low.
* [ ] Use different size of training set.
* [ ] Find the most effected factor for improving the result.

### Visualize images
* ground truth 
* prediction 
* GT and prediction
* point only, standard only, polygon only

### Project
