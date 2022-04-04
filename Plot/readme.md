# Microbiome plots

## Triple corr
Triple corr is our standart correlation plot. It's consists of 3 subplots: 1. The taxonomy tree showing the bakterias arranged in a tree by their taxonomy clustering and painted by their correlation to the tag. There is a main example in the file, but generaly the function recives a features dataframe x, and tag series y.
</br>
<img src="https://drive.google.com/uc?export=view&id=1N8O7922ngq6DxxXu56q7mg7m8eIGiHyG" width="500" height="400" />
</br>
## Real and shuffled hist
The Real and shuffled hist is one of the 3 subplots in the Triple corr plot. It's the bottom left one and it performs a histogram of the real correlation values vs the correlations when shuffling the tags.

## Plot positive negative bars
The positive negative bars is one of the 3 subplots in the Triple corr plot. It's the top left one and it shows the most correlative bakterias to the tag. only the top 1/2 percents are of the most correlative bakterias in abs are taken and shown in the plot

## Plot boxplot
The boxplot file plots a boxplot of the groups showing the median and 25/75 samples of each group by a given diversity csv file and a grouping file with either 1 or 2 columns.

<img src="https://drive.google.com/uc?export=view&id=1cTDO9pNDF4gX4WHJc2gRPoiZhWYEB8n8" width="400" />
##  New taxtree draw
Aid functions for the build and display of the taxonomy tree, including some new features such as heat-painted nodes and nodes in different sizes.

## Plot 3D
Plotting either a 3D Umap or a 3D Pcoa. the function recives either a 2D matrix or a dataframe(usually, a computed diversity matrix) and plots it's 3D embedding.![68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d314e384f373932326e6771364478785875353671376d67376d3865494769487947](https://user-images.githubusercontent.com/69215363/161541893-8fc017e1-e86e-45f3-a6fb-7105521faf89.png)


<img src="https://drive.google.com/uc?export=view&id=1r33fVjR3WJbvCI0IttF15VKgFA9PmdY6" width="400"  />

## Plot bakteria interaction network
Plotting the time interactions between the bakterias in a time points formed dataset. tte function recives a path to a csv squared file where the A[i,j] represents the affection of bakteria i in time point t on the bakteria j in time point t+1(mostly a coefficients matrix produced by a linear model). the plot consists from 2 subplots. The right one is a clustered map showing the interaction between the bakterias in time predictions where blue square represents a positive influence(a grown in the number of bakteria j in time t+1 compared to ti,e t) and a negative influence is represented by a red square. each bakteria was given a number and on the left plot there ia a visualy representation of the clustered map. Each circle is a bakteria(same nuber as given to that bakteria in the right table) and again a blue arrow is a positive influence and a red arrow is represents a negative influence.

<img src="https://drive.google.com/uc?export=view&id=1FDZndw_qeMb8p_gIM5wWBspeO0o4iy-c" width="500"  />
</br>
