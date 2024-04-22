# Multilabel Image Classification Model 

The original dataset is a 45GB folder containing 1920x1080 screenshots of different videogames from the Steam catalogue. There is also a JSON file with different attributes, including price (numeric), tags (categorical), and game names. The JSON file also includes the names of multiple screenshots per game, with the name exactly matching the name in the image folder. The data is formatted in the JSON file as a list of dictionaries, with each dictionary corresponding to a different game, which has the aforementioned attributes. 

The objective is to develop a program that separates the images into a train and test set at the level of the game, so that screenshots belonging to the game are all grouped together entirely in the train set or test set, but there should not be overlap. Additionally, we want to create a model that can predict the tags from the screenshots. 

It should also be noted that the tags are not hot one encoded, so that must also be accomplished. Currently, they are in the format: 'tags': [tag1, tag2, tag3]

The program provided does all of the above, and saves the model into a .keras file so that it can be reimported and tested against new data or implemented directly. 

There is also an optional python file that resizes the original images, which were quite large in storage space, to half size. This reduces the folder size to around 8GB. The 8GB-sized folder is the data used in the actual model.


