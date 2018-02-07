# Thoughts & Ideas
* add column to data frame of categories (from other csv file)... Actually, categories are already encoded. Some belong to multiple. Encode string to float number and add to feature vector.
  * Idea: include all categories encodings in feature vector... will have MAX_CATEGORIES_VID_IS_IN encodings. If a given video doesn’t have that many, assign the rest 0.0s... or something else. This part is tricky because we don’t want entries like [1,31,0,0,0,...] and [27,147,0,0,0,...] to be classified as similar just because they have 0s in common.
  * Actually... genius idea. For starters,scale all features to be between 0 and 1
  * For the categories, append a np.array of size NUM_CATEGORIES to each feature vector such that if vid is in category i array[i] = 1.0 else 0.0 (done by simple numpy.zeros())
  * As for scaling to millions of videos, turn this into bitmap
* Abandon the image classification thing. Not going to work and not worth the extra effort. Do it by classifying the text (maybe try pattern matching) and using categories
* First, I need to make this whole thing work based on an input ID... I should probably use a dictionary. Key by ID, Values are features vectors. In this way, I can easily locate a feature vector by its ID
* For starters, ID ==> feature vector ==> cluster ID, THEN I can use https://stackoverflow.com/questions/36195457/python-sklearn-kmeans-how-to-get-the-values-in-the-cluster to get all feature vectors in that cluster. Return 10 of them... if not enough, throw error
* Then, need to get the ID’s FROM the feature vectors... this is a very interesting question. How do I go from feature vector back to ID number? Obviously I don’t want to use the ID as a feature... but how else? 