
# Import appropriate libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Step 0: What does the data look like?
Let's import the data set using pandas into a DataFrame object and ask it some questions.
'''

video_df = pd.read_csv("similar-staff-picks-challenge/similar-staff-picks-challenge-clips.csv")
print("Data shape: " + str(video_df.shape))
print("\nInfo:")
print(video_df.info())
print("\nMissing Entries:")
print(video_df.isnull().sum())


'''
Step 1: Data Processing
In this stage, we need to "clean up" the data. This involves removing or accounting for null fields,
removing invaluable information, and determining which fields are best the classify the dataset.

The twelve features are:

'''

'''
Step 2: Feature Selection
Which ones I will choose & why...

'''


'''
Step 3: Modeling
Which learning model I chose & why...
For starters, it is clear I must use an unsupervised machine learning model to classify
the data. Although I am given the "categories" that each video is put in, videos in 
the same category aren't necessarily "similar." Furthermore, I don't need a model that
predicts output based on unknown input. I'm only using the input data to determine
the relationship between the data points, exactly what unsupervising machine learning
attempts to do.

The best machine learning model to do such a job is clustering. Clustering
attempts to segregate groups of data points with similar traits and put them
into clusters. This perfectly aligns with the goal of the project.
'''


'''
Step 4: Testing
Output results for the following clip_ids:
14434107, 249393804, 71964690, 78106175, 228236677, 11374425, 93951774,
35616659, 112360862, 116368
'''
