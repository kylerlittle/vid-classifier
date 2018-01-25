

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
print("\nStructure:")
print(video_df.head(5))   # first 5 rows
fig = plt.figure()
video_df.hist(column='duration')
#plt.show()

'''
Step 1: Data Processing
In this stage, I need to "clean up" the data. This involves removing or accounting for null fields and
removing invaluable information.
'''
# There are 105 missing entries in 'caption' field; need to fix this


'''
Step 2: Feature Selection
In this stage, I determine which fields best classify the dataset and why. I ascertain other features from the given fields.
frequency–inverse document frequency of "description" field
---------> first need to clean this field up... there are null values & some of them just don't make any sense
---------> https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66
---------> http://scikit-learn.org/stable/modules/feature_extraction.html
duration field
total_comments field 
thumbnail field
---------> extract a few features from the thumbnails such as: brightness, exposure, translation, rotation, scale, symmetry, intensity
'''
from PIL import Image
import requests
from io import BytesIO
#from skimage import io
import urllib.request
import binascii

def extract_im_brightness(URL):
    data = urllib.request.urlopen(URL).read()
    #r_data = binascii.unhexlify(data)
    response = requests.get(URL)
    bytesIOobj = BytesIO( bytes(response.content))
    bytesIOobj.seek(0)
    byteIm = bytesIOobj.read()
    #bytesIOobj.seek(0)
    img = Image.open(byteIm)
    img.save("test.png")
    '''
    response = requests.get(URL)
    print(response.headers)   # confirms image type is 'bytes'
    bytesIOobj = BytesIO(response.content)   # default mode is 'r'
    Image.open(bytesIOobj)

    # Read response.content into BytesIO buffer
    BytesIOimage = BytesIO(response.content)
    
    # Reset file cursor
    BytesIOimage.seek(0)
    
    # Before opening image, I need to write BytesIOimage locally as a FILE-- not a BytesIO object
    with open('temp.jpg', 'rb') as f:
        f.write(BytesIOimage)
    im = Image.open('temp.jpg')
    '''
    '''
    with urllib.request.urlopen(URL) as url:
        f = BytesIO(url.read())
    img = Image.open(f)
    img.show()


    (filename, _) = urlretrieve(URL)
    print(filename)
    # Convert filename to jpg

    response = requests.get(URL)
    # Open file
    im = Image.frombytes(response.content)
    '''

    
    '''
    
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # return the image
    return image
    '''


    
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_caption_tf_idf(caption_text):
    '''
    vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
    X = vect.fit_transform(caption_text)
    print(X)
    return X
    '''
    pass


def vectorize(row):
    # Initialize empty feature_vector as list
    feature_vector = []

    # Extract Features
    extract_im_brightness(row['thumbnail'])
    #feature_vector.append(extract_caption_tf_idf(row['caption']))

    # Return converted numpy array
    #return np.array(feature_vector)   

number_of_features = 10
feature_matrix = np.zeros((video_df.shape[0], number_of_features))
for index, row in video_df.iterrows():
    # for now, just test on first row
    if index == 0:
        #feature_matrix[index] =
        vectorize(row)


'''
Step 3: Modeling
In this stage, I determine which machine learning model is appropriate for the situation.

For starters, it is clear I must use an unsupervised machine learning model to classify
the data. Although I am given the "categories" that each video is put in, videos in 
the same category aren't necessarily "similar." Furthermore, I don't need a model that
predicts output based on unknown input. I'm only using the input data to determine
the relationship between the data points, exactly what unsupervising machine learning
attempts to do.

The best machine learning model to do such a job is clustering. Clustering
attempts to segregate groups of data points with similar traits and put them
into clusters. This perfectly aligns with the goal of the project.

For a given input (from the supplied data set), the model should output its 10 closest neighbors.
'''



'''
Step 4: Testing
Output results for the following clip_ids:
14434107, 249393804, 71964690, 78106175, 228236677, 11374425, 93951774,
35616659, 112360862, 116368
'''
