
"""
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

skin_df = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')


SIZE=32


le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
print(list(le.classes_))
 
skin_df['label'] = le.transform(skin_df["dx"]) 
print(skin_df.sample(10))



fig = plt.figure(figsize=(14,10))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type');

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')


ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.show()



from sklearn.utils import resample
print(skin_df['label'].value_counts())

#Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
#Separate each classes, resample, and combine back into single dataframe

test_0  = skin_df[skin_df['label'] == 0]
test_1 = skin_df[skin_df['label'] == 1]
test_2 = skin_df[skin_df['label'] == 2]
test_3 = skin_df[skin_df['label'] == 3]
test_4= skin_df[skin_df['label'] == 4]
test_5 = skin_df[skin_df['label'] == 5]
test_6 = skin_df[skin_df['label'] == 6]

n_samples=1000 
test_0_balanced = resample(test_0, replace=True, n_samples=n_samples, random_state=42) 
test_1_balanced = resample(test_1, replace=True, n_samples=n_samples, random_state=42) 
test_2_balanced = resample(test_2, replace=True, n_samples=n_samples, random_state=42)
test_3_balanced = resample(test_3, replace=True, n_samples=n_samples, random_state=42)
test_4_balanced = resample(test_4, replace=True, n_samples=n_samples, random_state=42)
test_5_balanced = resample(test_5, replace=True, n_samples=n_samples, random_state=42)
test_6_balanced = resample(test_6, replace=True, n_samples=n_samples, random_state=42)

skin_test_balanced = pd.concat([test_0_balanced, test_1_balanced, 
                              test_2_balanced, test_3_balanced, 
                              test_4_balanced, test_5_balanced, test_6_balanced])


print(skin_test_balanced['label'].value_counts())


image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('data/HAM10000/', '*', '*.jpg'))}

skin_test_balanced['path'] = skin_df['image_id'].map(image_path.get)

skin_test_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


n_samples = 7  


fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_test_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')


X = np.asarray(skin_test_balanced['image'].tolist())
X = X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=skin_df_balanced['label'] 
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y_cat, test_size=0.95, random_state=42)

x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.05, random_state=42)

clf = ak.ImageClassifier(max_trials=35) #MaxTrials - max. number of keras models to try
clf.fit(x_train_auto, y_train_auto, epochs=35)


#Evaluate the classifier on test data
_, acc = clf.evaluate(x_valid, y_valid)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('cifar_model.h5')


score = model.evaluate(x_valid, y_valid)
print('Test accuracy:', score[1])
