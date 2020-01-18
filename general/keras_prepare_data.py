import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(1000):
    random_younger = random.randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = random.randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    random_younger = random.randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = random.randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)


train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1))

for i in scaled_train_samples:
        print(i)
