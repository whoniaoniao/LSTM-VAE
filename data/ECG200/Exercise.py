
import pandas as pd
from scipy.io import arff

data = arff.loadarff('./data/ECG200/ECG200_TEST.arff')
df = pd.DataFrame(data[0])
data1 = arff.loadarff('./data/ECG200/ECG200_TRAIN.arff')
df1= pd.DataFrame(data1[0])
df=df.replace(b'1', 1)
df=df.replace(b'-1',1)
# df=df.replace(b'3', 3)
# df=df.replace(b'6',6)
# df=df.replace(b'4' , 4)


# df=df.replace(b'5',5)
# df=df.replace(b'7' , 7)
df1=df1.replace(b'1',1)
df1=df1.replace(b'-1', -1)

df.to_csv("./data/ECG200/ECG200_TEST.csv",header=False,index=False)
df1.to_csv("./data/ECG200/ECG200_TRAIN.csv",header=False,index=False)
