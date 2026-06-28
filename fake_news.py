from preprocessing import clean_text
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import clean_text
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
print(fake.shape)
print(true.shape)
print(fake.columns)
print(fake.info())
print(fake.isnull().sum())
print(true.isnull().sum())
fake["label"] = 0
true["label"] = 1
news = pd.concat([fake,true],ignore_index=True)
news = news.sample(frac=1,random_state=42)
news.reset_index(drop=True,inplace=True)
news = news.drop_duplicates()
print("Before Cleaning:")
print(news["text"].iloc[0])

news["text"] = news["text"].apply(clean_text)

print("\nAfter Cleaning:")
print(news["text"].iloc[0])
news.to_csv(
    "data/clean_news.csv",
    index=False
)
news["label"].value_counts().plot(kind="bar")

plt.show()
news["length"]=news["text"].apply(len)

plt.hist(news["length"],bins=50)

plt.show()
