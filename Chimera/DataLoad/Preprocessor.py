from pandas import DataFrame
import re
def Prepro(df):
    df['context'] = ApplyContextPrepro(df['context'])

def ApplyContextPrepro(context : DataFrame):
    return context.apply(lambda s: re.sub('[^a-zA-Z]+', '', s))