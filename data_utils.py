from openai import OpenAI
import pandas as pd
import re
import numpy as np

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="YOUR ACCESS TOKEN HERE",
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def find_exp(df):
     df.head()
     exps, scores = [], []
     for i in range(len(df)):
          e_i = df.iloc[i]['explaination']
          exps.append(re.search(r"(?<=[).*?(?=])", e_i).group(0))
     return exps, scores 

def text_trajectory(df, userid):
    weekdayDict = {
        0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday', 3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday',
    }
    item = df.loc[df['UserId']==userid]
    item['CheckinTime'] = pd.to_datetime(item['CheckinTime'])
    item['dayofweek'] = item.CheckinTime.apply(lambda x: weekdayDict[x.dayofweek])
    sequence = ""
    prev_X, prev_Y = None, None
    index = [i for i in range(len(item))]
    for i in range(len(item)):
        cur_item = item.iloc[i]
        CheckinTime, VenueType, dayofweek, X, Y = cur_item['CheckinTime'], cur_item['VenueType'], cur_item['dayofweek'], cur_item['X'], cur_item['Y']
        # add distance
        if i > 0:
            dist = np.sqrt((X-prev_X)**2 + (Y-prev_Y)**2)
            sequence += ', {:.1f} km ->'.format(dist/(10**3))
        prev_X, prev_Y = X, Y
        
        # form trajectory sequnece
        time = ':'.join(str(CheckinTime).split(' ')[1].split(':')[:-1])
        sequence += f"{dayofweek} {str(time)}, {VenueType}"

    return sequence, index
