import openai
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
df = pd.read_csv('./data/train_data_public.csv').sample(1000)


retrans = pd.DataFrame(columns=['oid', 'en', 'cn'])

client = openai.OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-t1Igves6yH15GLdjpuj6DLHQrbfX7rYuuotZoxrfpDWGdslk',
        )

repeat = 1


for i, row in tqdm(df.iterrows(), total=len(df)):
    data = []
    for j in range(repeat):
        chat_completion = client.chat.completions.create(
                model='gpt-4o-mini',
    messages=[{"role": "system", "content": ''' 你将扮演一个精通英文和擅长中文表达的金融翻译家

    每次我都会给你一句关于用户对于银行及其产品评论的中文：

    请你先作为翻译家，把它翻译成中文，用尽可能地道的英文表达。
    然后你扮演校对者，把它翻译回中文，用尽可能地道的中文口语表达。

    你的回答应该遵循以下的格式，不要额外加引号，保持用五个#分割中英文，不要随便加其他标记和换行符：
    英文部分#####中文部分'''}, { "role":"user", "content": row['text'] } ]
                )

        li = chat_completion.choices[0].message.content.split('#####')

        if(len(li)==2): data.append([row['id'],li[0],li[1]])
        else: data.append([row['id'],str(li), 'ERR'])

    # print(data)
    # print(df.columns)

    new_row = pd.DataFrame(data, columns=retrans.columns)
    retrans = pd.concat([retrans, new_row])
    # print(data,new_row,retrans)

    retrans.to_csv('./data/retrans.csv')
