import pandas as pd
import os
import functools
from dotenv import load_dotenv
import time
import ast
from openai import OpenAI
import tqdm
import pdb

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

def retry_on_ratelimit(max_retries=3, delay=60):
    """
    A simple retry decorator for handling rate limit errors from the OpenAI API.

    :param max_retries: The maximum number of retries.
    :param delay: The delay between retries in seconds.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Rate limit exceeded, retrying in {delay} seconds...")
                    print(e)
                    time.sleep(delay)

            print(f"Failed after {max_retries} retries.")
            return None  # Return None or consider raising an exception after all retries fail

        return wrapper

    return decorator


@retry_on_ratelimit(max_retries=3, delay=60)
def openai_evaluate(list_authors):
    """uses OpenAI gpt-4 to extract affiliations"""
    client = OpenAI(
        # This is the default and can be omitted
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        # model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": '''You will be given a python list of authors and affiliations. You must search the category of the entity and its country. \
                    The categories are education (e.g., university), company, and facility (e.g., research center, governmental agencies, \
                    non-profit). The countries should be in ISO 31166-1 A3 format. The affiliations come from all over the world so expect non-English text as well. \
                    You should only respond back in the form of a Python list: ["<author1 as string>, <affiliation as string>, <country code as string>", "<author2 as string>, <affiliation as string>, <country code as string>"] unless stated otherwise.\
                    For example, if the input is:
                    [“Anja Volk, Utrecht University”, “Tinka Veldhuis, Utrecht University”, “Katrien Foubert, KU Leuven, LUCA School of Arts”]
                    the output should be:
                    [“Anja Volk, Utrecht University, NLD, education”, “Tinka Veldhuis, Utrecht University, NLD, education”, “Katrien Foubert, KU Leuven, LUCA School of Arts, BEL, education”]''',
            },
            {"role": "user", "content": str(list_authors)},
        ],
    )
    try:
        print(ast.literal_eval(response.choices[0].message.content))
        resp = ast.literal_eval(response.choices[0].message.content)
    except:
        print(response.choices[0].message.content)
        pdb.set_trace()
        # resp = ['Qixin Deng, University of Rochester, USA, education', 'Qikai Yang, University of Illinois at Urbana-Champaign, USA, education', 'Ruibin Yuan, CMU, USA, education', 'Yipeng Huang, Multimodal Art Projection Research Community, Unknown, facility', 'Yi Wang, CMU, USA, education', 'Xubo Liu, University of Surrey, GBR, education', 'Zeyue Tian, Hong Kong University of Science and Technology, HKG, education', 'Jiahao Pan, The Hong Kong University of Science and Technology, HKG, education', 'Ge Zhang, University of Michigan, USA, education', 'Hanfeng Lin, Multimodal Art Projection Research Community, Unknown, facility', 'Yizhi Li, The University of Sheffield, GBR, education', 'Yinghao MA, Queen Mary University of London, GBR, education', 'Jie Fu, HKUST, HKG, education', 'Chenghua Lin, University of Manchester, GBR, education', 'Emmanouil Benetos, Queen Mary University of London, GBR, education', 'Wenwu Wang, University of Surrey, GBR, education', 'Guangyu Xia, NYU Shanghai, CHN, education', 'Wei Xue, The Hong Kong University of Science and Technology, HKG, education', 'Yike Guo, Hong Kong University of Science and Technology, HKG, education']

    return resp

if __name__ == "__main__":

    df = pd.read_csv('data/data_2024_authors.tsv', sep='\t', header=None)
    df.drop(columns=df.columns[0], inplace=True)

    all_new_data = []
    for i in tqdm.tqdm(range(df.shape[0])):
        this_list = df.iloc[i].dropna().tolist()
        new_data = openai_evaluate(this_list)
        this_str = '; '.join(new_data)
        all_new_data.append(this_str)
    
    df['full_data'] = all_new_data

    pdb.set_trace()
    countries = [[_.split(',')[2].strip() for _ in i.split(';')] for i in all_new_data]
    count = [', '.join(c) for c in countries]

    categories = [[_.split(',')[-1].strip() for _ in i.split(';')] for i in all_new_data]
    cats = [', '.join(c) for c in categories]

    affiliations = [[_.split(',')[1].strip() for _ in i.split(';')] for i in all_new_data]
    affs = [', '.join(c) for c in affiliations]

    df['countries'] = count
    df['categories'] = cats
    df['affs_names'] = affs

    df.to_csv('data/data_2024_authors_with_data.tsv', sep='\t')
    pdb.set_trace()