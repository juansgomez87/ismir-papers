
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd
import re

import argparse
import tqdm
import pdb


def replacer(initial_string, ch,
             replacing_character, occurrence):
    str_list = list(initial_string)
    new_list = []

    cnt_oc = 0
    for char in str_list:
        if char == ch:
            cnt_oc += 1
            if cnt_oc % occurrence == 0:
                new_list.append(replacing_character)
            else:
                new_list.append(char)
        else:
            new_list.append(char)
    return ''.join(new_list)


def get_data_papers(url, year):
    print('Retrieving data from {}!'.format(url))

    options = webdriver.SafariOptions()
    options.page_load_strategy = 'normal'
    time.sleep(1)
    driver = webdriver.Safari(options=options)
    driver.set_window_position(2000, 0)
    actions = ActionChains(driver)
    driver.maximize_window()
    driver.get(url)
    time.sleep(1)

    if year == 2020 or year == 2022:
        titles = []
        authors = []
        keywords = []
        abstracts = []
        affiliations = []
        links = []
        links_heads = driver.find_elements(By.XPATH, '//a[@href]')
        post_links = [_.get_attribute('href') for _ in links_heads if _.get_attribute('href').find('poster') >= 1]

        for link in tqdm.tqdm(post_links):
            driver.get(link)
            time.sleep(1)

            if year == 2020:
                auth = driver.find_elements(By.XPATH, '//h3')[0].text.replace('\n', '').replace('   ', '').strip()
                tit = driver.find_elements(By.XPATH, '//h2')[0].text.replace('\n', '').strip()
                texts = driver.find_elements(By.CLASS_NAME, 'card-text')
                texts = [_.text for _ in texts]
                keys = texts[0].split('Keywords:')[-1].replace('\n', '').replace('   ', '').strip()
                abs = texts[1].split('Abstract:')[-1].replace('\n', '').replace('   ', '').strip()
            elif year == 2022:
                auth = driver.find_elements(By.XPATH, '//h3')[0].text.replace('\n', '').replace('   ', '').replace('*','').strip()
                auth = replacer(auth, ',', ';', 2)
                tit = re.split(r"\: ", driver.find_elements(By.XPATH, '//h2')[0].text.replace('\n', '').strip(), maxsplit=1)[-1]
                texts = driver.find_elements(By.CLASS_NAME, 'card-text')
                texts = [_.text for _ in texts]
                keys = texts[0].split('Subjects (starting with primary):')[-1].replace('\n', '').replace('   ','').strip()
                keys = '; '.join([_.split(' -> ')[-1] for _ in keys.split(' ; ')])
                abs = texts[1].split('Abstract:')[-1].replace('\n', '').replace('   ', '').strip().replace(' Direct link to video', '')


            authors.append(auth)
            titles.append(tit)
            keywords.append(keys)
            abstracts.append(abs)
            links.append(link)

            # time.sleep(1)
            buttons = driver.find_elements(By.XPATH, '//button')
            paper_but = [_ for _ in buttons if _.text.find('Paper') > 0]
            # actions.move_to_element(paper_but[0]).click(paper_but[0]).perform()
            driver.execute_script('arguments[0].click();', paper_but[0])

            affil = []
            for person in auth.split(';'):
                aff_person = input('Input affiliation person {}\n'.format(person))
                affil.append('{}, {}'.format(person, aff_person))

            affiliations.append('; '.join(affil))
            print('Authors: {}'.format(auth))
            print('Title: {}'.format(tit))
            print('Keywords: {}'.format(keys))
            print('Abstract: {}'.format(abs))
            print('Affilitions {}'.format('; '.join(affil)))

        new_df = pd.DataFrame({'Authors': authors, 'Titles': titles, 'Year': year, 'Link': links, 'Authors with affiliations': affiliations, 'Author Keywords': keywords})
        new_df.to_csv('data/scrape_{}.csv'.format(year))

    elif year == 2023 or year == 2024:
        titles = []
        authors = []
        keywords = []
        abstracts = []
        auth_with_aff = []
        links = []
        affiliations = []
        links_heads = driver.find_elements(By.XPATH, '//a[@href]')
        post_links = [_.get_attribute('href') for _ in links_heads if _.get_attribute('href').find('poster') >= 1]

        for link in tqdm.tqdm(post_links):
            driver.get(link)
            # time.sleep(1)

            auth = driver.find_elements(By.XPATH, '//h3')[0].text.replace('\n', '').replace('   ', '').replace('*', '').strip()
            if auth.find('Shunsuke Yoshida') > 0:
                pdb.set_trace()
            if auth.find('))') > 0:
                affil = re.findall(r"\(([^()]+\(.*?\))\)", auth)
            else:
                affil = re.findall(r'\(.*?\)', auth)
                affil = [_.replace('(', '').replace(')', '') for _ in affil]

            auth = re.sub("\(.*?\)", '', auth)
            auth = auth.replace(' , ', '; ').replace(' ; ', '; ').strip()
            tit = re.split(r"\: ", driver.find_elements(By.XPATH, '//h2')[0].text.replace('\n', '').strip(), maxsplit=1)[-1]
            texts = driver.find_elements(By.CLASS_NAME, 'card-text')
            texts = [_.text for _ in texts]
            keys = texts[0].split('Subjects (starting with primary):')[-1].replace('\n', '').replace('   ', '').strip()
            keys = '; '.join([_.split(' -> ')[-1] for _ in keys.split(' ; ')])
            abs = texts[1].split('Abstract:')[-1].replace('\n', '').strip().replace('  If the video does not load properly please use the direct link to video', '')

            authors.append(auth)
            titles.append(tit)
            keywords.append(keys)
            abstracts.append(abs)
            links.append(link)

            aff = ['{}, {}'.format(p, a) for p, a in zip(auth.split('; '), affil)]
            # for person in zip(auth.split(';'), affil):
            #     aff.append('{}, {}'.format(person, affil))

            
            org = [_.split(',')[-1] for _ in aff]
            if len(org) == 0:
                aff = ['TISMIR']
                org = ['TISMIR']


            auth_with_aff.append(';'.join(aff))
            affiliations.append(';'.join(org))

            print('Authors: {}'.format(auth))
            print('Title: {}'.format(tit))
            print('Keywords: {}'.format(keys))
            print('Abstract: {}'.format(abs))
            print('Affilitions: {}'.format('; '.join(aff)))


        new_df = pd.DataFrame({'Authors': authors, 'Titles': titles, 'Year': year, 'Link': links, 'Authors with affiliations': auth_with_aff, 'Author Keywords': keywords, 'Abstract': abstracts, 'aff_names': affiliations})
        new_df.to_csv('data/scrape_{}.csv'.format(year))

    elif year == 2021:
        pdb.set_trace()

if __name__ == '__main__':
    # run safaridriver --enable
    parser = argparse.ArgumentParser()

    parser.add_argument('--year',
                        help='Input year to scrape (from 2020 to 2023)',
                        type=int,
                        required=True)
    args = parser.parse_args()

    if args.year == 2020:
        url = 'https://program.ismir2020.net/papers.html?filter=keywords'
    elif args.year == 2021:
        url = 'https://ismir2021.ismir.net/papers/'
    elif args.year == 2022:
        url = 'https://ismir2022program.ismir.net/papers.html?filter=keywords'
    elif args.year == 2023:
        url = 'http://ismir2023program.ismir.net/papers.html?filter=keywords'
    elif args.year == 2024:
        url = 'http://ismir2024program.ismir.net/papers.html?filter=keywords'

    # df = pd.read_csv('data/scopus-1.csv')

    get_data_papers(url, args.year)

    pdb.set_trace()

