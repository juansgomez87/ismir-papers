import requests
from bs4 import BeautifulSoup
import re
import os
import json

import argparse
import tqdm
import pdb


def get_pdfs(url, root):
    print('Retrieving files from ISMIR website!')
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    raw_html = requests.get(url, headers={'User-Agent': user_agent})
    soup = BeautifulSoup(raw_html.content, "html.parser")
    links = soup.find_all('a', href=re.compile(r'(.pdf)'))
    pdb.set_trace()

    url_list = [_['href'] for _ in links]

    # create dir
    if not os.path.exists(root):
        os.makedirs(root)

    for i, pdf in enumerate(tqdm.tqdm(url_list)):
        filename = os.path.join(root, '{:03}.pdf'.format(i))
        with open(filename, 'wb') as f:
            f.write(requests.get(pdf, headers={'User-Agent': user_agent}).content)


def get_pdfs_zenodo(url, root):
    print('Retrieving files from Zenodo!')
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    raw_html = requests.get(url, headers={'User-Agent': user_agent})
    soup = BeautifulSoup(raw_html.content, "html.parser")
    links_zen = soup.find_all('a', href=re.compile(r'(zenodo)'))
    url_list = [_['href'] for _ in links_zen]

    if not os.path.exists(root):
        os.makedirs(root)

    for i, url in enumerate(tqdm.tqdm(url_list)):
        filename = os.path.join(root, '{:03}.pdf'.format(i))
        json_file = os.path.join(root, '{:03}.json'.format(i))
        if os.path.exists(filename) and os.path.exists(json_file):
            pass
        else:
            this_html = requests.get(url, headers={'User-Agent': user_agent})
            this_soup = BeautifulSoup(this_html.content, "html.parser")
            # extract metadata
            div = this_soup.find('div', {'id': 'recordCitation'})
            try:
                this_link = this_soup.find_all('a', href=re.compile(r'(.pdf)'))[0]['href']
            except:
                pdb.set_trace()
            meta = json.loads(div.attrs['data-record'])
            author_list = [_['person_or_org']['name'] for _ in meta['metadata']['creators']]
            authors = ', '.join(author_list)
            title = meta['metadata']['title']
            abstract = meta['ui']['description_stripped']
            try:
                place = meta['custom_fields']['imprint:imprint']['place']
            except:
                place = ''
                pdb.set_trace()
            affil_list = [_ for _ in meta['ui']['creators']['affiliations']]
            affil = ', '.join(affil_list)
            # some years have data in meta['ui']['contributors'] others in meta['metadata']['creators']
            pdb.set_trace()
            meta_dict = {'filename': filename,
                         'authors': authors,
                         'title': title.replace('.', ''),
                         'abstract': abstract,
                         'location': place,
                         'affiliation': affil}

            pdf = 'https://zenodo.org' + this_link
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(meta_dict, f, indent=4)
            with open(filename, 'wb') as f:
                f.write(requests.get(pdf, headers={'User-Agent': user_agent}).content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--year',
                        help='Input year to scrape (from 2000 to 2022)',
                        type=int,
                        required=True)
    args = parser.parse_args()

    url = 'https://ismir.net/conferences/ismir{}.html'.format(args.year)
    root = './data/pdfs/{}'.format(args.year)

    # # scraping ismir website
    # get_pdfs(url, root)

    # # scraping zenodo
    get_pdfs_zenodo(url, root)

    # # scraping zenodo
    # years = [str(_) for _ in range(2000, 2023)]
    # for year in years:
    #     print('Processing year {}'.format(year))
    #     url = 'https://ismir.net/conferences/ismir{}.html'.format(year)
    #     root = './data/pdfs/{}'.format(year)
    #     get_pdfs_zenodo(url, root)