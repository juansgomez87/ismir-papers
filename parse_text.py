import pypdf
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import tqdm
import spacy
import spacy_fastlang
import sys
from typing import List

def remove_hyphens(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-"):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)


def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) :]
    return lines

def preprocess_text(text, filename):
    # remove hyphens
    text = remove_hyphens(text)
    # Remove page numbering
    text = text.replace(r'Page \d+ of \d+', '')
    # Remove line breaks
    text = text.replace('\n', '')
    # Remove tab breaks
    text = text.replace('\t', '')
    # Get possible title
    title = re.findall(r'(\b(?:[A-Z]+[a-z]?[A-Z]*|[A-Z]*[a-z]?[A-Z]+)\b(?:\s+(?:[A-Z]+[a-z]?[A-Z]*|[A-Z]*[a-z]?[A-Z]+)\b)*)',
               text)
    if len(title) == 0:
        # print('{} does not contain all caps text!'.format(filename))
        title = ['']

    # get abstract
    idx_abs = text.lower().find('abstract')
    idx_intro = text.lower().find('introduction')

    if idx_abs > 0 and idx_intro > 0:
        abstract = text[idx_abs: idx_intro]
        abstract = abstract.replace('ABSTRACT', '').replace('Abstract', '')
    else:
        abstract = ''

    # trim text from intro
    if idx_intro > 0:
        # trim initial data for nlp
        header = text[:idx_abs]
        text = text[idx_intro:]
    else:
        header = ''

    # remove references
    idx_ref = text.find('. REFERENCES')
    if idx_ref < 0:
        idx_ref = text.find('. BIBLIOGRAPHY')
    if idx_ref < 0:
        idx_ref = text.find('. References')
    if idx_ref < 0:
        idx_ref = text.find('. Bibliography')
    if idx_ref > 0:
        text = text[:idx_ref]
        idx_ref = text.find('.References')
    if idx_ref < 0:
        idx_ref = text.find('.Bibliography')
    if idx_ref > 0:
        text = text[:idx_ref]

    return text, title[0], header, abstract


def extract_text_from_pdf(pdf_file, nlp):
    text = ''
    empty_text = False
    with open(pdf_file, 'rb') as file:
        # check for corrupt pfds
        try:
            pdf_reader = pypdf.PdfReader(file)
        except:
            print('{} is corrupt!'.format(pdf_file))
            sys.exit()

        # try to extract data from metadata
        meta = pdf_reader.metadata
        # pdb.set_trace()
        if meta is None:
            meta = {}
        if '/Author' in meta.keys():
            author = pdf_reader.metadata.author

        else:
            author = None

        for page_num in range(len(pdf_reader.pages)):
            try:
                text += pdf_reader.pages[page_num].extract_text()
            except:
                text = ''
                print('{} does not retrieve text!'.format(pdf_file))
        text = text.encode('utf-8', 'ignore').decode('utf-8')

    if len(text) < 300:
        # TODO: use OCR to try and read files?
        empty_text = True

    # ensure text is in english
    try:
        doc = nlp(text)
    except:
        pdb.set_trace()

    if doc._.language != 'en' or doc._.language_score <= 0.7:
        # TODO: try to decode these examples?
        text = ''
        empty_text = True
        print('{} is encoded and not readable!'.format(pdf_file))

    text, title, header, abstract = preprocess_text(text, pdf_file)
    return text, author, title, header, abstract, empty_text
def get_pdfs_data(all_pdfs):
    df = pd.DataFrame({'filename': all_pdfs})
    df['year'] = [int(_.split('/')[-2]) for _ in all_pdfs]
    df.sort_values(by='year', inplace=True)
    df['author'] = ''
    df['title'] = ''
    df['institution'] = ''
    df['country'] = ''
    df['gender'] = ''
    df['abstract'] = ''
    df['diversity_flag'] = ''
    df['cross_cult_flag'] = ''
    df['empty_flag'] = ''
    return df

def nlp_processing(nlp, text):
    doc = nlp(text)

    print(list(doc.noun_chunks))
    pdb.set_trace()


def get_flags(text):
    flag_div = False
    flag_cross = False
    # todo: try lemmatizing?

    # text.lower().find('inclusion') > 0 or \
    if text.lower().find('diverseness') > 0 or \
       text.lower().find('diversity') > 0:
        flag_div = True
    if text.lower().find('cross cultural') > 0 or \
       text.lower().find('cross-cultural') > 0 or \
       text.lower().find('crosscultural') > 0:
        flag_cross = True

    return flag_div, flag_cross

def plot_data(df):
    # df['year'].value_counts().sort_index().plot(kind='bar')
    # plt.show()
    #
    # df.groupby('year').sum()['empty_flag'].plot(kind='bar')
    # plt.show()

    # plot general data about papers
    plot_df = pd.DataFrame(
        {'Count': df.groupby('year').count().filename,
         'Cross-cultural': df.groupby('year').sum()['cross_cult_flag'],
         'Diversity': df.groupby('year').sum()['diversity_flag']},
        index=df.year.unique())
    plot_df.plot.bar()
    plt.show()

    # plot data about empty texts
    df.groupby('year').sum()['empty_flag'].plot(kind='bar')
    plt.show()
    pdb.set_trace()

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('language_detector')
    pdf_root = './data/pdfs/'
    txt_root = './data/txts/'


    all_pdfs = glob.glob(os.path.join(pdf_root, '*/*.pdf'))
    print('Total of {} pdf articles!'.format(len(all_pdfs)))

    df = get_pdfs_data(all_pdfs)

    for pdf in tqdm.tqdm(df.filename.tolist()):
        text, author, title, header, abstract, empty_text = extract_text_from_pdf(pdf, nlp)

        # # try NLP on individual papers
        # nlp_processing(nlp, text)

        # find diversity flags on text
        flag_div, flag_cross = get_flags(text)

        # # save data
        # df.loc[df.filename == pdf, 'author'] = author
        # df.loc[df.filename == pdf, 'title'] = title
        # df.loc[df.filename == pdf, 'abstract'] = abstract
        df.loc[df.filename == pdf, 'diversity_flag'] = flag_div
        df.loc[df.filename == pdf, 'cross_cult_flag'] = flag_cross
        df.loc[df.filename == pdf, 'empty_flag'] = empty_text

        # save text files
        dir = os.path.join(txt_root, pdf.split('/')[-2])
        if not os.path.exists(dir):
            os.makedirs(dir)

        if text != '':
            filename = pdf.split('/')[-1].replace('.pdf', '.txt')
            f = open(os.path.join(dir, filename), 'w')
            f.write(text)
            f.close()

    plot_data(df)
    pdb.set_trace()


    df.to_csv('./data/papers.csv', sep='\t')

