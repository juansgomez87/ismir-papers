import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from openai import OpenAI
import fitz
import ast
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import functools
import time
from dotenv import load_dotenv
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
def openai_extract_affiliations(page_text):
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
                "content": '''You will be given a PDF as text. From the text, extract the author affiliations. Ignore any addresses or emails or similar information for \
                the affiliations. An affiliation should only be something such as a university, a company, or a similar entity. Each affiliation should also include the \
                country and the category of the entity. The countries should be in ISO 31166-1 A3 format. The categories are education (e.g., university), company, and \
                facility (e.g., research center, governmental agencies, non-profit). The affiliations come from all over the world so expect non-English text as well. \
                Compile this information into a single python dictionary that maps from string to string. You should only respond back in the form of: \
                {"<author as string>": "<affiliation as string>"} unless stated otherwise. "<affiliation as string>" should be separated by commas, for example: \
                Stanford University, USA, education. If you find an author but no affiliation, then set the value for the author as the empty string: "". If there \
                is no author/affiliation information in this context at all, then return: { }. Make sure the keys and values are always strings, really think carefully \'
                'if your response matches the technical specifications. Make sure you only respond with a single dictionary.''',
            },
            {"role": "user", "content": page_text},
        ],
    )

    return ast.literal_eval(response.dict()["choices"][0]["message"]["content"])


def get_affiliations(pdf_url):
    """Uses GPT-4 to read in a PDF file and automatically determine author affiliations
    Returns a dictionary that maps authors to their affiliations
    """
    # Download PDF
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "application/pdf,*/*",
    }
    response = requests.get(pdf_url, headers=headers)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    try:
        with fitz.open("temp.pdf") as doc:
            first_page = doc.load_page(0).get_text()
            last_page = doc.load_page(len(doc) - 1).get_text()

        os.remove("temp.pdf")
    except Exception as e:
        print(f"Error processing PDF: {pdf_url}")
        print(e)
        return {}

    # For some reason older papers use to have their author information on the last page.
    affiliations = openai_extract_affiliations(first_page)
    if not affiliations:
        affiliations = openai_extract_affiliations(last_page)

    return affiliations


def get_pdf_url(doi_url):
    """
    Uses Selenium to extract the PDF URL from a DOI URL
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get(doi_url)

    # Extract the PDF URL <link rel="alternate" type="application/pdf" href=<url>>
    try:
        pdf_url = driver.find_element(By.CSS_SELECTOR, 'link[rel="alternate"][type="application/pdf"]').get_attribute("href")
    except:
        time.sleep(2)
        driver.get(doi_url)
        pdf_url = driver.find_element(By.CSS_SELECTOR, 'link[rel="alternate"][type="application/pdf"]').get_attribute("href")

    driver.quit()

    return pdf_url


def generate_affiliations(path, save_dir):
    """
    Generates author affiliations for a dataframe of papers. Assumes that df
    has a column named "Authors with Affiliations" and a column named "Link"
    """
    df = pd.read_csv(f"{save_dir}/{path}")
    for index, doi_url in df["Link"].items():
        if (
            df["Authors with Affiliations"][index] == ""
            or pd.isna(df["Authors with Affiliations"][index])
            or df["Authors with Affiliations"][index] is None
        ):
            print(f"Getting affiliations for: {doi_url}")

            try:
                pdf_url = get_pdf_url(doi_url)
            except Exception as e:
                print(f"Failed to retrieve PDF link: {e}")
                pdf_url = "Missing"

            if pdf_url != 'Missing':
                affiliations_dict = get_affiliations(pdf_url)

                formatted_affiliations = ""
                if type(affiliations_dict) is tuple:
                    test = {}
                    [test.update(_) for _ in affiliations_dict]
                    affiliations_dict = test

                for author, affiliation in affiliations_dict.items():
                    formatted_affiliations += f"{author}, {affiliation};"

                df.loc[index, "Authors with Affiliations"] = formatted_affiliations[:-1]
                df.to_csv(f"{save_dir}/{path}", index=False)
            else:
                df.loc[index, "Authors with Affiliations"] = 'Not found'
                df.to_csv(f"{save_dir}/{path}", index=False)

        else:
            print(f"Affiliations already exist for: {doi_url} || Index: {index}")


def openai_get_abstract(page_text):
    """uses OpenAI gpt-4 to extract the abstract"""
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
                "content": '''You will be given a PDF as text. From the text, extract the abstract. An abstract is a short paragraph that summarizes the \
                 research. It should be a few sentences long. If you find an abstract, respond with the abstract as a string exactly as it was typed in \
                 the original context. If there is no abstract in this context at all, then return: "". Do not include the title of the paragraph (often \
                 the word "Abstract"). Make sure your response matches the technical specifications. DO NOT SUMMARIZE THE ABSTRACT!''',
            },
            {"role": "user", "content": page_text},
        ],
    )

    return response.dict()["choices"][0]["message"]["content"]


def get_abstract(pdf_url):
    # Download PDF
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "application/pdf,*/*",
    }
    response = requests.get(pdf_url, headers=headers)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    try:
        with fitz.open("temp.pdf") as doc:
            first_page = doc.load_page(0).get_text()

        os.remove("temp.pdf")
    except Exception as e:
        print(f"Error processing PDF: {pdf_url}")
        print(e)
        return ""

    abstract = openai_get_abstract(first_page)

    return abstract


def generate_abstracts(path, save_dir):
    """
    Generates abstracts for a dataframe of papers. Uses an LLM to do this so use this
    function at your own risk.
    """
    df = pd.read_csv(f"{save_dir}/{path}")
    for index, doi_url in df["Link"].items():
        if (
            df["Abstract"][index] == ""
            or df["Abstract"][index] is None
            or df["Abstract"][index] == "[TODO] Add abstract here."
        ):
            print(f"Getting abstract for: {doi_url}")
            pdf_url = get_pdf_url(doi_url)
            abstract = get_abstract(pdf_url)
            df.loc[index, "Abstract"] = abstract
            df.to_csv(f"{save_dir}/{path}", index=False)

        else:
            print(f"Abstract already exists for: {doi_url} || Index: {index}")


def parse_zenodo_record(record, pdf_url):
    """
    Returns the following metadata for a specific record as a dictionary

    dict = {
        'authors': List[String]
        'title': String
        'doi_url': String
        'affiliations': Dict[String, String]
        'keywords': List[String]
        'abstract': String
        'year': Int

    }

    JSON path for this data
        authors: json['metadata']['creators'][*]['name'] for each author
        title: json['metadata']['title']
        year: json['metadata']['publication_date'][0:4]
        doi_url: json['doi_url']
        affiliations: json['metadata']['creators'][*]['affiliation'] for each author
        keywords: you will have to use Selenium for this information if the information is available
    """

    authors = [author["name"] for author in record["metadata"]["creators"]]
    title = record["metadata"]["title"]
    doi_url = record["doi_url"]
    # affiliations = [author["affiliation"] for author in record["metadata"]["creators"]] # this is empty when using the Zenodo API unfortunately
    abstract = record["metadata"]["description"]
    year = int(record["metadata"]["publication_date"][0:4])

    affiliations = {}  # this can be a post processing step with a LLM

    return {
        "authors": authors,
        "title": title,
        "doi_url": doi_url,
        "affiliations": affiliations,
        "abstract": abstract,
        "year": year,
    }


def get_zenodo_record(record_id):
    """
    Returns a Zenodo JSON response for a given record ID
    """
    r = requests.get(f"https://zenodo.org/api/records/{record_id}")
    return r.json()


def extract_table_data(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get(url)

    table_elements = driver.find_elements(By.XPATH, "//table/tbody/tr")
    paper_links = {}
    for element in table_elements:
        links = element.find_elements(By.TAG_NAME, "a")
        doi_link = links[0].get_attribute("href")
        pdf_link = links[1].get_attribute("href")

        if "doi.org" in doi_link:
            zenodo_id = int(re.search(r"zenodo\.(\d+)", doi_link).group(1))
        else:
            zenodo_id = "Missing"

        if zenodo_id == "Missing":
            print("No doi.org link found. Skipping this paper: ", doi_link)
            continue

        if not pdf_link:
            print("PDF link not found. Attempting to retrieve from doi link: ")
            try:
                pdf_link = get_pdf_url(doi_link)
            except Exception as e:
                print(f"Failed to retrieve PDF link: {e}")
                pdf_link = "Missing"

        paper_links[zenodo_id] = pdf_link

    driver.quit()

    return paper_links


def format_data(metadata):
    authors = ""
    for author in metadata["authors"]:
        authors += author + ";"
    authors = authors[:-1]

    title = metadata["title"]
    year = metadata["year"]
    link = metadata["doi_url"]
    authors_affiliations = ""
    abstract = metadata["abstract"]

    return [authors, title, year, link, authors_affiliations, abstract]


def scrape_website(url):
    """
    Scrapes a website for paper information and saves it to a csv
    """
    print("Extracting metadata from: ", url)
    zendodo_ids = extract_table_data(url)

    all_data = []
    for record_id, pdf_url in tqdm(
        zendodo_ids.items(), desc="Processing Zenodo Records"
    ):
        # if the record id is not a number, skip it
        if isinstance(record_id, str):
            print(f"No information found. Skipping | {record_id}")
            continue

        record = get_zenodo_record(record_id)

        print(f"Parsing record: {record_id}")
        try:
            metadata = parse_zenodo_record(record, pdf_url)
        except:
            time.sleep(2)
            record = get_zenodo_record(record_id)
            metadata = parse_zenodo_record(record, pdf_url)

        row = format_data(metadata)
        all_data.append(row)

    df = pd.DataFrame(
        all_data,
        columns=[
            "Authors",
            "Title",
            "Year",
            "Link",
            "Authors with Affiliations",
            "Abstract",
        ],
    )

    return df


def scrape_all_websites(urls):
    for url in urls:
        year = url.split("/")[-1].split(".")[0][-4:]
        print(f"Scraping ISMIR website papers for year: {year}")
        if not os.path.exists(f"data/ismir/ismir_{year}.csv"):
            data = scrape_website(url)
            # check if data directory exists
            if not os.path.exists("data"):
                os.makedirs("data")
            data.to_csv(f"data/ismir/ismir_{year}.csv")
        else:
            print(f"Data for year {year} was already scraped!")



def postprocess_all_data(paths, save_dir):
    for data_path in tqdm(paths, desc="Postprocessing Data"):
        # Author affiliation generation
        print(f"Generating affiliations for: {data_path}")
        generate_affiliations(data_path, save_dir)

        print(f"Extracting abstracts for: {data_path}")
        generate_abstracts(data_path, save_dir)

        print(f"Finished processing: {data_path}")


if __name__ == "__main__":

    # ISMIR websites to scrape
    years = np.arange(2000, 2024)

    urls = ['https://ismir.net/conferences/ismir{}.html'.format(year) for year in years]

    # Scrape information
    # scrape_all_websites(urls)

    # # Use an LLM to automate tasks in post processing step
    # # Tasks can include abstract extraction, keyword extraction, and author affiliation extraction
    save_dir = "data/ismir"
    data_paths = os.listdir(save_dir)
    postprocess_all_data(data_paths, save_dir)
