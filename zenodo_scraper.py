import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
import fitz
import pdfplumber
import ast
import os
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import functools
import time
from dotenv import load_dotenv
import utils

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI KEY: ", API_KEY)
MODEL = "gpt-4o-mini"
LAYOUT = False
X_TOLERANCE = 1


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
                    print(f"{func.__name__} failed, retrying in {delay} seconds...")
                    print(e)
                    time.sleep(delay)

            print(f"Failed after {max_retries} retries.")
            return None

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
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You will be given a PDF as text which potentially contains author information from a research paper. The text may have formatting issues that you need to handle:

1. Joined words (e.g., "JohnSmith" → "John Smith", "StanfordUniversity" → "Stanford University")
2. Special symbols for affiliations (e.g., ♭, ♮, †, *, [1], 1)

From the cleaned text, extract:
1. Authors' names
2. Their affiliation name (institution only, not departments)
3. Affiliation country
4. Affiliation category

Ignore any addresses, emails, or departments. An affiliation should only be something such as a university, company, or similar entity.

The affiliation categories are strictly:
- education (e.g., university)
- company (e.g., commercial entities)
- facility (e.g., research center, governmental agencies, non-profit)

Compile this information into a single python dictionary that maps from string to string. You should only respond back in the form of:
{"<author name as string, in the form of FIRST_NAME MIDDLE_NAME(S) LAST_NAME>": "<affiliation info as string, in the form of AFFILIATION_NAME>AFFILIATION_COUNTRY (ISO format)>AFFILIATION_CATEGORY>"}

Important rules:
1. Multiple affiliations should be separated by | character
2. If you find an author but no affiliation, use the value "Unknown>Unknown>Unknown"
3. If no author/affiliation information is found, return: { }
4. Countries must be in ISO 31166-1 A3 format, which are always 3 letters (Sweden should be SWE, not SE)
5. If country cannot be determined, use "Unknown" for the country ISO code
6. If the category cannot be determined, use "Unknown" for the category

Name are usually underneath or near the paper's title. Sometimes, you will be given the last page of the paper, which may contain the author's information in its own subsection. 
The author's name is usually followed by their affiliation, or have citation numbers or symbols next to their name. 
The affiliation is usually followed by the country. The country and category are not always present.

DO NOT include or use information from things like the title of the paper, abstract, citation lists, etc. Only use sections of 
the text that contain author information of the paper itself. 

Sometimes all of the author names and other info are on their own line in terms of formatting. Make sure you can handle this and 
think carefully to how a human might process this visually.

Sometimes only one affiliation is given for multiple authors. In this case, you should assign the same affiliation to all authors.

If initially you think a country is unknown, ask yourself again if you know the organization name, and if yes, what country it is in or founded in.
For example, MetaBrainz is a USA organization; Meertens Instituut is a Dutch organization.


Examples:
Input:
\nAuthor Information\nDaniel Bendor Mark Sandler\nUndergraduate School of Electrical Engineering Department of Electronic Engineering\nUniversity of Maryland at College Park King(cid:213)s College London\ndbendor@glue.umd.edu mark.sandler@kcl.ac.uk

Output:
{
   "Daniel Bendor": "University of Maryland at College Park>USA>education",
   "Mark Sandler": "King's College London>GBR>education"
}

Input:
JohnSmith[1,2], M.Wang[1]
[1]StanfordUniversity
[2]GoogleResearch

Output:
{
   "John Smith": "Stanford University>USA>education|Google Research>USA>company",
   "M. Wang": "Stanford University>USA>education"
}

Input:
YingWang♮, JamesB.Smith♭
♮BeijingLab,MicrosoftAsia
♭UnivofTokyo

Output:
{
   "Ying Wang": "Microsoft Asia>CHN>company",
   "James B. Smith": "University of Tokyo>JPN>education"
}

Input:
John Doe, Jane Smith
Stanford University, Google Research

Output:
{
    "John Doe": "Stanford University>USA>education",
    "Jane Smith": "Google Research>USA>company"
}

Input:
Kosetsu Tsukuda Masahiro Hamasaki Masataka Goto
National Institute of Advanced Industrial Science and Technology (AIST), Japan
{k.tsukuda, masahiro.hamasaki, m.goto}@aist.go.jp

Output:
{
    "Kosetsu Tsukuda": "National Institute of Advanced Industrial Science and Technology (AIST)>JPN>facility",
    "Masahiro Hamasaki": "National Institute of Advanced Industrial Science and Technology (AIST)>JPN>facility",
    "Masataka Goto": "National Institute of Advanced Industrial Science and Technology (AIST)>JPN>facility"
}

Input:
Andres Ferraro Jaehun Kim Sergio Oramas
Andreas Ehmann Fabien Gouyon
Pandora-SiriusXM, Oakland

Output:
{
    "Andres Ferraro": "Pandora-SiriusXM>USA>company",
    "Jaehun Kim": "Pandora-SiriusXM>USA>company",
    "Sergio Oramas": "Pandora-SiriusXM>USA>company",
    "Andreas Ehmann": "Pandora-SiriusXM>USA>company",
    "Fabien Gouyon": "Pandora-SiriusXM>USA>company"
}

Make sure the keys and values are always strings, and verify your response matches these technical specifications exactly. Respond only with the dictionary.""",
            },
            {"role": "user", "content": page_text},
        ],
    )

    return ast.literal_eval(response.choices[0].message.content)


def get_affiliations(pdf_url):
    """Uses GPT to read in a PDF file and automatically determine author affiliations
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
            first_page = doc[0].get_text("text")
            if len(doc) > 1:
                second_page = doc[1].get_text("text")
            else:
                second_page = ""
            last_page = doc[-1].get_text("text")

        os.remove("temp.pdf")
    except Exception as e:
        print(f"Error processing PDF: {pdf_url}")
        print(e)
        return {}

    # For some reason older papers use to have their author information on the last page.
    affiliations = openai_extract_affiliations(first_page)
    if not affiliations:
        affiliations = openai_extract_affiliations(last_page)

    if not affiliations and second_page:
        affiliations = openai_extract_affiliations(second_page)

    return affiliations


@retry_on_ratelimit(max_retries=20, delay=3)
def get_pdf_url(doi_url):
    """
    Uses Selenium to extract the PDF URL from a DOI URL
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get(doi_url)

    # Extract the PDF URL <link rel="alternate" type="application/pdf" href=<url>>
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'link[rel="alternate"][type="application/pdf"]')
            )
        )
        pdf_url = driver.find_element(
            By.CSS_SELECTOR, 'link[rel="alternate"][type="application/pdf"]'
        ).get_attribute("href")
    except:
        print(
            f"PDF link not found for: {doi_url}, likely due to broken link or page contents not loaded"
        )
        pdf_url = "Missing"

    driver.quit()

    return pdf_url


def generate_affiliations(path, save_dir):
    """
    Generates author affiliations for a dataframe of papers. Assumes that df
    has a column named "Authors with Affiliations" and a column named "Link"

    Here's the definitions of delimiters used in the CSV files
    +: for a single author, separates their name and their affiliation string
    ;: separates pairs of <author>+<affiliation> for a single paper
    >: separates the affiliation name, country, and category for a single author
    """
    df = pd.read_csv(f"{save_dir}/{path}", dtype={"Authors with Affiliations": str})
    for index, doi_url in df["Link"].items():
        if (
            df["Authors with Affiliations"][index] == ""
            or pd.isna(df["Authors with Affiliations"][index])
            or df["Authors with Affiliations"][index] is None
            or df["Authors with Affiliations"][index] == "Not found"
        ):
            print(f"Getting affiliations for: {doi_url}")

            try:
                pdf_url = get_pdf_url(doi_url)
            except Exception as e:
                print(f"Failed to retrieve PDF link: {e}")
                pdf_url = "Missing"

            if pdf_url != "Missing":
                affiliations_dict = get_affiliations(pdf_url)

                formatted_affiliations = ""
                for author, affiliation in affiliations_dict.items():
                    formatted_affiliations += f"{author}+{affiliation};"

                if not affiliations_dict:
                    df.loc[index, "Authors with Affiliations"] = "Not found"
                    df.to_csv(f"{save_dir}/{path}", index=False)

                else:
                    df.loc[index, "Authors with Affiliations"] = formatted_affiliations[
                        :-1
                    ]
                    df.to_csv(f"{save_dir}/{path}", index=False)
            else:
                df.loc[index, "Authors with Affiliations"] = "Not found"
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
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You will be given a PDF research paper page as text. The text may have formatting issues that need to be handled:

1. Joined words (e.g., "researchpaper" → "research paper")
2. Special characters and formatting symbols (e.g., \n, -, _, etc.)
3. Inconsistent spacing

Your task is to:
1. Find the abstract section
2. Extract its content exactly as written
3. Preserve the original meaning and content

Rules:
1. Do NOT include the word "ABSTRACT" or any section headers
2. Do NOT summarize or modify the content
3. Do NOT add interpretations or explanations
4. Return empty string "" if:
   - No abstract is found
   - Text is unreadable/corrupted
   - Cannot determine abstract boundaries


Example 1:
Input:
"1. Introduction
This paper..."

Output:
""

Respond ONLY with the cleaned abstract text or empty string. No additional text or explanations.""",
            },
            {"role": "user", "content": page_text},
        ],
    )

    return response.choices[0].message.content


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
            first_page = doc[0].get_text("text")

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


@retry_on_ratelimit(max_retries=5, delay=2)
def get_zenodo_record(record_id):
    """
    Returns a Zenodo JSON response for a given record ID
    """
    r = requests.get(f"https://zenodo.org/api/records/{record_id}")
    return r.json()


def extract_table_data_zenodo(url):
    """
    This function only works for years 2000-2023, assuming the paper is regisered on Zenodo

    If in Zenodo, assumes, the url is a doi.org link

    Skips papers not on Zenodo.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
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


def scrape_website_zenodo(url):
    """
    Scrapes a website for paper information and saves it to a csv
    """
    print("Extracting metadata from: ", url)
    zendodo_ids = extract_table_data_zenodo(url)

    all_data = []
    for record_id, pdf_url in tqdm(
        zendodo_ids.items(), desc="Processing Zenodo Records"
    ):
        # if the record id is not a number, skip it
        if record_id == "Missing":
            print(f"No information found. Skipping | {record_id}")
            continue

        record = get_zenodo_record(record_id)
        print(f"Parsing record: {record_id}")
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


@retry_on_ratelimit(max_retries=3, delay=60)
def generate_affiliation_country_and_category(affiliation):
    """
    Uses LLM to gerenate the country and category of an affiliation

    affiliation: String
    """
    client = OpenAI(
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You will be given an affiliation name as a string. From the name, determine the country and category of the affiliation. The categories are education (e.g., university), company, and facility (e.g., research center, governmental agencies, non-profit). 
                The countries should be in ISO 31166-1 A3 format. Respond with a string in the format: AFFILIATION_NAME>COUNTRY (ISO format)>CATEGORY. If you can't determine the country, use "Unknown".
                
                Make sure your response matches the technical specifications. Make sure you only respond with a single string. Do NOT have quotes around the string.

                More examples:
                
                Input:
                Stanford University
                
                Output:
                "Stanford University>USA>education"

                Input:
                Google Inc

                Output:
                "Google Inc>USA>company"

                Input:
                LIT AI Lab, Linz Institute of Technology

                Output:
                "LIT AI Lab, Linz Institute of Technology>AUT>facility"

                Input:
                Unknown University

                Output:
                "Unknown University>Unknown>education"

                Input:
                University of Tokyo

                Output:
                "University of Tokyo>JPN>education"

                Input:
                Unknown

                Output:
                "Unknown>Unknown>Unknown"
                """,
            },
            {"role": "user", "content": affiliation},
        ],
    )

    return response.choices[0].message.content


def scrape_website_without_zenodo(url):
    """
    This function is used for websites that do not have Zenodo links in their HTML elements, and instead
    uses an LLM to extract various pieces of information from a found PDF. Skips papers without a PDF link.

    This function currently is used to scrape ISMIR 2024 papers, but this can change in the future

    Assumes that the PDF is within a div class with label pp-card-header

    Returns a pandas dataframe with the following columns
        Authors: String
        Title: String
        Year: Int
        Link: String (it will just be the url root + the href)
        Authors with Affiliations: String
        Abstract: String
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get(url)

    year = int(url[13:17])  # assumes the year is near the beginning
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "pp-card-header"))
    )
    paper_cards = driver.find_elements(By.CLASS_NAME, "pp-card-header")
    all_data = [
        ["Authors", "Title", "Year", "Link", "Authors with Affiliations", "Abstract"]
    ]

    links = []
    for card in paper_cards:

        try:
            html_link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
            title = card.find_element(By.CLASS_NAME, "card-title").text
            links.append(html_link)
        except:
            print(
                f"HTML link not found. Skipping this paper: {card.find_element(By.CLASS_NAME, 'card-title').text}"
            )
            continue

    for html_link in tqdm(links, desc="Processing Papers"):
        # Open the html link URL and find title, authors, authors with affiliations, and abstract
        driver.get(html_link)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".card-title.main-title.text-left")
            )
        )
        title = driver.find_element(
            By.CSS_SELECTOR, ".card-title.main-title.text-left"
        ).text
        abstract = (
            driver.find_element(By.ID, "abstractExample")
            .find_element(By.TAG_NAME, "p")
            .text
        )

        print(f"Scraping paper: {title}, at URL: {html_link}")

        # Get authors and affiliations using LLM
        author_html_infos = driver.find_element(
            By.CSS_SELECTOR, ".card-subtitle.mb-2.text-muted.text-left"
        ).find_elements(By.TAG_NAME, "a")

        authors = ""
        author_affs = ""
        for author_element in author_html_infos:
            split = author_element.text.split("(")

            if len(split) == 1:
                author = split[0]
                affiliation = "Unknown"
                generated_affiliation = "Unknown>Unknown>Unknown"
            else:
                author = split[0]
                affiliation = split[1].split(")")[0]
                generated_affiliation = generate_affiliation_country_and_category(
                    affiliation
                )
            authors += author.strip() + ";"  # author;
            author_affs += (
                author + "+" + generated_affiliation + ";"
            )  # author+affiliation; where affiliation is affiliation>country>category

        authors = authors[:-1]
        author_affs = author_affs[:-1]

        all_data.append([authors, title, year, html_link, author_affs, abstract])

    driver.quit()
    df = pd.DataFrame(all_data)
    return df


def scrape_all_websites(urls):
    """ """
    for url, year in urls:
        print(f"Scraping ISMIR website papers for year: {year}")
        if not os.path.exists(f"data/ismir/ismir_{year}.csv"):
            data = scrape_website_zenodo(url)
            if not os.path.exists("data"):
                os.makedirs("data")
            if not os.path.exists("data/ismir"):
                os.makedirs("data/ismir")

            data.to_csv(f"data/ismir/ismir_{year}.csv")
        else:
            print(f"Data for year {year} was already scraped!")


def postprocess_all_data(paths, save_dir):
    for data_path in tqdm(paths, desc="Postprocessing Data"):

        # only run on csv that dont have these already
        print(f"Generating affiliations for: {data_path}")
        generate_affiliations(data_path, save_dir)

        print(f"Extracting abstracts for: {data_path}")
        generate_abstracts(data_path, save_dir)

        # Drops missing info
        df = pd.read_csv(f"{save_dir}/{data_path}")
        to_drop = df[
            (df["Abstract"] == "")
            | (df["Authors with Affiliations"] == "Not found")
            | (df["Abstract"].isna())
        ].index
        df.drop(to_drop, inplace=True)
        df.to_csv(f"{save_dir}/{data_path}", index=False)

        print(f"Finished processing: {data_path}")


if __name__ == "__main__":

    # ISMIR websites to scrape
    years = list(range(2000, 2024))
    urls = [(f"https://ismir.net/conferences/ismir{year}.html", year) for year in years]

    # Scrape information
    # scrape_all_websites(urls)

    # Use an LLM to automate tasks in post processing step
    # Tasks can include abstract extraction, keyword extraction, and author affiliation extraction
    # print("Postprocessing all data")
    # save_dir = "data/ismir"
    # data_paths = sorted(os.listdir(save_dir))
    # postprocess_all_data(data_paths, save_dir)

    # Concatenate all data, get first author info, and embedding info
    # utils.create_concatenated_data(save_dir)
    # utils.create_first_author_columns()
    utils.create_embeddings()

    print("All data has been scraped and processed!")
