# ISMIR conference analysis

This repository contains code to analyze data from ISMIR conferences.

### Installation 
Install the required dependencies:
```
python3 -m venv .venv
pip install -r requirements.txt
touch .env
```
Complete the .env file with the following keys:
```
openai_api_key=xxx
```
To scrape conferences from ISMIR yearly websites:
```
python conf_scraper.py
```
To parse text from pdfs (not really working):
```
python parse_text.py
```
To scrape data from ISMIR/Zenodo (@Erick, please update this):
```
python scrape.py
```
To produce topics based on sentence embeddings:
```
python topic_papers.py
```