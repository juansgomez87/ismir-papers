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
OPENAI_API_KEY=xxx
```
To scrape conferences from ISMIR yearly websites:
```
python conf_scraper.py
```
To scrape data from ISMIR/Zenodo (edit source code to add websites to scrape from):
```
python zenodo_scraper.py
```
To produce topics based on sentence embeddings:
```
python topic_papers.py
```
