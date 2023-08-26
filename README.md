# Glassdoor Review Analysis

The goal of this is to take job reviews from Glassdoor for Honeywell and perform a summarization on it using LLMs found on Huggingface. 

## 1. Scraping

### Dependencies

* Selenium

### Executing program

The first part involves scraping reviews from the glassdoor website. This can be done using the "Webscrape.ipynb". The scraping is done by Selenium.
In order to start scraping, you need an account with access to the review page that you're trying to scrape. 
In this case, the page that we are scraping is the Honeywell website. 

Since the website has mechanisms against repeated scraping, the scraping may fail after a few pages.

The code will save text files with the corresponding page number. 

Change the "start_page" variable to change the page on which to start scraping again. 

Change the website to scrap reviews from another website:
```
def navigate_to_reviews():
    driver.get("NEW WEBSITE HERE")
```


## 2. Summarization

### Dependencies

* Pandas
* Gensim 
* Transformers
* Json

###  Executing program

Use: "Glassdoor Summary Tree.ipynb"

This part of the program can be run without the previous part if you want to use the Honeywell data available on this github.

This part involves taking all of the reviews scraped in the previous section and running a summarization LLM from Huggingface on this.

All of the reviews will be separated by year, then rating, then topic. Then the reviews are combined and summarized. 

This is then all put into a json tree where it can later be visualized 

## 3. Visualization

### Dependencies

* D3.js

###  Executing program

Open the "Website" folder and run the "index.html" file to see the visualization tree.

The parent nodes are the summarization and the leaf nodes are original reviews.

These nodes can be hovered over to see the full text and clicked on to open and close that branch. 


