import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_site(base_url):
    visited_links = set()
    try:
        main_response = requests.get(base_url)
        main_response.raise_for_status()
        main_soup = BeautifulSoup(main_response.text, 'html.parser')
        nav_container = main_soup.find('div', id='leftmenuinnerinner')
        links = nav_container.find_all('a', href=True)
        base_url_corrected = 'https://www.w3schools.com/sql/'

        tutorial_links = set()
        for link in links:
            href = link['href']
            if not href.startswith('http'):
                full_url = base_url_corrected + href if not href.startswith('/') else 'https://www.w3schools.com' + href
            else:
                full_url = href

            if full_url not in visited_links:
                tutorial_links.add(full_url)

        for url in tutorial_links:
            if url not in visited_links:
                visited_links.add(url)
                scrape_individual_page(url)
                time.sleep(1)

    except requests.RequestException as e:
        print(f"Error accessing {base_url}: {str(e)}")

def scrape_individual_page(url):
    data = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_title = soup.find('title').get_text(strip=True).replace(' - SQL Tutorial - W3Schools', '')

        examples = soup.find_all('div', class_='w3-example')

        for example in examples:
            sql_query = example.find('div', class_='w3-code')
            if sql_query:
                cleaned_sql = clean_sql(sql_query.text)
                prompt = f"Generate SQL query for: {page_title}"
                if cleaned_sql:
                    data.append({"prompt": prompt, "sql": cleaned_sql})

    except requests.RequestException as e:
        print(f"Error during requests to {url}: {str(e)}")
    finally:
        if data:
            append_data_to_json(data)

def clean_sql(sql_text):
    """
    Cleans SQL text by removing HTML tags, newlines, and non-breaking spaces.
    """
    if not sql_text.strip():
        return None
    try:
        soup = BeautifulSoup(sql_text, 'html.parser')
        text = soup.get_text(strip=True)
        # Remove non-breaking spaces and extra newlines
        cleaned_text = text.replace('\u00a0', ' ').replace('\n', ' ')
        return cleaned_text
    except Exception as e:
        print(f"Error parsing SQL text: {str(e)}")
        return None


def append_data_to_json(new_data):
    json_file_path = 'trainingData.json'
    try:
        data = []
        try:
            with open(json_file_path, 'r+') as file:
                data = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass

        data.extend(new_data)
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

    except Exception as e:
        print(f"Error appending data to JSON file: {str(e)}")

# Example usage
base_url = 'https://www.w3schools.com/sql/default.asp'
scrape_site(base_url)
