import os
import json
import requests
import argparse
from bs4 import BeautifulSoup
from time import sleep
from random import randrange

def fetch_json(url, params):
    endpoint = "%s/api/v1/archive" % url
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()

def fetch_html(url):
    response = requests.get(url)
    return response.text

def fetch_and_parse(url):
    limit = 12
    offset = 0
    results_len = 1
    while results_len != 0:
        params = {'limit': limit, 'offset': offset}
        entries = fetch_json(url, params=params)
        for item in entries:
            Link = item['canonical_url']
            Title = item['title']
            Date = item['post_date']
            Html = fetch_html(Link)
            soup = BeautifulSoup(Html, 'html.parser')
            content = soup.find('div', {'class': 'markup'})
            if content:
                yield {
                    'title': Title,
                    'link': Link,
                    'date': Date,
                    'content': content.text,
                }
            timeout = randrange(2, 20)
            sleep(timeout)
        offset = limit + offset
        results_len = len(entries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch posts from a Substack blog.')
    parser.add_argument('blog', help='The name of the Substack blog to fetch posts from.')
    args = parser.parse_args()

    url = f'https://{args.blog}.substack.com'
    filename = 'data/' + url.replace('https://', '').replace('.', '-') + '.jsonl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for post in fetch_and_parse(url):
            f.write(json.dumps(post) + '\n')