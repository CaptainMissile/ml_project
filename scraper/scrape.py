import re
import requests
from bs4 import BeautifulSoup
# from extractive_summary.textrank_text_summarization import extractive_summary

class Scraper:
    def __get_links(self, query):
        irrelevant_domains = ['facebook.com', 'bn.wikipedia', 'google.com', 'youtube.com', 'instagram', 'hackerearth']

        resp = requests.get('https://www.google.com/search?q='+ query)
        soup = BeautifulSoup(resp.text, 'html.parser')

        links = []
        pattern = r"/url\?q=(.*)&sa"

        for a_tag in soup.select('a'):
            match = re.match(pattern, a_tag.get('href'))

            if match:
                link = match.groups()[0]

                if not any(domain in link for domain in irrelevant_domains):
                    links.append(link)

        return links


    def __rmv_special_chars(self, text):
        return re.sub('[^a-zA-Z0-9 \n\.]', '', text)



    def __rmv_one_or_two_word_lines(self, text):
        return ' '.join ([re.sub('^(\w+)$|^(\w+ \w+)$', '', line) for line in text.split('\n')])


    def __scrape_single_page(self, link):
        resp = requests.get(link)
        soup = BeautifulSoup(resp.text, 'html.parser')
        p_tags = soup.select('p')

        res = []

        for p_tag in p_tags:
            res.append(self.__rmv_one_or_two_word_lines(self.__rmv_special_chars(p_tag.text)))

        return ' '.join(res)

    # def __single_page_summary(self, text):
    #     return extractive_summary(text)



    def scrape_pages(self, query):
        links = self.__get_links(query)[1:3]
        res = []

        for link in links:

            res.append({
                'link': link,
                'text': self.__scrape_single_page(link)
            })

        return res


if __name__ == '__main__':
    import console_colors as console_colors
    search_key = input(f'{console_colors.WARNING}Search >>> ')

    scraper = Scraper()
    pages_lst = scraper.scrape_pages(query= search_key)


    
    for page in pages_lst:
        print(page['link'], '\n', page['text'])
        print('==================================')
        print()