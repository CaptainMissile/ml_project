from django.shortcuts import render
from scraper.scrape import Scraper


# Create your views here.
def index(request):
    return render(request, 'index.html')


def get_summary(request):
    query = request.GET['query']
    scraper = Scraper()
    data = scraper.scrape_pages(query)

    return render(request, 'index.html', context = {'data' : data})