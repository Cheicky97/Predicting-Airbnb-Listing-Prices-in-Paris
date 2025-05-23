import requests
from bs4 import BeautifulSoup
import pandas as pd

class ScrapFromInsideAirbnb:
    """
    Scrap and load airbnb listings data from url
    """
    def __init__(self, url:str='https://insideairbnb.com/fr/get-the-data/'):
        """
        url: str, optional
            url to the insideairbnb data
            The default value is 'https://insideairbnb.com/fr/get-the-data/'.
        """
        self.url = url
        self.status = {200: 'Successfully requested!',
                  404: 'Not found.',
                  403: 'Forbidden access!',
                  401: 'Unauthorized access!',
                  300: 'Multiple choices.'}
    
    def request(self, **kwargs):
        """
        Make the request and print the status message.
        ------
        kwargs: dict of additional option for getting request
        """
        self.req = requests.get(self.url, **kwargs)
        print(self.status[self.req.status_code])

    def get_data_url_of_city(self, city:str='Paris', parser:str='html.parser'):
        """
        get the url to the listings.csv.gz of the city
        -------
        city: str, optional
            The city for which we want to get listings.csv data
            The default is the city of Paris.
        parser:str, optional
            library that will be used to parse html data.
            The default is 'html.parser'.
            An alternative is 'html5lib'.
        """
        city_not_found=True

        self.html = self.req.text
        self.soup = BeautifulSoup(self.html, parser)
        list_of_tbody_tags = self.soup.find_all('tbody')
        for tbody_tag in list_of_tbody_tags:
            if city.title() in tbody_tag.tr.td.contents[0]:
                city_not_found = False
                self.city_listings_data_url =  tbody_tag.tr.a['href']
                break
        if city_not_found:
            print(f"{city} not found amongst existing cities.")
        else:
            print(f'url to listings data of {city}: {self.city_listings_data_url}')

    def load_to_df(self, compression:str='gzip', **kwargs):
        """
        Load listings.csv data into a pd.DataFrame
        -------
        compression:str, optional
            type of compression.
            The default is 'gzip'.
        kwargs:dict,
            dictionary of additional reading argument-value.
        """
        self.data = pd.read_csv(self.city_listings_data_url, compression=compression, **kwargs)

    def save_to_csv(self, filename:str, **kwargs):
        """
        Save to csv
        -------
        kwargs: dict of additional argument-value for saving to csv. See pd.DataFrame.to_csv
        """
        self.data.to_csv(filename, **kwargs)

    def print_html(self):
        """
        Print html tree
        """
        print(self.soup.prettify())

if __name__ == '__main__':
    # e.g.: listings data for the city of Paris
    url = 'https://insideairbnb.com/fr/get-the-data/' #optional, we could have ignore the url

    # instantiate the class
    scraper = ScrapFromInsideAirbnb(url) # or ScrapFromInsideAirbnb()
    # send a request
    scraper.request()
    # get the response of the request
    # which is the url to the listings csv data of the city
    scraper.get_data_url_of_city('paris')
    # we can then load the data into a dataframe (from this later url)
    scraper.load_to_df()
    # finally we can save this dataframe
    scraper.save_to_csv(filename='./listing_paris.csv')
