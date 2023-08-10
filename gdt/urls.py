import yaml
from collections import namedtuple
import os
import logging

logging.getLogger(__file__)


def create_urls():

    urls_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'urls.yml'))

    urls = {}
    _urls = namedtuple('end_point', ['url', 'description'])
    try:
        with open(urls_file, 'r') as fid:
            urls = yaml.safe_load(fid)
    except ValueError as e:
        logging.error(e)

    return {u: _urls(urls[u]['url'], urls[u]['description']) for u in urls}


end_points = create_urls()

