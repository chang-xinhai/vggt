# Usage
#     python3 download_collection.py -o <collection_owner> -c <collection_name>
# blob:https://app.gazebosim.org/983e90b0-ae77-4fcc-b02e-6d2142e24c3b
# Description
#     This script will download all models contained within a collection.

import json
import requests
import os
import argparse

parser = argparse.ArgumentParser(description='Download models from a collection.')
parser.add_argument('-d', '--download_path', type=str, default="./", help='The path to save the downloaded models.')
parser.add_argument('-c', '--continue_index', type=int, default=0, help='The index of the model to continue from.')
args = parser.parse_args()

download_path = os.path.join(args.download_path, "GSO_DATA")              
continue_index = args.continue_index

os.makedirs(download_path, exist_ok=True)

collection_name = 'Scanned Objects by Google Research'.replace(" ", "%20")
owner_name = 'GoogleResearch'


sensor_config_file = ''
private_token = ''

print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

page = 1
count = 0

# The Fuel server URL.
base_url ='https://fuel.gazebosim.org/'

# Fuel server version.
fuel_version = '1.0'

# Path to get the models in the collection
next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

# Path to download a single model in the collection
download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)

print(download_url)

# Iterate over the pages
while True:
    url = base_url + fuel_version + next_url
    # Get the contents of the current page.
    r = requests.get(url)

    if not r or not r.text:
        break

    # Convert to JSON
    models = json.loads(r.text)

    # Compute the next page's URL
    page = page + 1
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

    # Download each model
    for model in models:
        count+=1
        if count <= continue_index:
            continue
        
        model_name = model['name']
        save_path = os.path.join(download_path, model_name+'.zip')
        print ('Downloading (%d) %s' % (count, model_name), f"saved in {save_path}")

        download = requests.get(download_url+model_name+'.zip', stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

print('Downloading completed, saved in {}'.format(download_path))