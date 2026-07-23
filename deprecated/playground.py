import requests

url = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search'
params = {
    'sf': 'code,name',
    'terms': 'F32.9'
}

try:
    response = requests.get(url, params=params)
    data = response.json()
    n_matches = data[0]
    codes = data[1]
    results = data[3]
    print(f'\n{n_matches}\n{codes}\n{results}')
except Exception as e:
    print(f'An error occured: {e}')


'''
Returns
========
1
['F32.9']
[['F32.9', 'Major depressive disorder, single episode, unspecified']]
'''