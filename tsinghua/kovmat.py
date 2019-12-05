# coding: utf-8

# In[1]:


import requests
from lxml import etree
import time
import datetime
import os
import json

now = datetime.datetime.now

# In[2]:


TIME = 2
base_url = 'http://www.kovmat.sav.sk'
datasrc = 'kovmat'
venue_name = 'Kovove Materialy - Metallic Materials'
issn = '0023-432X'
data_file = './data/kovmat/'


# In[11]:


def parse(tree, rr, cc):
    titles = tree.xpath("//table[@width='95%']//p/font[@size='+1']")
    t = tree.xpath("//table[@width='95%']//td/font[@size='+1']")
    titles.extend(t)
    raw_titles = [etree.tostring(i).decode('utf8').strip() for i in titles]
    titles = [etree.tostring(i).decode('utf8')[19:-11].strip() for i in titles]
    titles

    authors_keywords = tree.xpath("//table[@width='95%']//p/font[@size='+0']")
    ak = tree.xpath("//table[@width='95%']//td/font[@size='+0']")
    authors_keywords.extend(ak)
    authors = [etree.tostring(i).decode('utf8')[19:-11].strip() for i in authors_keywords[::2]]
    keywords = [etree.tostring(i).decode('utf8')[34:-15].strip() for i in authors_keywords[1::2]]
    keywords

    dois = tree.xpath('//table//p/text()[2]')
    d = tree.xpath('//table//td/text()')[13]
    dois.append(d)
    dois = [i.split('DOI:')[1].strip() for i in dois if 'DOI' in i]
    dois

    vnps = tree.xpath('//table//p/text()[1]')
    v = tree.xpath('//table//td/text()')[12]  # diff
    vnps.append(v)
    vols = [i.split(',')[0].split('vol.')[1].strip() for i in vnps if 'vol' in i]
    vols = [i.split()[0] for i in vols]
    nos = [i.split(',')[1].split('no.')[1].strip() for i in vnps if 'no' in i]
    pps = [i.split(',')[2].split('pp.')[1].strip() for i in vnps if 'pp' in i]
    pps = [i.replace(' ', '') for i in pps]
    vols, nos, pps

    page_start = [p.split('-')[0].strip() for p in pps]
    page_end = [p.split('-')[1].strip() for p in pps]
    page_start, page_end

    # [print(len(x)) for x in [titles,authors,keywords,dois,vols,nos,page_start,page_end,pps]]

    [print(i) for i in page_start]
    pdfs = [list([f'http://www.kovmat.sav.sk/full.php?rr={rr}&cc={cc}&ss={page_start[i]}']) for i in range(len(titles))]

    urls = [list([link[0].replace('full', 'abstract')]) for link in pdfs]
    abstracts = [parse_abstract(url) for url in urls]

    dois = dois if len(dois) == len(titles) else [""] * len(titles)
    assert len(titles) == len(authors) == len(keywords) == len(dois) == len(pdfs)

    return urls, raw_titles, titles, abstracts, authors, keywords, dois, vols, nos, page_start, page_end, pps, pdfs


def parse_abstract(url):
    time.sleep(TIME)
    response = requests.get(url[0])
    abstract = ''
    if response.ok:
        tree = etree.HTML(response.text)
        abstract = tree.xpath('//font[@style="text-align: justify"]')[0]
        abstract = etree.tostring(abstract).decode('utf8')
    else:
        print('Abstract Extraction Error！', response.status_code)
    return abstract


# parse(tree, 49, 6)
# parse_abstract(['http://www.kovmat.sav.sk/abstract.php?rr=49&cc=6&ss=393'])


# In[4]:


def generate(infos, year, venue, src='kovove', issn='0023-432X'):
    def foo_title(x):
        x = x.replace('<sub>', '')
        x = x.replace('</sub>', '')
        return x

    def foo_author(author):
        a_list = []
        for i, name in enumerate(author.split('.,')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = str(hash(name))
            a_list.append(a)
        return a_list

    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])

    pubs = []
    for url, raw_title, title, abstract, author, keyword, doi, vol, no, page_start, page_end, pp, pdf in zip(*infos):
        pub = dict()
        pub.update(zip(
            ['url', 'raw_title', 'title', 'abstract', 'authors', 'venue', 'keywords', 'page_start', 'page_end',
             'page_str', 'volume', 'issue', 'pdf_src', 'hash', 'ts', 'src', 'lang', 'sid', 'year', 'issn'],
            [url, raw_title, foo_title(title), abstract, foo_author(author), venue,
             ';'.join([i.strip() for i in keyword.split(',')]), page_start, page_end, pp, vol, no, pdf, foo_hash(title),
             str(now()), src, 'en', str(hash(title)), year,
             issn]))

        if doi != '' or doi.strip() != '':
            pub['doi'] = doi.strip()
        #         print(repr(pub))
        #         print('--------------')
        pubs.append(pub)

    return pubs


# response = requests.get('http://www.kovmat.sav.sk/issue.php?rr=33&cc=1')
# tree = etree.HTML(response.text)
# r = parse(tree, 33, 1)
# venue = {'name':'Kovove Materialy - Metallic Materials', 'type':1, 'sid':'1'}
# r = generate(r, 2019, venue)


# In[5]:


# Save
def save(data, file):
    if os.path.exists(file):
        with open(file, 'a') as f:
            # f.write(repr(data))
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        with open(file, 'w') as f:
            # f.write(repr(data))
            json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Saved records: {len(data)}')


# In[6]:


# import copy
# save_data = copy.deepcopy(all_pubs)

# for line in save_data:
#     line['sid'] = str(line['sid'])
#     for author in line['authors']:
#         author['sid'] = str(author['sid'])

#     year = line['venue']
#     line['year'] = year
#     line['venue'] = {'name':venue_name, 'type':1, 'sid':'1'}

# save(save_data, data_file+'kovmat_1995-2019.json')


# In[12]:


if __name__ == '__main__':
    all_pubs = []
    for rr in range(58, 59):
        vol_pubs = []
        for cc in range(1, 7):
            url = f'http://www.kovmat.sav.sk/issue.php?rr={rr}&cc={cc}'

            year = 1995 - 33 + rr
            print(f'parsing {rr}-{cc}-{year}')

            venue = {'name': 'Kovove Materialy - Metallic Materials', 'type': 1, 'sid': '1'}

            response = requests.get(url)
            time.sleep(TIME)

            if response.ok:
                tree = etree.HTML(response.text)

                # 0 records
                records = int(tree.xpath('.//i[contains(text(),"records ")]/text()')[0].split()[1])
                if records <= 1: continue
                print(f'found  records: {records} ')

                inofs = parse(tree, rr, cc)
                pubs = generate(inofs, year, venue)
                print(f'spider success: {len(pubs)}')

                vol_pubs.extend(pubs)
            else:
                print('ERROR! ', response.status_code)
        save(vol_pubs, f'{data_file}kovmat_{year}.json')
        all_pubs.extend(vol_pubs)

print('All spidered!')
save(all_pubs,  f'{data_file}kovmat_all.json')


# # Parse Lasted

# In[48]:


def parse_without_pages(tree, rr, cc):
    titles = tree.xpath("//table[@width='95%']//p/font[@size='+1']")
    t = tree.xpath("//table[@width='95%']//td/font[@size='+1']")
    titles.extend(t)
    raw_titles = [etree.tostring(i).decode('utf8').strip() for i in titles]
    titles = [etree.tostring(i).decode('utf8')[19:-11].strip() for i in titles]
    titles

    authors_keywords = tree.xpath("//table[@width='95%']//p/font[@size='+0']")
    ak = tree.xpath("//table[@width='95%']//td/font[@size='+0']")
    authors_keywords.extend(ak)
    authors = [etree.tostring(i).decode('utf8')[19:-11].strip() for i in authors_keywords[::2]]
    keywords = [etree.tostring(i).decode('utf8')[34:-15].strip() for i in authors_keywords[1::2]]
    keywords

    dois = tree.xpath('//table//p/text()[2]')
    d = tree.xpath('//table//td/text()')[13]
    dois.append(d)
    dois = [i.split('DOI:')[1].strip() for i in dois if 'DOI' in i]
    dois

    vnps = tree.xpath('//table//p/text()[1]')
    v = tree.xpath('//table//td/text()')[12]  # diff
    vnps.append(v)
    vols = [i.split(',')[0].split('vol.')[1].strip() for i in vnps if 'vol' in i]
    vols = [i.split()[0] for i in vols]
    nos = [i.split(',')[1].split('no.')[1].strip() for i in vnps if 'no' in i]
    pps = [i.split(',')[2].split('pp.')[1].strip() for i in vnps if 'pp' in i]
    pps = [i.replace(' ', '') for i in pps]
    vols, nos, pps

    abs_urls = tree.xpath('//a[contains(@href,"abstract")]/@href')
    urls = [list([base_url + '/' + i]) for i in abs_urls]
    abstracts = [parse_abstract(u) for u in urls]

    dois = dois if len(dois) == len(titles) else [""] * len(titles)
    assert len(titles) == len(authors) == len(keywords) == len(dois)

    return urls, raw_titles, titles, abstracts, authors, keywords, dois, vols, nos


# In[51]:


def generate_without_pages(infos, year, venue, src='kovove', issn='0023-432X'):
    def foo_title(x):
        x = x.replace('<sub>', '')
        x = x.replace('</sub>', '')
        return x

    def foo_author(author):
        a_list = []
        for i, name in enumerate(author.split('.,')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = str(hash(name))
            a_list.append(a)
        return a_list

    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])

    pubs = []
    for url, raw_title, title, abstract, author, keyword, doi, vol, no in zip(*infos):
        pub = dict()
        pub.update(zip(
            ['url', 'raw_title', 'title', 'abstract', 'authors', 'venue', 'keywords', 'volume', 'issue', 'hash', 'ts',
             'src', 'lang', 'sid', 'year', 'issn'],
            [url, raw_title, foo_title(title), abstract, foo_author(author), venue,
             ';'.join([i.strip() for i in keyword.split(',')]), vol, no, foo_hash(title), str(now()), src, 'en',
             str(hash(title)), year,
             issn]))

        if doi != '' or doi.strip() != '':
            pub['doi'] = doi.strip()
        #         print(repr(pub))
        #         print('--------------')
        pubs.append(pub)

    return pubs


# response = requests.get('http://www.kovmat.sav.sk/issue.php?rr=33&cc=1')
# tree = etree.HTML(response.text)
# r = parse(tree, 33, 1)
# venue = {'name':'Kovove Materialy - Metallic Materials', 'type':1, 'sid':'1'}
# r = generate(r, 2019, venue)


# In[52]:


rr = 58
cc = 1
url = f'http://www.kovmat.sav.sk/issue.php?rr={rr}&cc={cc}'
tree = etree.HTML(response.text)
# 0 records
records = int(tree.xpath('.//i[contains(text(),"records ")]/text()')[0].split()[1])
print(f'found  records: {records} ')
inofs = parse_without_pages(tree, rr, cc)
pubs = generate_without_pages(inofs, year, venue)
print(f'spider success: {len(pubs)}')
save(pubs, f'{data_file}kovmat_2020.json')

# In[53]:


all_data = []
for year in range(1995, 2021):
    with open(f'{data_file}kovmat_{year}.json') as f:
        data = eval(''.join(f.readlines()))
        all_data.extend(data)

print(len(all_data))
save(all_data, f'{data_file}kovmat_all.json')

