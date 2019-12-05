
# coding: utf-8

# In[1]:


import requests
from lxml import etree
import time
import datetime
import os
import json
now = datetime.datetime.now


# In[142]:


TIME = 2
base_url = 'http://www.southasiaarchive.com'
datasrc = 'southasiaarchive'
venue_name = 'JOURNAL OF THE INDIAN CHEMICAL SOCIETY'
# issn = '1'
data_file='./data/southasiaarchive/'


# In[74]:


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
    print(f'saved records: {len(data)}')


# In[143]:


def url_generator(url='http://www.southasiaarchive.com/Content/sarf.120026'):
    response = requests.get(url)
    tree = etree.HTML(response.text)
    h4s = tree.xpath('//h4/span/text()')
    uls = tree.xpath('//ul[@class="js-loi-block"]')

    for vol, ul in zip(h4s, uls):
        for li in ul.xpath('li'):
            url = base_url + li.xpath('a/@href')[0]
            issue = li.xpath('a/text/text()')[0].split()[1]
            year = int(li.xpath('text()')[1].split()[0])
            print(vol,issue,year,url)
            yield vol,issue,year,url


# In[144]:


def parse(tree):
    urls = tree.xpath('//td[@class="toc-title-col"]/a/@href')
    urls = [base_url + i for i in urls]
    
    authors = [parse_authors(url) for url in urls]
    # only one author      
#     authors = tree.xpath('//td[@class="toc-author-col"]/text()')
#     authors = [i.strip() for i in authors]
    
    titles = tree.xpath('//td[@class="toc-title-col"]/a/@title')
    
    raw_titles = tree.xpath('//td[@class="toc-title-col"]/a')
    raw_titles = [etree.tostring(i).decode('utf8').strip() for i in raw_titles]
    
    pages = tree.xpath('//td[@class="toc-pages-col"]/text()')
    pages = [i.strip() for i in pages]
    pages_start = [i.split('-')[0].strip() for i in pages]
    pages_end = [i.split('-')[1].strip() for i in pages]
    
    pdfs = tree.xpath('//td[@class="toc-pdf-col"]/a/@href')
    pdfs = [base_url+i+'480' for i in pdfs]
    
    # [print(i) for i in pdfs]
    assert len(urls) == len(authors) == len(titles) == len(pages) == len(pdfs)
    # return raw_titles,titles,authors,pages_start,pages_end,pages,pdfs
    return urls,raw_titles,titles,authors,pages_start,pages_end,pages,pdfs

def parse_authors(url):
    time.sleep(1)
    response = requests.get(url)
    tree = etree.HTML(response.text) # [contains(text(),"PDF")]
    td = tree.xpath('//td[contains(text(),"Author")]')[0]
    authors = td.xpath('following-sibling::td//text()')[0]
    authors = authors.strip()
    # print(authors)
    return authors

# ite = url_generator()
# vol,issue,year,url = next(ite)
# url = 'http://www.southasiaarchive.com//Content/sarf.120026/205512'
# response = requests.get(url)
# tree = etree.HTML(response.text)
# parse(tree, vol, issue)
# parse_authors('http://www.southasiaarchive.com/Content/sarf.120026/205512/002')


# In[145]:


def generate(infos, year, vol, issue, venue, src=datasrc):
    
    def foo_author(author):
        a_list = []
        for i, name in enumerate(author.split('&')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = str(hash(name))
            a_list.append(a)
        return a_list
    
    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])
    
    pubs = []
    for url,raw_title,title,author,page_start,page_end,page,pdf in zip(*infos):
        pub = dict()
        pub.update(zip(['url','raw_title','title','authors','venue','page_start','page_end','page_str','volume','issue','pdf_src','hash','ts','src','lang','sid','year'],                           [[url], raw_title, title, foo_author(author), venue, page_start, page_end, page, vol, issue, pdf,                            foo_hash(title),str(now()), src, 'en', str(hash(title)), year]))

#         print(repr(pub))
#         print('--------------')
#         [print(f'{k}:{v}') for k,v in pub.items()]
        pubs.append(pub)
    
    return pubs


# In[146]:


if __name__ == '__main__':
    all_pubs = []
    ite = url_generator()
    venue = {'name':venue_name, 'type':1, 'sid':'1'}
    try:
        while True:
            vol,issue,year,url = next(ite)
            
            response = requests.get(url)
            time.sleep(TIME)
            
            if response.ok:
                tree = etree.HTML(response.text)
                infos = parse(tree)
                pubs = generate(infos, year, vol, issue, venue)
                print(f'spider success:\t{len(pubs)}')
                all_pubs.extend(pubs)
            else:
                print('ERROR! ', response.status_code)
    except Exception as e:
        print(e)
        print('All spidered!')
        
save(all_pubs,  f'{data_file}{datasrc}_all.json')

