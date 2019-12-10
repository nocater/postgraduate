
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


TIME = 1
base_url = 'http://www.techno-press.org/'
datasrc = 'techno-press_anr'
venue_name = 'Advances in Nano Research'
issn = '2287-5301'
data_file='./data/anr/'

journal='anr'
venue = {'name':venue_name, 'type':1, 'sid':datasrc}


# In[3]:


def parse(tree):
    tag_a = tree.xpath('//a[contains(@href,"page=article")]')
    
    urls = [i.get('href') for i in tag_a]
    urls = [base_url+i for i in urls]
    
    pdfs = [i.replace('/content/?page=article&','/download.php?') for i in urls]
    
    raw_titles = [etree.tostring(i).decode('utf8').strip() for i in tag_a]
    titles = [i.xpath('text()')[0].strip() for i in tag_a]
    
    authors = tree.xpath('//div[@class="paper_info"]/ul/table/tr[1]/td/text()[2]')
    authors = [i.strip().replace(' and', ',') for i in authors]
    
    abstracts, keywords, addresses = [],[],[]
    for view in tree.xpath('//p[@class="paper_view"]'):
        infos = view.xpath('text()')
        infos = [i.strip() for i in infos if i.strip()]
        
        # lamb = lambda x:x[0].append(x[1])
        # [lamb(i) for i in zip([abstracts, keywords, addresses], infos)]
        # equals above code
        abstracts.append(etree.tostring(view).decode('utf8').strip())
        keywords.append(infos[1])
        addresses.append(infos[2])
        
    pages = tree.xpath('//td[@class="pages"]//text()')
    pages = [i.split()[1].strip().replace('.', '') for i in pages if '-' in i and i.split()]
    
    page_start = [i.split('-')[0] for i in pages]
    page_end = [i.split('-')[1] for i in pages]
    
    dois = tree.xpath('//div[@class="paper_info"]/ul/table/tr[2]/td[3]//text()')
    dois = [i.replace('DOI:', '').strip() for i in dois]
    
    #[print(repr(i)) for i in abstracts]
    assert len(urls) == len(titles) == len(pdfs) == len(authors)            == len(abstracts) == len(keywords) == len(addresses)
    return urls,raw_titles,titles,abstracts,authors,keywords,dois,page_start,page_end,pages,pdfs

# url = 'http://www.techno-press.org/?page=container&volumeno=1/1&journal=acc'
# response = requests.get(url)
# tree = etree.HTML(response.text)
# parse(tree)


# In[4]:


def generate(infos, vol, no, year, venue, src='kovove', issn='0023-432X'):
    
    def foo_author(author):
        a_list = []
        for i, name in enumerate(author.split(',')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = str(hash(name))
            a_list.append(a)
        return a_list
    
    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])
    
    pubs = []
    for urls,raw_title,title,abstract,author,keyword,doi,page_start,page_end,page,pdf in zip(*infos):
        pub = dict()
        pub.update(zip(['url','raw_title','title','abstract','authors','venue','keywords','page_start','page_end','page_str','volume','issue','pdf_src','hash','ts','src','lang','sid','year','issn','doi'],                           [[url], raw_title, title, abstract, foo_author(author), venue, ';'.join([i.strip() for i in keyword.split(',')]),page_start, page_end, page, vol, no, pdf,                            foo_hash(title), str(now()), src, 'en', str(hash(title)), year,
                           issn, doi]))

#         print(repr(pub))
#         print('--------------')
        pubs.append(pub)
    
    return pubs


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
    print(f'saved records: {len(data)}')


# In[6]:


if __name__ == '__main__':
    vol_nos = ['1/1', '1/2', '1/3', '1/4', '2/1', '2/2', '2/3', '2/4',              '3/1', '3/2', '3/3', '3/4', '4/1', '4/2', '4/3', '4/4',              '5/1', '5/2', '5/3', '5/4', '6/1', '6/2', '6/3', '6/4',              '7/1', '7/2', '7/3', '7/4', '7/5', '7/6']

    all_pubs = []
    for vol_no in vol_nos:
        print(f'parsing:{vol_no}')
        vol, no = vol_no.split('/')
        year = 2012 + int(vol)
        url = f'http://www.techno-press.org/?page=container&volumeno={vol_no}&journal={journal}'

        time.sleep(TIME)
        response = requests.get(url)
        tree = etree.HTML(response.text)
        infos = parse(tree)
        pubs = generate(infos, vol, no, year, venue, src=datasrc, issn=issn)
        all_pubs.extend(pubs)

    save(all_pubs, f'{data_file}{datasrc}_all.json')

