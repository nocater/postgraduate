
# coding: utf-8

# # BRAZILIAN ARCHIVES OF BIOLOGY AND TECHNOLOGY

# In[2]:


import requests
from lxml import etree
import time
import datetime
import os
import json
from tqdm import tqdm
import re
now = datetime.datetime.now


# In[3]:


TIME = 1
base_url = 'https://doaj.org/toc/1678-4324'
venue_name = 'BRAZILIAN ARCHIVES OF BIOLOGY AND TECHNOLOGY'
datasrc = 'baobat'
issn = '1516-8913'
data_file='./data/baobat/'

journal = venue_name
venue = {'name':venue_name, 'type':1, 'sid':datasrc}


# In[4]:


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


# # Spider All Data
# 此数据源可找到josn数据API：
# [All data.json](https://doaj.org/query/article/_search?ref=toc&callback=jQuery34004152944930358162_1577268946017&source=%7B%22query%22%3A%7B%22filtered%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22index.issn.exact%22%3A%5B%221516-8913%22%2C%221678-4324%22%5D%7D%7D%2C%7B%22term%22%3A%7B%22_type%22%3A%22article%22%7D%7D%5D%7D%7D%2C%22query%22%3A%7B%22match_all%22%3A%7B%7D%7D%7D%7D%2C%22size%22%3A2800%2C%22aggs%22%3A%7B%22volume%22%3A%7B%22terms%22%3A%7B%22field%22%3A%22bibjson.journal.volume.exact%22%2C%22size%22%3A10%2C%22order%22%3A%7B%22_count%22%3A%22desc%22%7D%7D%7D%2C%22issue%22%3A%7B%22terms%22%3A%7B%22field%22%3A%22bibjson.journal.number.exact%22%2C%22size%22%3A10%2C%22order%22%3A%7B%22_count%22%3A%22desc%22%7D%7D%7D%2C%22year_published%22%3A%7B%22date_histogram%22%3A%7B%22field%22%3A%22index.date%22%2C%22interval%22%3A%22year%22%7D%7D%2C%22month_published%22%3A%7B%22date_histogram%22%3A%7B%22field%22%3A%22index.date_toc_fv_month%22%2C%22interval%22%3A%22month%22%7D%7D%7D%2C%22_source%22%3A%7B%7D%7D&_=1577268946018)

# In[ ]:


# url = 'https://doaj.org/query/article/_search?ref=toc&callback=jQuery34004152944930358162_1577268946017&source=%7B%22query%22%3A%7B%22filtered%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22index.issn.exact%22%3A%5B%221516-8913%22%2C%221678-4324%22%5D%7D%7D%2C%7B%22term%22%3A%7B%22_type%22%3A%22article%22%7D%7D%5D%7D%7D%2C%22query%22%3A%7B%22match_all%22%3A%7B%7D%7D%7D%7D%2C%22size%22%3A2800%2C%22aggs%22%3A%7B%22volume%22%3A%7B%22terms%22%3A%7B%22field%22%3A%22bibjson.journal.volume.exact%22%2C%22size%22%3A10%2C%22order%22%3A%7B%22_count%22%3A%22desc%22%7D%7D%7D%2C%22issue%22%3A%7B%22terms%22%3A%7B%22field%22%3A%22bibjson.journal.number.exact%22%2C%22size%22%3A10%2C%22order%22%3A%7B%22_count%22%3A%22desc%22%7D%7D%7D%2C%22year_published%22%3A%7B%22date_histogram%22%3A%7B%22field%22%3A%22index.date%22%2C%22interval%22%3A%22year%22%7D%7D%2C%22month_published%22%3A%7B%22date_histogram%22%3A%7B%22field%22%3A%22index.date_toc_fv_month%22%2C%22interval%22%3A%22month%22%7D%7D%7D%2C%22_source%22%3A%7B%7D%7D&_=1577268946018'

# response = requests.get(url)
# data = json.loads(response.text[41:-1])
# data = data['hits']['hits']
# save(data, f'{data_file}origin.json')


# # Parse Data

# In[ ]:


data = json.loads(''.join(open(f'{data_file}origin.json').readlines()))


# In[ ]:


vol_year = {'59' : 2016 ,
'46' : 2003 ,
'51' : 2008 ,
'49' : 2006 ,
'42' : 1999 ,
'60' : 2017 ,
'53' : 2010 ,
'41' : 1998 ,
'48' : 2005 ,
'57' : 2014 ,
'jubilee' : 2001 ,
'44' : 2001 ,
'54' : 2011 ,
'62' : 2019 ,
'50' : 2007 ,
'58' : 2015 ,
'55' : 2012 ,
'45' : 2002 ,
'47' : 2004 ,
'56' : 2013 ,
'61' : 2018 ,
'52' : 2009 ,
'43' : 2000 ,}


# In[ ]:


def parse_pdf(url):
    response = requests.get(url)
    if not response.ok        or 'Page not found!' in response.text: 
        return None
    tree = etree.HTML(response.text)
    pdf = tree.xpath('//a[contains(@href,"/pdf/")]/@href')[0]
    pdf = 'http://www.scielo.br'+pdf
    return pdf

def parse(info):
    def foo_author(authors):
        for i,auth in enumerate(authors):
            auth['pos'] = i
            auth['sid'] = str(hash(auth['name']))
        return authors
    
    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])
    
    pub = {}
    
    pub['url'] = ['https://doaj.org/'+info['_type']+'/'+info['_id']]
    link = info['_source']['bibjson']['link']
    if isinstance(link, list):
        pub['url'].append(link[0]['url'])
    else:
        pub['url'] = [link['url']]
    
    pub['raw_title'] = '<b><a name="tx"></a>'+info['_source']['bibjson']['title']+'</b>'
    pub['title'] = info['_source']['bibjson']['title']
    if 'abstract' in info['_source']['bibjson'].keys():
        pub['abstract'] = info['_source']['bibjson']['abstract']
    if 'author' in info['_source']['bibjson'].keys():
        authors = info['_source']['bibjson']['author']
        pub['authors'] = foo_author(authors)
    pub['venue'] = venue
    if 'keywords' in info['_source']['bibjson'].keys():
        pub['keywords'] = ';'.join(info['_source']['bibjson']['keywords'])
    
    if 'page_start' in info['_source']['bibjson'].keys():
        pub['page_start'] = info['_source']['bibjson']['start_page']
        pub['page_end'] = info['_source']['bibjson']['end_page']
        pub['page_str'] = pub['page_start'] + '-' + pub['page_end']
    
    pub['volume'] = info['_source']['bibjson']['journal']['volume']
    if 'number' in info['_source']['bibjson']['journal'].keys():
        pub['issue'] = info['_source']['bibjson']['journal']['number']
    
    # pub['pdf_src'] = info['_source']['index']['fulltext']
    pdf_url = 'http:'+info['_source']['index']['fulltext']
    pdf_href = parse_pdf(pdf_url)
    if pdf_href:
        pub['pdf_src'] = pdf_href
    pub['hash'] = foo_hash(pub['title'])
    pub['ts'] = str(now())
    pub['src'] = datasrc
    pub['lang'] = info['_source']['bibjson']['journal']['language'][0].lower()
    pub['sid'] = str(hash(pub['title']))
    if 'year' in info['_source']['bibjson']:
        pub['year'] = int(info['_source']['bibjson']['year'])
    else:
        pub['year'] = vol_year[pub['volume']]
    
    pub['issn'] = issn
    if 'doi' in info['_source']['index'].keys():
        pub['doi'] = info['_source']['index']['doi']
    
    # [print(f'{repr(k)}:\t{repr(v)}') for k,v in pub.items()]
    
    return pub

# parse(data[1])
# data[1]['_source']['bibjson']['journal'].keys()


# In[ ]:


# data_json = list(map(parse, data))
# save(data_json, f'{data_file}{venue_name.replace(" ", "_")}.json')


pubs = []
for i,d in zip(tqdm(range(len(data[1979:]))),data[1979:]) :
    pubs.append(parse(d))
save(pubs, f'{data_file}{venue_name.replace(" ", "_")}_v1.json')


# # Parse Details

# In[55]:


def parse_details(url):
    response = requests.get(url)
    if response.ok:
        html = response.text
        tree = etree.HTML(html)
        
        # Page Type 1
        institutions = tree.xpath('//p[contains(@class,"aff")]//text()')
        if institutions:
            print('Page Type 1')
            institutions = [i.strip() for i in institutions if i.strip()]
            institutions = {institutions[i]:institutions[i+1] for i in range(0, len(institutions), 2)}
            
            authors = tree.xpath('//p[contains(@class,"author")]')
            authors_institutions = []
            for author in authors:
                index = author.xpath('sup/a/text()')
                index = [i.strip() for i in index if ('1'<=i.strip()<='9')]
                author_ins = [institutions[i] for i in index]
                authors_institutions.append(author_ins)
                
                
            references = tree.xpath('//p[contains(@class,"ref")]')
            references = [etree.tostring(i).decode('utf8').strip() for i in references]
            return authors_institutions, references
        else: # Page Type 2
            print('Page Type 2')
            index = tree.xpath('//a[text()="*"]/../../sup/text()')
            if 'I' not in index:
                index = tree.xpath('//a/sup[text()="*"]/../../sup/text()')
            
            keys = tree.xpath('//p[position()<10]/font/sup/text()')
            values = tree.xpath('//p[position()<10]/font/sup/../text()')
            
            index = [i.strip() for i in index if i.strip()]
            keys = [i.strip() for i in keys if i.strip()]
            values = [i.strip() for i in values if i.strip()]
            
            authors_institutions = []
            if len(keys) == len(values) == 0:
                instititions = tree.xpath('//p[position()<10]/font/text()')
                instititions = [i.strip() for i in instititions if i.strip()]
                instititions = ' '.join(instititions)
                authors_institutions = [instititions]
            else:
                institutions = dict(zip(keys, values))
                for ind in index:
                    positions = ind.split(',')
                    if len(positions) == 1:
                        positions = ind.split()
                    positions = [i.strip() for i in positions if i.strip()]
                    author_ins = [institutions[i.strip()] for i in positions]
                    authors_institutions.append(author_ins)
        
            references = re.findall('<!-- ref -->(.*?)<!-- end-ref -->', html)
            return authors_institutions, references
            
        # print(len(authors_institutions))
        # [print(len(i), i) for i in authors_institutions]
        # print('r:', lens, len(authors_institutions))

        
    else:
        print('parse_details error:', response.status_code, 'at ', url)
        return None,None


# In[ ]:


pubs = json.loads(''.join(open(f'{data_file}{venue_name.replace(" ", "_")}_v1.json').readlines()))
# for i,p in zip(tqdm(range(len(pubs))), pubs):
for i,p in enumerate(pubs):
    if 'authors' not in p.keys(): continue
    
    institutions, references = parse_details(p['url'][-1])
    
    if references:
        p['reference'] = references
    
    # 多个作者只有一个组织 直接copy times
    if len(institutions)==1 and len(p["authors"])>1:
        institutions = [institutions[0]] * len(p["authors"])

    print(f'\r{i}/{len(pubs)} authors:{len(p["authors"])}-{len(institutions)} references:{len(references)}')
    
    if institutions:
        if len(institutions) == len(p['authors']):
            for i in range(len(institutions)):
                p['authors'][i]['org'] = institutions[i]
    else:
        if 'authors' in p.keys() and len(p['authors'])>0:
            print(f'Error institutions not match authors:{i} {p["url"][-1]}')
    
#     print(f'\r{i}/{len(pubs)} authors:{len(p["authors"])}-{len(institutions)} references:{len(references)}', end='')
save(pubs, f'{data_file}{venue_name.replace(" ", "_")}_v2.json')

