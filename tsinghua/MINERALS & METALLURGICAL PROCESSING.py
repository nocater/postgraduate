
# coding: utf-8

# In[1]:


import requests
from lxml import etree
import time
import datetime
import os
import json
now = datetime.datetime.now


# In[120]:


TIME = 1
base_url = 'http://mmp.smenet.org/'
datasrc = 'MINERALS_&_METALLURGICAL_PROCESSING'
venue_name = 'MINERALS & METALLURGICAL PROCESSING'
issn = '0747-9182'
data_file='./data/m&mp/'

journal='m&mp'
venue = {'name':venue_name, 'type':1, 'sid':datasrc}


# In[31]:


# 获取所有vol 链接信息
url = 'http://mmp.smenet.org/issue.cfm'
response = requests.get(url)
tree = etree.HTML(response.text)

links = ''

for li in tree.xpath('//li[contains(text(),"Volume")]'):
    vol, no = li.xpath('text()')[0].split(',')
    vol = vol.split()[-1]
    no = no.split()[-1]
    
    href = base_url+li.xpath('a/@href')[0]
    year = li.xpath('a/text()')[0].split()[1]
    
    # print(year, vol, no, href)
    links+= ' '.join([year, vol, no, href]) + '\n'


# In[40]:


BASE_INFO = """2018 35 4 http://mmp.smenet.org/issue.cfm?issueid=322
2018 35 3 http://mmp.smenet.org/issue.cfm?issueid=319
2018 35 2 http://mmp.smenet.org/issue.cfm?issueid=315
2018 35 1 http://mmp.smenet.org/issue.cfm?issueid=310
2017 34 4 http://mmp.smenet.org/issue.cfm?issueid=306
2017 34 3 http://mmp.smenet.org/issue.cfm?issueid=300
2017 34 2 http://mmp.smenet.org/issue.cfm?issueid=298
2017 34 1 http://mmp.smenet.org/issue.cfm?issueid=292
2016 33 4 http://mmp.smenet.org/issue.cfm?issueid=289
2016 33 3 http://mmp.smenet.org/issue.cfm?issueid=284
2016 33 2 http://mmp.smenet.org/issue.cfm?issueid=280
2016 33 1 http://mmp.smenet.org/issue.cfm?issueid=273
2015 32 4 http://mmp.smenet.org/issue.cfm?issueid=270
2015 32 3 http://mmp.smenet.org/issue.cfm?issueid=266
2015 32 2 http://mmp.smenet.org/issue.cfm?issueid=262
2015 32 1 http://mmp.smenet.org/issue.cfm?issueid=256
2014 31 4 http://mmp.smenet.org/issue.cfm?issueid=253
2014 31 3 http://mmp.smenet.org/issue.cfm?issueid=249
2014 31 2 http://mmp.smenet.org/issue.cfm?issueid=245
2014 31 1 http://mmp.smenet.org/issue.cfm?issueid=241
2013 30 4 http://mmp.smenet.org/issue.cfm?issueid=235
2013 30 3 http://mmp.smenet.org/issue.cfm?issueid=230
2013 30 2 http://mmp.smenet.org/issue.cfm?issueid=165
2013 30 1 http://mmp.smenet.org/issue.cfm?issueid=161
2012 29 4 http://mmp.smenet.org/issue.cfm?issueid=156
2012 29 3 http://mmp.smenet.org/issue.cfm?issueid=142
2012 29 2 http://mmp.smenet.org/issue.cfm?issueid=137
2012 29 1 http://mmp.smenet.org/issue.cfm?issueid=136
2011 28 4 http://mmp.smenet.org/issue.cfm?issueid=141
2011 28 3 http://mmp.smenet.org/issue.cfm?issueid=140
2011 28 2 http://mmp.smenet.org/issue.cfm?issueid=139
2011 28 1 http://mmp.smenet.org/issue.cfm?issueid=138
2010 27 4 http://mmp.smenet.org/issue.cfm?issueid=147
2010 27 3 http://mmp.smenet.org/issue.cfm?issueid=146
2010 27 2 http://mmp.smenet.org/issue.cfm?issueid=145
2010 27 1 http://mmp.smenet.org/issue.cfm?issueid=144
2009 26 4 http://mmp.smenet.org/issue.cfm?issueid=152
2009 26 3 http://mmp.smenet.org/issue.cfm?issueid=151
2009 26 2 http://mmp.smenet.org/issue.cfm?issueid=150
2009 26 1 http://mmp.smenet.org/issue.cfm?issueid=148
2008 25 4 http://mmp.smenet.org/issue.cfm?issueid=153"""

BASE_INFO = [i.split() for i in BASE_INFO.split('\n')]


# In[113]:


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


# In[110]:


# 解析vol_page
def parse_vol(vol_href):
    response = requests.get(vol_href)
    tree = etree.HTML(response.text)
    urls = tree.xpath('//a[contains(@href, "abstract")]/@href')
    urls = [base_url+i for i in urls]
    
    # print(urls)
    return urls


# In[159]:


def parse_page(tree):
    title = tree.xpath('//h1/text()')[1].strip()
    raw_title = etree.tostring(tree.xpath('//h1')[1]).decode('utf8').strip()
    
    page_athor = [i.strip() for i in tree.xpath('//h2/text()') if i.strip()]
    page = page_athor[0].split()[-1]
    
    author = page_athor[1] if len(page_athor)>1 else None
    
    page_star, page_end = page.split('-')
    
    e_doi = tree.xpath('//a[contains(@href, "doi")]/@href')
    doi = e_doi[0].strip() if len(e_doi)>0 else None
    
    e_abstract = tree.xpath('//div[@style="padding:35px;"]')[0]
    abstract = etree.tostring(e_abstract).decode('utf8')
    abstract = '<div style="padding:35px;">'+abstract.split('ABSTRACT:')[1]
    # print(abstract)
    
    return title, raw_title, author, page, page_star, page_end, doi, abstract

# reponse = requests.get('http://mmp.smenet.org/abstract.cfm?aid=6747')
# tree = etree.HTML(reponse.text)
# parse_page(tree)


# In[133]:


def generate(info, url, vol, no, year, venue, src=datasrc, issn=issn):
    def foo_title(x):
        x = x.replace('<sub>','')
        x = x.replace('</sub>','')
        x = x.replace('\\', '')
        x = ' '.join(x.split())
        return x
    
    def foo_author(author):
        if author == None: return None
        a_list = []
        for i, name in enumerate(author.split(';')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = str(hash(name))
            a_list.append(a)
        return a_list
    
    def foo_hash(title):
        return ''.join([i[0].lower() for i in title.split()])
    
    title, raw_title, author, page, page_start, page_end, doi, abstract = info
    pub = dict()
    pub.update(zip(['url','raw_title','title','abstract','authors','venue','page_start','page_end','page_str','volume','issue','hash','ts','src','lang','sid','year','issn','doi'],                       [[url], raw_title, foo_title(title), abstract, foo_author(author), venue, page_start, page_end, page, vol, no,                        foo_hash(title), str(now()), src, 'en', str(hash(title)), year,
                       issn, doi]))
    
    keys = [k for k,v in pub.items() if v==None]
    if keys: print(f'del keys:{keys}')
    [pub.pop(k) for k in keys]
    return pub


# In[160]:


pubs = []
for line in BASE_INFO:
    year, vol, no, vol_href = line
    print(f'parsing{year,vol,no}')
    year = int(year)
    
    urls = parse_vol(vol_href)
    
    for url in urls:
        print(url)
        reponse = requests.get(url)
        tree = etree.HTML(reponse.text)
        info = parse_page(tree)
        
        pub = generate(info, url, vol, no, year, venue)
        pubs.append(pub)
    print(f'records:{len(urls)}')
    time.sleep(TIME)
save(pubs, data_file+f'{datasrc}_all.json')


# In[162]:


save(pubs, data_file+f'{datasrc}_all_-1.json')


# 'http://mmp.smenet.org/abstract.cfm?aid=2855' 无法解析  
# 手动解析最后一条

# In[163]:


url = 'http://mmp.smenet.org/abstract.cfm?aid=2856'
print(url)
reponse = requests.get(url)
tree = etree.HTML(reponse.text)
info = parse_page(tree)

pub = generate(info, url, vol, no, year, venue)
pubs.append(pub)
save(pubs, data_file+f'{datasrc}_all.json')


# # 去除空Abstract

# In[171]:


pubs = open(data_file+f'{datasrc}_all.json').readlines()
pubs = eval(''.join(pubs))

no_abs = 'This page does not have an abstract.'
no_abs2 = 'Index to M&amp;MP Vol'
for it in iter(pubs):
    if no_abs in it['abstract'] or no_abs2 in it['abstract']:
        it.pop('abstract')
        print(it['url'][0])

save(pubs, data_file+f'{datasrc}_all_abs.json')

