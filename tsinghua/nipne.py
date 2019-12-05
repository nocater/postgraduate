
# coding: utf-8

# In[ ]:


import requests
from lxml import etree
import time
import datetime
import os
import json
now = datetime.datetime.now


# In[ ]:


TIME = 2
base_url = 'http://www.nipne.ro/rjp/'
datasrc = 'nipne'
venue_name = 'Romanian Journal of Physics'
issn = '1221-146X'
data_file='./data/nipne/'


# In[ ]:


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


# In[ ]:


def parse(tree, rr, cc):
    titles = tree.xpath('//span[@class="toct"]')
    titles = [''.join(i.xpath('text()')).strip() for i in titles]
    raw_titles = [etree.tostring(e).decode('utf8') for e in tree.xpath('//span[@class="toct"]')]
    
    authors = tree.xpath('//span[@class="toca"]/text()')
    authors = [i.strip() for i in authors]
    
    pdfs_ = tree.xpath('//a[contains(text(),"Full text")]/@href')
    if pdfs_:
        pdfs = [base_url+i for i in pdfs_]
    else:
        pdfs_ = tree.xpath('//a[contains(text(),"PDF")]/@href')
        pdfs = [base_url+i for i in pdfs_]
    
    
    pages = tree.xpath('//span[@class="times"]/text()')
    if len(pages)>1:
        pages_start = pages
        pages_end = [str(int(i.split('/')[1].split('_')[1][:-4])) for i in pdfs_]
        pages = [pages_start[i]+'-'+pages_end[i] for i in range(len(pages))]
    else:
        pages = tree.xpath('//span[@style="font-size:8pt;"]')
        pages = [''.join(i.xpath('text()')).strip() for i in pages]
        pages = [i.split()[5] for i in pages]
        if '-' in pages[0]:
            pages_start = [i.split('-')[0] for i in pages]
            pages_end = [i.split('-')[1] for i in pages]
        else:
            pages_start = pages
            pages_end = [None] * len(pages)
    
#     print(len(titles),len(raw_titles),len(pages),len(pages_end),len(pdfs))
    assert len(titles) == len(raw_titles) == len(pages) == len(pages_end) == len(pdfs)
    return raw_titles,titles,authors,pages_start,pages_end,pages,pdfs


# response = requests.get('http://www.nipne.ro/rjp/2014_59_1-2.html')
# tree = etree.HTML(response.text)
# parse(tree, 59, '1-2')


# In[ ]:


def generate(infos, url, year, vol, issue, venue, src=datasrc, issn=issn):
    
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
    for raw_title,title,author,page_start,page_end,page,pdf in zip(*infos):
        pub = dict()
        pub.update(zip(['url','raw_title','title','authors','venue','page_start','page_str','volume','issue','pdf_src','hash','ts','src','lang','sid','year','issn'],                           [[url], raw_title, title, foo_author(author), venue, page_start, page, vol, issue, pdf,                            foo_hash(title),str(now()), src, 'en', str(hash(title)), year,
                           issn]))
        
        if page_end:
            pub['page_end'] = page_end

#         print(repr(pub))
#         print('--------------')
#         [print(f'{k}:{v}') for k,v in pub.items()]
        pubs.append(pub)
    
    return pubs

# r = parse(tree, 59, '1-2')
# venue = {'name':venue_name, 'type':1, 'sid':'1'}
# r = generate(r, 'http://www.nipne.ro/rjp/2014_59_1-2.html', 2019, '59', '1-2', venue)


# In[ ]:


if __name__ == '__main__':
    all_pubs = []
    for rr in range(59, 65):
        vol_pubs = []
        for cc in range(1, 10, 2):
            
            # end
            if rr == 64 and cc == 9:break
            
            year = 2014 - 59 + rr
            print(f'parsing {rr}-{cc}-{year}')
            
            url = f'http://www.nipne.ro/rjp/{year}_{rr}_{cc}-{cc+1}.html'
            
            venue = {'name':venue_name, 'type':1, 'sid':'1'}

            response = requests.get(url)
            time.sleep(TIME)

            if response.ok:
                tree = etree.HTML(response.text)
                inofs = parse(tree, rr, cc)
                pubs = generate(inofs, url, year, str(rr), f"{cc}-{cc+1}", venue)
                print(f'spider success:\t{len(pubs)}')
                vol_pubs.extend(pubs)
            else:
                print('ERROR! ', response.status_code)
            #break#end for issue
        #break#end for vol
        # each volume save
        save(vol_pubs,  f'{data_file}{datasrc}_{year}.json')
        all_pubs.extend(vol_pubs)

print('All spidered!')
save(all_pubs,  f'{data_file}{datasrc}_all.json')

