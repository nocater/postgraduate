# coding: utf-8

# In[ ]:


import requests
from lxml import etree
import time
import datetime

now = datetime.datetime.now

# In[ ]:


TIME = 2


# In[ ]:


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


# In[ ]:


def generate(infos, year, venue, src='kovove', issn='0023-432X'):
    def foo_title(x):
        x = x.replace('<sub>', '')
        x = x.replace('</sub>', '')
        return x

    def foo_author(author):
        a_list = []
        for i, name in enumerate(author.split(',')):
            a = dict()
            a['name'] = name.strip()
            a['pos'] = i
            a['sid'] = hash(name)
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
             str(now()), src, 'en', hash(title), year,
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


# In[ ]:


# Save
def save(data):
    if os.path.exists('./kovmat.json'):
        with open('./kovmat.json', 'a') as f:
            f.write(repr(data))
    else:
        with open('./kovmat.json', 'w') as f:
            f.write(repr(data))
    print('Saved!')


# In[ ]:


if __name__ == '__main__':
    all_pubs = []
    for rr in range(33, 59):
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
                print(f'found {records} records')

                inofs = parse(tree, rr, cc)
                pubs = generate(inofs, url, year, venue)
                print(f'spider success! {len(pubs)}')
                save(pubs, f'./kovmat_{year}.json')
                all_pubs.extend(pubs)
            else:
                print('ERROR! ', response.status_code)
    #         break
    #     break
save(all_pubs, f'./kovmat_all.json')