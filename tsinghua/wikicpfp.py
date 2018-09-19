# WikiCFP抓取
import requests
from lxml import etree
import time
import json
from operator import itemgetter, attrgetter

class thConference:
    """
    每一届具体会议的对象
    """
    def __init__(self, name=None, abbr=None,
                 link=None, official_link:'会议官网'=None,
                 year=None, when=None,
                 where=None,
                 submission_deadline=None,
                 notification_due=None,
                 final_version_due=None,
                 categories=None):
        self.name = name
        self.abbr = abbr
        self.link = link
        self.official_link = official_link
        self.year = year
        self.when = when
        self.where = where
        self.submission_deadline = submission_deadline
        self.notification_due = notification_due
        self.final_version_due = final_version_due
        self.categories = categories


class Conference:
    def __init__(self, name:'会议全程'=None, abbr:'会议别称'=None,
                 link:'链接地址'=None, main_link:'会议主网址'=None,
                 past=None, future=None, event:list()=None):
        self.name = name
        self.abbr = abbr
        self.link = link
        self.main_link = main_link
        self.past = past
        self.future = future
        self.event = event


domin = 'http://www.wikicfp.com/'
urls = ['http://www.wikicfp.com/cfp/series?t=c&i='+chr(i) for i in range(65,65+26)]


# 按字母表顺序抓取会议URL
confs = []
for url in urls:
    response = requests.get(url)
    page = etree.HTML(response.text)
    bgcolors = ['#f6f6f6', '#e6e6e6']
    for bgcolor in bgcolors:
        a_s = page.xpath('//tr[@bgcolor="'+bgcolor+'"]/td/a')
        tds = page.xpath('//tr[@bgcolor="'+bgcolor+'"]/td/text()')
        # a_s = [s.strip() for s in a_s if s != '\n']
        tds = [s.replace('-','').strip() for s in tds if s != '\n']
        for a,td in zip(a_s,tds):
            confs.append(Conference(name=td, abbr=a.text, link=domin+a.attrib['href']))

    # [print(c.abbr, c.name, c.link ,sep=':') for c in confs]
    print('已经抓取',len(confs),'会议URL')
    break

# 对每个会议进行详细抓取
for conf in confs[:5]:
    response = requests.get(conf.link)
    page = etree.HTML(response.text)

    past = page.xpath('//span[@class="theme"]/following-sibling::a/@href')
    if past and len(past)==1: conf.past = past[0]
    future = page.xpath('//following-sibling::span[@class="theme"]/following-sibling::span/a/text()')
    if future and len(future)==1: conf.future = int(future[0].split(' ')[0])

    print('抓取会议', conf.abbr, conf.link)
    # 2 抓取所有届会议信息
    events = []
    bgcolors = ['#f6f6f6', '#e6e6e6']
    for bgcolor in bgcolors:
        a_s = page.xpath('//tr[@bgcolor="' + bgcolor + '"]/td/a')
        for a in a_s:
            events.append(thConference(abbr=a.text, name=conf.name, year=int(a.text.strip().split()[-1]),link=domin+a.attrib['href']))

    # 2.1抓取届会议详细信息
    for event in events:
        print("抓取届信息",event.year)
        response = requests.get(event.link)
        page = etree.HTML(response.text)

        offilical_link = page.xpath('//td[contains(text(), "Link:")]/a/@href')
        if len(offilical_link)==1: event.official_link = offilical_link[0]

        when = page.xpath('//th[contains(text(),"When")]/following-sibling::td/text()')
        if when and len(when)==1 : event.when = when[0].strip()

        where = page.xpath('//th[contains(text(),"Where")]/following-sibling::td/text()')
        if where and len(where)==1: event.where = where[0].strip()

        submission_deadine = page.xpath(
            '//th[contains(text(),"Submission")]/following-sibling::td//span/span[@property="v:startDate"]'
            '/@content')
        if submission_deadine and len(submission_deadine)==1:
            event.submission_deadline = submission_deadine[0].strip().split('T')[0]

        notification_due = page.xpath(
            '//th[contains(text(),"Notification")]/following-sibling::td//span/span[@property="v:startDate"]'
            '/@content')
        if notification_due and len(notification_due)==1:
            event.notification_due = notification_due[0].strip().split('T')[0]

        final_version_due = page.xpath(
            '//th[contains(text(),"Final")]/following-sibling::td//span/span[@property="v:startDate"]'
            '/@content')
        if final_version_due and len(final_version_due)==1:
            event.final_version_due = final_version_due[0].strip().split('T')[0]

        categories = page.xpath('//a[@class="blackbold"]/following-sibling::a/text()')
        event.categories = categories


        # print(event.when, event.where, event.submission_deadline, event.notification_due, event.final_version_due, sep=" = ")
        # time.sleep(1)
        # break # 只抓一届会议

    # 将会议按年份排序
    events = sorted(events, key=attrgetter('year'), reverse=True)
    conf.event = events

    # 判断会议主网站是否存在
    offilical_links = [e.official_link.split('/')[2] for e in events]
    if offilical_links and len(set(offilical_links))==1:
        conf.main_link = 'http://'+offilical_links[0]

    # [print(e.year, e.abbr, e.name, e.link) for e in events]
    # break #只抓一个会议


# 将对象序列化 使用json
data = json.dumps(confs, default=lambda obj: obj.__dict__)
with open(r'D:\cfppp.json', 'w') as f:
    f.write(data)