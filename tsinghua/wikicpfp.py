# WikiCFP抓取
import requests
from lxml import etree
import time
import json
from operator import itemgetter, attrgetter
from datetime import datetime
from tsinghua.Conference import Conference
from tsinghua.thConference import thConference
import re


def parseConference(conf: Conference):

    response = requests.get(conf.link)

    # 1 抓取 past future
    page = etree.HTML(response.text)
    past = page.xpath('//span[@class="theme"]/following-sibling::a/@href')
    if past and len(past) == 1: conf.past = past[0]
    future = page.xpath('//following-sibling::span[@class="theme"]/following-sibling::span/a/text()')
    if future and len(future) > 0: conf.future = int(future[0].split(' ')[0])

    print('抓取会议', conf.abbr, conf.link)

    # 2 抓取所有届会议信息
    events = []
    bgcolors = ['#f6f6f6', '#e6e6e6']
    for bgcolor in bgcolors:
        a_s = page.xpath('//tr[@bgcolor="' + bgcolor + '"]/td/a')
        for a in a_s:
            year = a.text.strip().split()[-1]
            year = re.findall('\d{1,4}', year)[0]
            events.append(thConference(thabbr=a.text, thname=conf.name, year=int(year),
                                       link=domin + a.attrib['href']))

    # 2.1抓取届会议详细信息
    for event in events:
        # print('解析届会议', event.year)
        event, series = parsethConf(event)
        if series: conf.series = series

        # print(event.stardate, event.submission_deadline, event.notification_due, event.final_version_due, sep=" = ")
        # time.sleep(1)
        # break # 只抓一届会议

    print('抓取届会议信息完成', len(events))
    # 将会议按年份排序
    events = sorted(events, key=attrgetter('year'), reverse=True)
    conf.event = events

    # 判断会议主网站是否存在
    offilical_links = [e.official_link.split('/')[2] for e in events if e.official_link]
    if offilical_links and len(set(offilical_links)) == 1:
        conf.main_link = 'http://' + offilical_links[0]

    return conf


def parsethConf(event: thConference=None):
    SERIES = None
    response = requests.get(event.link)
    page = etree.HTML(response.text)

    thname = page.xpath('//span[@property="v:description"]/text()')
    if thname and len(thname) == 1: event.thname = thname[0].strip()

    series = page.xpath('//a[contains(@href,"/cfp/program?id=")]/@href')
    if series : SERIES = series[0].split('id=')[1].split('&')[0]

    offilical_link = page.xpath('//td[contains(text(), "Link:")]/a/@href')
    if len(offilical_link) == 1: event.official_link = offilical_link[0]

    when = page.xpath('//th[contains(text(),"When")]/following-sibling::td/text()')
    if when and len(when) == 1 and when[0].strip() != 'N/A':
        when = when[0].strip()
        stardate, enddate = [datetime.strptime(i.strip(), '%b %d, %Y') for i in when.split('-')]
        event.stardate = datetime.strftime(stardate, '%Y-%m-%d')
        event.enddate = datetime.strftime(enddate, '%Y-%m-%d')

    where = page.xpath('//th[contains(text(),"Where")]/following-sibling::td/text()')
    if where and len(where) == 1 and where[0].strip() != 'N/A': event.where = where[0].strip()

    submission_deadine = page.xpath(
        '//th[contains(text(),"Submission")]/following-sibling::td//span/span[@property="v:startDate"]'
        '/@content')
    if submission_deadine and len(submission_deadine) == 1 and submission_deadine[0].strip() != 'N/A':
        event.submission_deadline = submission_deadine[0].strip().split('T')[0]

    notification_due = page.xpath(
        '//th[contains(text(),"Notification")]/following-sibling::td//span/span[@property="v:startDate"]'
        '/@content')
    if notification_due and len(notification_due) == 1 and notification_due[0].strip() != 'N/A':
        event.notification_due = notification_due[0].strip().split('T')[0]

    final_version_due = page.xpath(
        '//th[contains(text(),"Final")]/following-sibling::td//span/span[@property="v:startDate"]'
        '/@content')
    if final_version_due and len(final_version_due) == 1 and final_version_due[0].strip() != 'N/A':
        event.final_version_due = final_version_due[0].strip().split('T')[0]

    categories = page.xpath('//a[@class="blackbold"]/following-sibling::a/text()')
    event.categories = categories
    return event, SERIES


def getAllAbbts():
    """
    获取所有的更新页面会议简称
    :return:
    """
    urls = [r'http://www.wikicfp.com/cfp/allcfp?page=' + str(i) for i in range(1, 11)]
    domin = r'http://www.wikicfp.com'
    thabbrs = []

    for url in urls:
        response = requests.get(url)
        page = etree.HTML(response.text)

        bgcolors = ['#f6f6f6', '#e6e6e6']
        for bgcolor in bgcolors:
            temp = page.xpath('//tr[@bgcolor="' + bgcolor + '"]/td/a/text()')
            thabbrs.extend(temp)
    print('Updated all thConference abbrs:', len(thabbrs))
    return thabbrs

if __name__ == '__main__':
    domin = 'http://www.wikicfp.com/'
    urls = ['http://www.wikicfp.com/cfp/series?t=c&i='+chr(i) for i in range(65,65+26)]

    # 按字母表顺序抓取会议URL
    confs = []
    for url in urls[:1]:
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
        print('已经抓取', len(confs), '会议URL')
        # break

    # 对每个会议进行详细抓取
    for conf in confs[:5]:
        parseConference(conf)
        # [print(e.year, e.abbr, e.name, e.link) for e in events]
        # break #只抓一个会议

    # 将对象序列化 使用json
    data = json.dumps(confs, default=lambda obj: obj.__dict__)
    with open(r'D:\Documents\清华数据任务\data\cfp.json', 'w') as f:
        f.write(data)