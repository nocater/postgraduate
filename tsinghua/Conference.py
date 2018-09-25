class Conference:
    def __init__(self, id=None, name:'会议全程'=None, abbr:'会议别称'=None,series:'是否在Series中'=None,
                 link:'链接地址'=None, main_link:'会议主网址'=None,
                 past=None, future=None, event:list()=None):
        self.id = id
        self.name = name
        self.abbr = abbr
        self.series = series
        self.link = link
        self.main_link = main_link
        self.past = past
        self.future = future
        self.event = event