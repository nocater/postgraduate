class thConference:
    """
    每一届具体会议的对象
    """
    def __init__(self, name=None, abbr=None,
                 link=None, official_link:'会议官网'=None,
                 year=None,
                 stardate=None,
                 enddate=None,
                 submission_deadline=None,
                 notification_due=None,
                 final_version_due=None,
                 categories=None):
        self.name = name
        self.abbr = abbr
        self.link = link
        self.official_link = official_link
        self.year = year
        self.stardate = stardate
        self.enddate = enddate
        self.submission_deadline = submission_deadline
        self.notification_due = notification_due
        self.final_version_due = final_version_due
        self.categories = categories