from datetime import datetime


class UtilsTime:
    @staticmethod
    def cdateTm():
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M")
        return date_time
