from datetime import datetime


def get_current_datetime() -> str:
    return datetime.now().strftime("%d %B %Y %H:%M")
