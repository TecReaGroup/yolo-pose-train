"""Console and daily project logs in UTC+08:00."""
from datetime import datetime, timedelta, timezone
import logging
import sys
from types import TracebackType

from pen_pose.path import LOG_DIR

LOG_TIMEZONE = timezone(timedelta(hours=8))
LOG_FORMAT = '[%(asctime)s +08:00] [%(levelname)s] [%(name)s] - %(message)s'


class BeijingFormatter(logging.Formatter):
    """Render timestamps independently of the operating system timezone."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, LOG_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')


class DailyFileHandler(logging.Handler):
    """Route each record to its UTC+08:00 calendar date."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = datetime.fromtimestamp(record.created, LOG_TIMEZONE).strftime('%Y-%m-%d')
            with (LOG_DIR / f'log_{day}.log').open('a', encoding='utf-8') as stream:
                stream.write(self.format(record) + '\n')
        except Exception:
            self.handleError(record)


def log_uncaught_exception(
    exception_type: type[BaseException],
    exception: BaseException,
    traceback: TracebackType | None,
) -> None:
    """Persist uncaught application failures without swallowing interrupts."""
    if issubclass(exception_type, KeyboardInterrupt):
        sys.__excepthook__(exception_type, exception, traceback)
        return
    logging.getLogger('pen_pose').critical(
        'Unhandled exception', exc_info=(exception_type, exception, traceback),
    )


def configure_logging() -> None:
    """Initialize application logging once at each command entry point."""
    logger = logging.getLogger('pen_pose')
    if logger.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = BeijingFormatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    daily_file = DailyFileHandler()
    daily_file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(daily_file)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    sys.excepthook = log_uncaught_exception
