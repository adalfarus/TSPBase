# ------------------------------------------------------
# TimidTimer, a timer that can easily be set up and used,
# but is very accurate at the same time.
# ------------------------------------------------------
from datetime import timedelta
from typing import Optional
from timeit import default_timer


class TimidTimer:
    """Uses the timeit.default_timer function to calculate passed time."""
    def __init__(self, start_seconds: Optional[int] = 0,
                 start_now: Optional[bool] = True):
        self.starter = start_seconds
        self.ender = None
        self.timedelta = None

        if start_now:
            self.start()

    def start(self) -> float:
        self.starter = default_timer() + self.starter
        return self.starter

    def end(self, return_end_time: Optional[bool] = False):
        self.ender = default_timer()
        self.timedelta = timedelta(seconds=self.ender - self.starter)

        if not return_end_time:
            return self.timedelta
        else:
            return self.ender

    def tick(self, return_time: Optional[bool] = False):
        """Return how much time has passed till the start."""
        if not return_time:
            return timedelta(seconds=default_timer() - self.starter)
        else:
            return default_timer()

    def tock(self, return_time: Optional[bool] = False):
        """Returns how much time has passed till the last tock."""
        last_time = self.ender or self.starter
        self.ender = default_timer()

        if not return_time:
            return timedelta(seconds=self.ender - last_time)
        else:
            return last_time

    @staticmethod
    def test_delay() -> timedelta:
        return TimidTimer().end()

    @staticmethod
    def test_tock_delay() -> timedelta:
        timer = TimidTimer()
        timer.tock()
        return timer.end()

    @staticmethod
    def time() -> float:
        return default_timer()


if __name__ == "__main__":
    print(TimidTimer.test_tock_delay())
