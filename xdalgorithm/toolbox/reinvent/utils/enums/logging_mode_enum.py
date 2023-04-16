class LoggingModeEnum():
    _LOCAL = "local"
    _REMOTE = "remote"
    _NEPTUNE = "neptune"

    @property
    def LOCAL(self):
        return self._LOCAL

    @LOCAL.setter
    def LOCAL(self, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field")

    @property
    def REMOTE(self):
        return self._REMOTE

    @REMOTE.setter
    def REMOTE(self, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field")

    @property
    def NEPTUNE(self):
        return self._NEPTUNE

    @NEPTUNE.setter
    def NEPTUNE(self, value):
        raise ValueError("Do not assign value to a LoggingModeEnum field")