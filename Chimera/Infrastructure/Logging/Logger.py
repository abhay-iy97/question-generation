import logging

class Logger:
    def __init__(self):
        pass

    def Info(self, message):
        print('INFO: ' + message)

    def Warning(self, message):
        print('WARNING: ' + message)

    def Error(self, message):
        print('ERROR: ' + message)

class NativeLogger(Logger):
    def __init__(self):
        pass

    def Info(self, message):
        logging.info(message)

    def Warning(self, message):
        logging.warning(message)

    def Error(self, message):
        logging.error(message)

class PrintLogger(Logger):
    def __init__(self):
        pass

    def Info(self, message):
        print('INFO: ' + message)

    def Warning(self, message):
        print('WARNING: ' + message)

    def Error(self, message):
        print('ERROR: ' + message)