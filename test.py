
class Logger:
    def __init__(self):
        self.output = ""

    def write(self, what):
        self.output += str(what) + '\n'

    def print(self):
        print(self.output)

def runTestsFor(dictinfo):
    for name in dictinfo:
        if name.startswith("test_") and callable(dictinfo[name]):
            logger = Logger();
            print(f"Running {name}...")
            try:
                success = dictinfo[name](logger)
                if not success:
                    print(f"\033[91m{name} FAILED\033[0m")
                    print(name + " log:")
                    logger.print()
            except Exception as e:
                print(f"\033[91m{name} FAILED ({e})\033[0m")
                print(name + " log:")
                logger.print()


import inquarting
runTestsFor(inquarting.__dict__)
