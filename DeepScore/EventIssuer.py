########################################
########################################
####### Author : Abhinandan Dubey (alivcor)
####### Stony Brook University

from logwriter import *
import sys

class bcolors:
    PINK = '\033[95m'
    LIME = '\033[0;32m'
    YELLOW = '\033[93m'
    VIOLET = '\033[0;35m'
    BROWN = '\033[0;33m'
    INDIGO = "\033[0;34m"
    BLUE = "\033[0;34m"
    LIGHTPURPLE = '\033[1;35m'
    LIGHTRED = '\033[1;31m'
    NORMAL = '\033[0;37m'
    SHARP = '\033[1;30m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    SLEEP = '\033[90m'
    UNDERLINE = '\033[4m'


    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.END = ''


def issueMessage(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.NORMAL}{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.NORMAL}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)


def issueSleep(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.SLEEP}{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.SLEEP}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)

def issueSharpAlert(text, logfilename, highlight=False):
    if highlight:
        toprint = "{.BOLD}(DeepScore Engine)" + " : " + text + "{.END}"
        print toprint.format(bcolors, bcolors)
    else:
        toprint = "{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)


def issueError(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.FAIL}{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.FAIL}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)

def issueWelcome(logfilename):
    print "{.BLUE}{.BOLD} __         __          {.END}".format(bcolors, bcolors, bcolors)
    print "{.BLUE}{.BOLD}|  \ _ _ _ (_  _ _  _ _ {.END}".format(bcolors, bcolors, bcolors)
    print "{.BLUE}{.BOLD}|__/(-(-|_)__)(_(_)| (- {.END}".format(bcolors, bcolors, bcolors)
    print "{.BLUE}{.BOLD}        |               {.END}".format(bcolors, bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write(" __         __          \n|  \ _ _ _ (_  _ _  _ _ \n|__/(-(-|_)__)(_(_)| (- \n        |               ")
    print "\n\n"
    with open(logfilename, 'a') as f:
        f.write("\n\n(DeepScore Engine) : Welcome to DeepScore v0.1")
    toprint = "{.BLUE}{.BOLD}(DeepScore Engine){.END}" + " : " + "Welcome to DeepScore v0.1"
    print toprint.format(bcolors, bcolors, bcolors)

def issueSuccess(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.LIME}{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.LIME}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)

def genLogFile(logfilename, ts, strts):
    toprint = "{.LIME}{.BOLD}(DeepScore Engine){.END}" + " : " + "Logging all events to " + str(ts)
    print toprint.format(bcolors, bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + "Log File Created at : " + str(strts))
        f.write("\n(DeepScore Engine) : " + "Logging all events to " + str(ts))


def issueWarning(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.BROWN}{.BOLD}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors, bcolors)
    else:
        toprint = "{.BROWN}(DeepScore Engine){.END}" + " : " + text
        print toprint.format(bcolors, bcolors)
    with open(logfilename, 'a') as f:
        f.write("\n(DeepScore Engine) : " + text)

# issueWelcome()
# issueMessage("I'm glad you're here !")
# issueSleep("I'm turning to sleep mode.")
# issueSharpAlert("I'm back up. You need to look into this.")
# issueWarning("There's some problem with my core")
# issueError("I have to shutdown. Please mail the administrator the log file I've generated.")
#

def issueExit(logfilename, ts):
    toprint = "{.LIGHTPURPLE}{.BOLD}(DeepScore Engine){.END}" + " : Shutting down the engine. Logs have been saved. Have a good day !"
    print toprint.format(bcolors, bcolors, bcolors)
    genpdfcmd = "python logwriter.py " + logfilename + " -S \"LOG FILE\" -A \"DeepScore Engine\" -o logs/DeepScore_Log_" + str(ts) + ".pdf"
    os.system(genpdfcmd)