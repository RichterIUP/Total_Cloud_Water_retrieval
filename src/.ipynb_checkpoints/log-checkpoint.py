#!/usr/bin/python3
'''@package docstring
Write results to logfiles
'''

import sys
import inp
import os
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import aux2 as aux
import datetime as dt

def log_prog_start():
    '''Initialise logfile
    '''
    with open("{}/retrieval_log.dat".format(inp.PATH), "a") as file_:
        file_.write("\n\n#########################################\n")
        file_.write("# TCWret\n")
        file_.write("#\n")
        file_.write("# Started: {}\n".format(dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%dT%H:%M:%S")))
        file_.write("# Spec: {}\n".format(dt.datetime.strftime(aux.DATETIME, "%Y-%m-%dT%H:%M:%S")))
        for i in range(len(aux.MICROWINDOWS)):
            file_.write("# Microwindow: {}\n".format(aux.MICROWINDOWS[i]))
        for element in aux.CLOUD_GRID:
            file_.write("# Cloud layer: {}\n".format(element))
        file_.write("# Cloud Temperature: {}\n".format(aux.CLOUD_TEMP))
        file_.write("#########################################\n\n")
    
    return

def write(text):
    '''Write arb. text to retrieval_log.dat
    
    @param text Text to be written to retrieval_log.dat
    '''
    with open("{}/retrieval_log.dat".format(inp.PATH), "a") as file_:
        file_.write("{}\n".format(text))

    return
