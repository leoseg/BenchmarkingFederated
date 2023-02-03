
def get_time_logs(filepath:str,erase=False):
    file = open(filepath, 'r')
    lines = file.readlines()
    logs = {}
    for count,line in enumerate(lines):
        if line == "Client training time\n":
            logs["client_time"] =float(lines[count+1][:-1])
    if erase:
        open(filepath, 'w').close()
    return logs