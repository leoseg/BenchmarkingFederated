
def get_time_logs(filepath:str,erase=False):
    file = open(filepath, 'r')
    lines = file.readlines()
    logs = {}
    for count,line in enumerate(lines):
        if line == "Client training time":
            logs["client_time"] = lines[count+1]
    if erase:
        open(filepath, 'w').close()
    return logs