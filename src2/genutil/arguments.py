def parse_from_file(filename, parser):
    with open(filename,"r") as f:
        string=f.read()
        args=parser.parse_args(string.split())
    return args
