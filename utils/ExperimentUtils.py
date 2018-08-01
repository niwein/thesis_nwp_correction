from optparse import OptionParser


# This functions are used to parse inputs to execute experiments

def parse_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

# parse with information for each programm input
def getOptionParser():
    parser = OptionParser()
    parser.add_option("--script", metavar="string", help="string to describe script to run")
    parser.add_option("--preprocessing", metavar="string", help="string to describe the type of preprocessing")
    parser.add_option("--experiment-path", metavar="PATH", help="path to the root node of programs file structure")
    parser.add_option("--input-source", metavar="PATH", help="path to preprocessed data folder")
    parser.add_option("--model-type", help="either network or baseline")
    return parser