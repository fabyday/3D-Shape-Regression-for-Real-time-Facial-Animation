import visualizer as vis 
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--width")

parse_args = parser.parse_args()
input_dir_name = parse_args["input_dir"]
output_dir_name = parse_args["output_dir"]
width = int(parse_args["width"])




vis.resize_img(width=width)
