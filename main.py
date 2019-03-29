import sys
import argparse
from nn_transfer import run_style_transfer, show_results, load_img
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply style transfer')
    parser.add_argument('style_path',help='filename of the style example image')
    parser.add_argument('content_path',help='filename of the content image')
    parser.add_argument('output_path',help='filename of the output image')
    args = parser.parse_args(sys.argv[1:])
    content_path = args.content_path
    style_path = args.style_path
    output_path = args.output_path
    best, best_loss = run_style_transfer(content_path, 
            style_path, num_iterations=1000)
    show_results(best, content_path, style_path) 
    best.save(output_path)

