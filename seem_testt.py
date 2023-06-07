import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "dependencies/SEEM/demo_code"))
os.chdir(os.path.join(os.getcwd(), 'dependencies/SEEM/demo_code'))
from seem_extraction import SEEMPipeline

if __name__ == "__main__":
    SEEMPipeline("test_image", "output_img", "cube")
