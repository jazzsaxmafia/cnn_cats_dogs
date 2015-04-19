import pandas as pd
import os
from config import *

def print_to_html(f, x):
    image_full_path = "/home/taeksoo/Study/Machine_Learning/DeepLearning_oxford/images"
    print x
    f.write("<img src=\"" + os.path.join(image_full_path, x + '.jpg') + "\" width=\"200px\" height=\"200px\"/>")
    f.write("\n")

result_data = pd.read_pickle(os.path.join(data_path, 'result.pickle'))
class_0 = result_data[result_data['class'] == 0]
class_1 = result_data[result_data['class'] == 1]

with open(os.path.join(data_path, 'result.html'), "w") as f:
    f.write("""<html>
                <head>
                    <style>
                        div#content_left {
                            width:48%;
                            float:left;
                        }
                        div#content_right {
                            width:48%;
                            float:right;
                        }

                    </style>
                </head>
                <body>
            """)
    f.write("<div id=\"content_left\">")
    class_0['image'].map(lambda x: print_to_html(f, x))
    f.write("</div>")
    f.write("<div id=\"content_right\">")
    class_1['image'].map(lambda x: print_to_html(f, x))
    f.write("</div>")

    f.write("""
                </body>
            </html>
            """)

