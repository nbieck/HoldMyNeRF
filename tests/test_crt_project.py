from crt_project.hand_remove import image_reader, remove

if __name__ =='__main__':
    input_path = 'test_image/'
    output_path = 'output/contour/'

    reader = image_reader(input_path)
    remove(reader, output_path)
    