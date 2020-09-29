from xml.etree.ElementTree import Element, SubElement, ElementTree
import os
import cv2

folder_name = 'ky_synth'

if not os.path.exists(os.path.join(folder_name, 'Annotation')):
    os.makedirs(os.path.join(folder_name, 'Annotation'))

synth_img = os.listdir(folder_name)

imgs = [img for img in synth_img if img.endswith('.jpg')]

# for img in imgs:
#     img_ = cv2.imread(os.path.join('synth_img', img))
#     print(img_.shape)

# xml 파일을 읽기 좋게 만들어주는 함수
def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


for img_ in imgs:

    filename = img_

    label = img_.split('_')[0]
    img_ = cv2.imread(os.path.join(folder_name, filename))

    root = Element('annotation')
    SubElement(root, 'folder').text = folder_name
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = os.path.join(folder_name, filename)
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(img_.shape[1])
    SubElement(size, 'height').text = str(img_.shape[0])
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    obj = SubElement(root, 'object')
    SubElement(obj, 'name').text = label
    SubElement(obj, 'pose').text = 'Unspecified'
    SubElement(obj, 'truncated').text = '0'
    SubElement(obj, 'difficult').text = '0'
    bbox = SubElement(obj, 'bndbox')
    SubElement(bbox, 'xmin').text = str(0)
    SubElement(bbox, 'ymin').text = str(0)
    SubElement(bbox, 'xmax').text = str(img_.shape[1])
    SubElement(bbox, 'ymax').text = str(img_.shape[0])

    indent(root)
    tree = ElementTree(root)

    filename_ = os.path.splitext(filename)[0]
    tree.write(os.path.join(folder_name,'Annotation', filename_ + '.xml'))
    # tree.write('./' + filename + '.xml')