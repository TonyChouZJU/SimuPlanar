import os
import os.path as osp
import xml.etree.ElementTree as ET
from lxml import etree
from PIL import Image

def insert_object_node(root_node, xmin, ymin, xmax, ymax, label):
    object_node = ET.SubElement(root_node, 'object')
    name_node = ET.SubElement(object_node, 'name')
    name_node.text = label
    pose_node = ET.SubElement(object_node, 'pose')
    pose_node.text = 'Unspecified'
    truncated_node = ET.SubElement(object_node, 'truncated')
    truncated_node.text = '0'
    difficult_node = ET.SubElement(object_node, 'difficult')
    difficult_node.text = '0'

    bndbox_node = ET.SubElement(object_node, 'bndbox')
    xmin_node = ET.SubElement(bndbox_node, 'xmin')
    xmin_node.text = str(xmin)
    ymin_node = ET.SubElement(bndbox_node, 'ymin')
    ymin_node.text = str(ymin)
    xmax_node = ET.SubElement(bndbox_node, 'xmax')
    xmax_node.text = str(xmax)
    ymax_node = ET.SubElement(bndbox_node, 'ymax')
    ymax_node.text = str(ymax)


def save_xml(im_name, results, DETS_DIR):
    # create node
    root_node = ET.Element('annotation')

    folder_node = ET.SubElement(root_node, 'folder')
    folder_node.text = im_name
    filename_node = ET.SubElement(root_node, 'filename')
    filename_node.text = im_name.split('/')[-1].split('.')[0]
    path_node = ET.SubElement(root_node, 'path')
    path_node.text = im_name

    source_node = ET.SubElement(root_node, 'source')
    database_node = ET.SubElement(source_node, 'database')
    database_node.text = 'gesture'

    img = Image.open(path_node.text)
    imgSize = img.size

    size_node = ET.SubElement(root_node, 'size')
    width_node = ET.SubElement(size_node, 'width')
    width_node.text = str(imgSize[0])
    height_node = ET.SubElement(size_node, 'height')
    height_node.text = str(imgSize[1])
    depth_node = ET.SubElement(size_node, 'depth')
    depth_node.text = '3'

    segmented_node = ET.SubElement(root_node, 'segmented')
    segmented_node.text = '0'

    for key, bboxes in results.iteritems():
        for bbox in bboxes:
            [xmin, ymin, xmax, ymax] = [bbox[0], bbox[1], bbox[0] + bbox[2]-1, bbox[1] + bbox[3]-1]
            insert_object_node(root_node, xmin, ymin, xmax, ymax, key)

    jpg = os.path.split(im_name)[-1]
    xml_name = os.path.splitext(jpg)[0]
    write_xml = DETS_DIR + '/' + xml_name + '.xml'

    tree = ET.ElementTree(root_node)
    tree.write(write_xml, encoding='utf-8', xml_declaration=True)

    # lxml
    parser = etree.XMLParser()
    document = etree.parse(write_xml, parser)
    document.write(write_xml, pretty_print=True, encoding='utf-8')