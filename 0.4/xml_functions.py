import xml.etree.cElementTree as ET
import numpy as np

def setNeutral(xml_path, points):
    neutral = convertXMLPoints(xml_path)[0]
   
    width_points = points[16][0]-points[0][0]
    width_neutral = neutral[16][0]-neutral[0][0]
    width_fac = width_neutral/width_points
    
    neutral = np.divide(neutral, [width_fac,width_fac]).astype(int)
    
    for i in range(48,68):
        points[i] = neutral[i]
        
    return points

def makeXML(xml_path):

    dataset = ET.Element("dataset")
    name = ET.SubElement(dataset, "name").text = "Training faces"
    comment = ET.SubElement(dataset, "comment").text = "XML data for use with DLib shape trainer."
    images = ET.SubElement(dataset, "images")

    tree = ET.ElementTree(dataset)
    tree.write(xml_path)

def appendXML(points, box, file_path, xml_path):

    try:
        tree = ET.parse(xml_path)

    except:
        'no xml found. Creating new one.'
        makeXML(xml_path)
        tree = ET.parse(xml_path)
        
    root = tree.getroot()
    images = root.find('images')
    
    image = ET.SubElement(images, "image", file=file_path)

    top = box.top()
    left = box.left()
    height = box.bottom()-box.top()
    width = box.right()-box.left()
    
    box_frame = ET.SubElement(image, "box", top="%d" % top, left="%d" % left, width="%d" % width, height="%d" % height)

    for element in range(len(points)):
            x=points[element][0]
            y=points[element][1]
            ET.SubElement(box_frame, "part", name="%02d" % element,x = "%d" % x, y="%d" % y)

    tree.write(xml_path)

def updateXML(images):

    for image in images:
        if image.attrib.get('file') == filename:
            image[0][0].set('x', str(99))
            
            for i in range(68):
                image[0][i].set('x', str(points[i][0]))
                image[0][i].set('y', str(points[i][1]))

    tree.write(xml_path)
    print('written')

def getXMLPoints(xml_image):
    write_points = []
    for point in xml_image:
        x = int(point.attrib.get('x'))
        y = int(point.attrib.get('y'))
        write_points.append((x,y))
    write_points = np.array(write_points)

    return write_points

def convertXMLPoints(table):
    tree = ET.parse(table)
    root = tree.getroot()
    images = root.find('images')

    write_images = []

    for image in images:
        box=image[0]
        points = getXMLPoints(box)
        write_images.append(points)

    write_images=np.array(write_images)

    return write_images
