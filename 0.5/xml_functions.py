import xml.etree.cElementTree as ET
import numpy as np
import os

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
    use = ET.SubElement(dataset, "use").text = "StrongTrack Face Tracking"
    project = ET.SubElement(dataset, "project").text = "Project Name"
    images = ET.SubElement(dataset, "images")

    tree = ET.ElementTree(dataset)
    
    # Check if file already exists at path.
    if os.path.exists(xml_path):
        print('Error: File already at write path')
    else:
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

    #Check whether file associated with strongtrack
    check = verifyXML(xml_path)

    if check == True:
        tree.write(xml_path)
    else:
        print('Error: XML path not associated with StrongTrack')

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

def verifyXML(table):
    data = ET.parse(table)
    root = data.getroot()
    
    try:
        use = root.find('use').text
        target = 'StrongTrack Face Tracking'
    
        if use == target:
            check = True
        else:
            check = False
    except:
        check = False
        
    return check

def verifyVideo(path):
    extension = os.path.splitext(os.path.split(path)[1])[1]
    exts = ['.mp4','.avi']
    if extension in exts:
        check = True
    else:
        check = False
    return check

def getProjectName(table):
    data = ET.parse(table)
    root = data.getroot()
    project = root.find('project').text
    return project
