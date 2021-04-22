import xml.etree.cElementTree as ET
import numpy as np
import os

class Solve:
    def __init__(self, name, videoName, videoFrame, coeffs, morphs, region):
        self.name = name
        self.videoName = videoName
        self.videoFrame = videoFrame
        self.coeffs = coeffs
        self.morphs = morphs
        self.region = region

def makeSolveXML(xml_path):

    dataset = ET.Element("dataset")
    use = ET.SubElement(dataset, "use").text = "StrongTrack Solves"
    solves = ET.SubElement(dataset, "solves")
    
    tree = ET.ElementTree(dataset)
    
    tree.write(xml_path)

def classFromXML(table):
    tree = ET.parse(table)
    root = tree.getroot()
    solves = root.find('solves')

    classData = []
    for solve in solves:
        name = solve[0].text
        videoName = solve[1].text
        videoFrame = int(solve[2].text)
        coeffs = []
        for coeff in solve[3]:
            coeffs.append(float(coeff.attrib.get('val')))
        morphs = []
        for morph in solve[4]:
            morphs.append(float(morph.attrib.get('val')))

        entry = Solve(name, videoName, videoFrame, coeffs, morphs)
        classData.append(entry)

    return classData

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

def appendXMLFromClass(xml_path, entry):

    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    base = root.find('solves')
    solve = ET.SubElement(base, "solve")
    name = ET.SubElement(solve, "name").text = entry.name
    videoName = ET.SubElement(solve, "videoName").text = entry.videoName
    videoFrame = ET.SubElement(solve, "videoFrame").text = str(entry.videoFrame)
    coeffs = ET.SubElement(solve, "coeffs")
    morphs = ET.SubElement(solve, "morphs")
    region = ET.SubElement(solve, "region").text = entry.region

    for i in range(len(entry.coeffs)):
        
        ET.SubElement(coeffs, "coeff", index = str(i), val = str(entry.coeffs[i]))

    for j in range(len(entry.morphs)):

        ET.SubElement(morphs, "morph", index = str(j), val = str(entry.morphs[j]))
    
    use = root.find('use')
    if use.text == 'StrongTrack Solves':
        tree.write(xml_path)
    else:
        print('XML not a solve file. Not Saving')

def classFromXML(table):
    tree = ET.parse(table)
    root = tree.getroot()
    solves = root.find('solves')

    classData = []
    for solve in solves:
        name = solve[0].text
        videoName = solve[1].text
        videoFrame = int(solve[2].text)
        coeffs = []
        for coeff in solve[3]:
            coeffs.append(float(coeff.attrib.get('val')))
        morphs = []
        for morph in solve[4]:
            morphs.append(float(morph.attrib.get('val')))

        entry = Solve(name, videoName, videoFrame, coeffs, morphs)
        classData.append(entry)

    return classData

def classFromXMLTemp(table):
    tree = ET.parse(table)
    root = tree.getroot()
    solves = root.find('solves')

    classData = []
    for solve in solves:
        name = solve[0].text
        videoName = solve[1].text
        videoFrame = int(solve[2].text)
        coeffs = []
        for coeff in solve[3]:
            pointduo = (coeff.attrib.get('val')).split(',')
            pointduo = (float(pointduo[0][1:]), float(pointduo[1][:-1]))
            coeffs.append(pointduo)

        morphs = []
        for morph in solve[4]:
            morphs.append(float(morph.attrib.get('val')))

        region = solve[5].text
        entry = Solve(name, videoName, videoFrame, coeffs, morphs, region)
        classData.append(entry)

    return classData

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
    exts = ['.mp4','.avi', '.mov']
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
