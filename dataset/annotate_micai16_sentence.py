import xml.etree.ElementTree as ET
import os

'''
reference 
'''
# INSTRUMENT_CLASSES = ('tissue', 'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver',
#     'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier', 'stapler', 'maryland_dissector','spatulated_monopolar_cautery')

# ACTION_CLASSES = (
#     'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 'Tool_Manipulation', 'Cutting', 'Cauterization'
#     , 'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')



# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    organ = []
    tools = []
    actions = []
    centers = []

    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['annotations'] = []
    c = 0

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
        
        # Get details of the bounding box 
        elif elem.tag == "objects":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["tool"] = subelem.text.lower()
                    if c == 0: 
                        organ.append(subelem.text.lower())
                        c +=1
                    else: tools.append(subelem.text.lower())
                elif subelem.tag == "interaction":
                    bbox["action"] = subelem.text.lower() 
                    if c != 1: 
                        actions.append(subelem.text.lower()) 
                    c +=1                  
                elif subelem.tag == "bndbox":
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)  
                        if subsubelem.tag == 'xmin': xmin = subsubelem.text
                        elif subsubelem.tag == 'ymin': ymin = subsubelem.text
                        elif subsubelem.tag == 'xmax': xmax = subsubelem.text
                        elif subsubelem.tag == 'ymax': ymax = subsubelem.text
                    
                    x_c = int(xmin) + ((int(xmax) - int(xmin))/2)
                    y_c = int(ymin) + ((int(ymax) - int(ymin))/2)
                    centers.append([x_c, y_c])
            info_dict['annotations'].append(bbox)
    
    return organ, tools, actions, centers, info_dict

def make_annotations(organ, tools, actions, centers):
    questions = ['what organ is being operated?', 'what tools are operating the organ?']
    answers = []
    ans = 'organ being operated is ' + organ[0]
    answers.append(ans)

    ans = 'the tools operating are '
    for i in range(len(tools)):
        if i != len(tools) - 1:
            ans += tools[i].replace("_", " ") + ' , '
        else:
            ans += tools[i].replace("_", " ")
    answers.append(ans)

    for i in range(len(tools)):
        q = 'what is the state of ' + tools[i].replace("_", " ") + '?'
        questions.append(q)
        ans = 'action done by ' + tools[i].replace("_", " ") + ' is ' + actions[i].replace("_", " ")
        answers.append(ans)

    for i in range(len(tools)):
        q = 'where is '+  tools[i].replace("_", " ") + ' located?'
        questions.append(q)
        x_loc = ''
        y_loc = ''
        if centers[0][0] <= centers[i+1][0]: x_loc  = 'right'
        else: x_loc = 'left'

        if centers[0][1] <= centers[i+1][1]: y_loc  = 'top'
        else: y_loc = 'bottom'

        a = tools[i].replace("_", " ") + ' is located at ' + x_loc + '-' + y_loc
        answers.append(a)
    
    return questions, answers


folder = r'/home/adithya/Desktop/Adi/Intern/NUS/VQA/Data/instruments18'

for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    ann_path = os.path.join(subfolder_path, 'complex1.2')
    # os.mkdir(ann_path)
    for subsubfolder in os.listdir(subfolder_path):
        if subsubfolder == 'xml':
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            for file in os.listdir(subsubfolder_path):
                file_path = os.path.join(subsubfolder_path, file)

                organ ,tools, actions, centers, info_dict = extract_info_from_xml(file_path)

                questions, answers = make_annotations(organ, tools, actions, centers)

                path = os.path.join(ann_path, file[:-4])
                f= open(path + "_QA.txt","w+")

                for i in range(len(questions)):
                    s = str(questions[i]) + '|' + str(answers[i]) + '\n'
                    f.write(s)
                f.close()