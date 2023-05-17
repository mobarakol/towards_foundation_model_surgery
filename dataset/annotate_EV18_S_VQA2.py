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
            print(elem.text)
        
        # Get details of the bounding box 
        elif elem.tag == "objects":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["tool"] = subelem.text.replace("_", " ")
                    if  c == 0: 
                        organ.append(subelem.text.replace("_", " "))
                        c +=1
                    else: tools.append(subelem.text.replace("_", " "))
                elif subelem.tag == "interaction":
                    bbox["action"] = subelem.text.replace("_", " ")
                    if c != 1: 
                        actions.append(subelem.text.replace("_", " ")) 
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
    questions = ['Tissue|Which organ is being operated?&What is the organ of interest?&What is the name of the organ?']
    answers = []
    ans = 'The organ being operated is '+ organ[0] + '.&' + organ[0] + ' is the organ of interest.' + '&The organ name is ' + organ[0] + '.'
    answers.append(ans)

    q = 'Tools|What tools are present in the scene?&What instruments are currently used for operating the organ?'
    questions.append(q)
    ans = 'The tools present in the scene are '
    for i in range(len(tools)):
        if i == 0:
            ans += tools[i]
        elif i != len(tools) - 1:
            ans += ', ' + tools[i]
        else:
            ans += ' and ' + tools[i] + '.'
    ans += '&The instruments used for operating the organs are '
    for i in range(len(tools)):
        if i == 0:
            ans += tools[i]
        elif i != len(tools) - 1:
            ans += ', ' + tools[i]
        else:
            ans += ' and ' + tools[i] + '.'
    answers.append(ans)

    for i in range(len(tools)):
        q = 'Action|What is the state of ' + tools[i] + '?&'+ 'What is ' + tools[i] + ' doing?&' + 'What is the action of ' + tools[i] + '?'
        questions.append(q)
        ans = 'The state of ' + tools[i] + ' is ' + actions[i] + '.&The ' + tools[i] + ' is ' + actions[i] + '.&The action of the ' + tools[i] + ' is '+ actions[i] + '.'
        answers.append(ans)

    for i in range(len(tools)):
        q = 'Location|Where is '+  tools[i] + ' located?&' + 'What is the location of '+  tools[i] + '?&' 'Where is '+  tools[i] + '?'
        questions.append(q)
        x_loc = ''
        y_loc = ''
        if centers[i+1][0] >= 640: x_loc  = 'right'
        else: x_loc = 'left'

        if centers[i+1][1] < 512 : y_loc  = 'top'
        else: y_loc = 'bottom'

        a = x_loc + '-' + y_loc

        ans = 'The ' + tools[i] + ' is located at ' + a + '.&The location of ' + tools[i] + ' is ' + a + '.&The ' + tools[i] + 'is at ' + a + '.'
        answers.append(ans)
    
    return questions, answers


folder = 'EndoVis-18-VQA/'
subfolders = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16]
# subfolders = [1]

for subfolder in subfolders:

    subfolder_path = os.path.join(folder, 'seq_'+str(subfolder))
    ann_path = os.path.join(subfolder_path, 'vqa2')
    # os.mkdir(ann_path)
    ann_path = os.path.join(subfolder_path, 'vqa2/Sentence')
    os.mkdir(ann_path)
    for subsubfolder in os.listdir(subfolder_path):
        if subsubfolder == 'xml':
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            for file in os.listdir(subsubfolder_path):
                file_path = os.path.join(subsubfolder_path, file)

                organ ,tools, actions, centers, info_dict = extract_info_from_xml(file_path)
                # print(organ, tools, actions, centers)
                questions, answers = make_annotations(organ, tools, actions, centers)
                # print(questions, answers)
                path = os.path.join(ann_path, file[:-4])
                f= open(path + "_QA.txt","w+")

                for i in range(len(questions)):
                    s = str(questions[i]) + '|' + str(answers[i]) + '\n'
                    # print(s)
                    f.write(s)
                f.close()