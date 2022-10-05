import pyAgrum as gum
import pyAgrum.lib.image as gumImage
import heart_disease_table


nodes=['Smokes', 'Hypertension', 'Heart_Disease', 'Cholesterol', 'Alcohol_Abuse', 'Physical_Activity', 'MI', 'ST_Segment_Elevation', 'Chest_Pain']

links = [('Smokes','Hypertension'),('Hypertension','Heart_Disease'),('Smokes','Heart_Disease'),
('Cholesterol','Heart_Disease'),('Alcohol_Abuse','Hypertension'),('Alcohol_Abuse','Heart_Disease'),('Physical_Activity','Heart_Disease'),
('Heart_Disease','MI'),('MI','ST_Segment_Elevation'),('Chest_Pain','MI')]


def build_model():

    global links
    #Topology:
    ##########
    
    bn=gum.BayesNet('MI_bayes')
    for node in nodes:
        bn.add(node,2)
    
    
    for link in links:
        bn.addArc(*link)

    bn = addProbabilities(bn)

    gum.saveBN(bn,"./output/MI_bayes.bif")
    # with open("MI_bayes.bif","r") as out:
    #     print(out.read())

    gumImage.export(bn,"./output/MI_bayes.png")

    return bn

def build_model_decision():

    global links
    dn=gum.InfluenceDiagram()
    for node in nodes:
        dn.addChanceNode(node,2)
    
    dn.addDecisionNode("Call_Ambulance",2)
    
    dn.addUtilityNode(gum.LabelizedVariable("Utility","Utility",1))
   
    links+=[("Call_Ambulance", "Utility"),("MI","Utility")]
    
    for link in links:
        
        dn.addArc(*link)
    
    dn = addProbabilities(dn)
    
    #Utility Node:
    ##############
    #NOT MI <=> Not an emergency (bad)
    
    # 0 0 0
    # 0 0 1
    # 0 1 0
    # 0 1 1
    # 1 0 0
    # 1 0 1
    # 1 1 0
    # 1 1 1

    dn.utility('Utility')[{'Call_Ambulance':0,'MI':0}]= 0   #You fine and nothing bad happens + no wasted medical resources

    #-150 working best 
    dn.utility('Utility')[{'Call_Ambulance':0,'MI':1}]= -100   #You dead af

    dn.utility('Utility')[{'Call_Ambulance':1,'MI':0}]= -50     #Wasted medical resources (and your own money)

    dn.utility('Utility')[{'Call_Ambulance':1,'MI':1}]= 100     #Your life will most likely be saved

    dn.saveBIFXML("./output/MI_decision.bifxml")
    with open("./output/MI_decision.bifxml","r") as out:
        print(out.read())

    gumImage.export(dn,"./output/MI_decision.png")

    #CPT
    #############
    #Pr(Call_Ambulance | MI):

    # dn.cpt('Call_Ambulance')[{'MI':0}] = [0.97,0.03]
    # dn.cpt('Call_Ambulance')[{'MI':1}] = [0.395,0.605]


    return dn

def addProbabilities(model):

    #Prior Probabilties
    ##########

    #P(Smoking):
    model.cpt("Smokes").fillWith([0.735,0.265])

    #P(Alcohol_Abuse):
    model.cpt("Alcohol_Abuse").fillWith([0.777,0.223])

    #P(Physical_Activity):
    model.cpt("Physical_Activity").fillWith([0.275,0.725])

    #P(Cholestrol):
    model.cpt('Cholesterol').fillWith([0.59,0.41])

    #P(Chest_Pain):
    model.cpt('Chest_Pain').fillWith([0.7,0.3])
    #CPTs
    ##########

    #P(Hypertension | Smoking, Alcohol_Abuse):

    model.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':0}] = [0.701, 0.299]
    model.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':0}] = [0.715, 0.285]
    model.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':1}] = [0.665, 0.335]
    model.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':1}] = [0.68, 0.32]

    #P(Heart_Disease | Hypertension, Cholesterol, Smokes, Alcohol_Abuse, Physical_Activity)
    heart_disease_cpt = heart_disease_table.get_conditional_dictionary()
    for key, value in heart_disease_cpt.items():
        key_dict = {'Hypertension':key[0], 'Cholesterol':key[1], 'Smokes': key[2], 'Alcohol_Abuse':key[3], 'Physical_Activity':key[4]}
        model.cpt("Heart_Disease")[key_dict] = [value[0], value[1]]


    #P(MI | Heart_Disease):

    # model.cpt('MI')[{'Heart_Disease':0}] = [0.999609,1-0.999609]
    # model.cpt('MI')[{'Heart_Disease':1}] = [0.5489,0.4511]

    #P(ST_Segment_Elevation | MI):

    model.cpt("ST_Segment_Elevation")[{'MI':0}] = [0.89,0.11]
    model.cpt("ST_Segment_Elevation")[{'MI':1}] = [0.175,0.825]

    

    # P(Chest Pain | MI):

    # model.cpt('Chest_Pain')[{'MI':0}] = [0.75,0.25]
    # model.cpt('Chest_Pain')[{'MI':1}] = [1/3,2/3]

    #P(MI | Heart_Disease, Chest_Pain):

#     -|-- : 99,96% (1)
# +|-- : 0,4%

# -|+- 54,89% (1)
# +|+- 45,11%

# -|-+ : 0.85 (2)
# +|-+ : 0.15

# -|++ : 0.28
# +|++ : 0.72 (2)

    model.cpt("MI")[{'Heart_Disease':0, 'Chest_Pain':0}] = [0.9996, 0.0004]
    model.cpt("MI")[{'Heart_Disease':0, 'Chest_Pain':1}] = [0.85,0.15]
    model.cpt("MI")[{'Heart_Disease':1, 'Chest_Pain':0}] = [0.8489,0.1511]
    model.cpt("MI")[{'Heart_Disease':1, 'Chest_Pain':1}] = [0.28,0.72]

    return model

def inference_decision(model, evidence = None):

    ie = gum.ShaferShenoyLIMIDInference(model)
    if ((evidence is not None) and (isinstance(evidence,dict))):
        ie.setEvidence(evidence)
    ie.makeInference()
    for node in nodes:
        print (ie.posterior(node))
    print("Optimal Decision:")
    print(ie.optimalDecision("Call_Ambulance"))

def inference(model, evidence = None):
    
    ie=gum.LazyPropagation(model)
    if ( (evidence is not None) and (isinstance(evidence,dict))):
        ie.setEvidence(evidence)
    
    ie.makeInference()
    for node in nodes:
        print (ie.posterior(node))


def testIndep(model,x,y,knowing):
    res="" if model.isIndependent(x,y,knowing) else " NOT"
    giv="." if len(knowing)==0 else f" given {knowing}."
    print(f"{x} and {y} are{res} independent{giv}")