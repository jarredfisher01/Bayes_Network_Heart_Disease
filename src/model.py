import pyAgrum as gum
import pyAgrum.lib.image as gumImage
import heart_disease_table


nodes=['Smokes', 'Hypertension', 'Heart_Disease', 'Cholesterol', 'Alcohol_Abuse', 'Physical_Activity', 'MI', 'ST_Segment_Elevation', 'Chest_Pain']

links = [('Smokes','Hypertension'),('Hypertension','Heart_Disease'),('Smokes','Heart_Disease'),
('Cholesterol','Heart_Disease'),('Alcohol_Abuse','Hypertension'),('Alcohol_Abuse','Heart_Disease'),('Physical_Activity','Heart_Disease'),
('Heart_Disease','MI'),('MI','ST_Segment_Elevation'),('Chest_Pain','MI')]


def build_model():
    '''This is the method that makes the nodes and the arcs for the standard Bayesian Network.
    This method does not include any links/nodes for influence/decision diagram functionality.
    The method returns the model and exports the model as both a BIF and image files'''

    global links
    #Topology:
    ##########
    
    bn=gum.BayesNet('MI_bayes')
    for node in nodes:
        bn.add(node,2)
    
    
    for link in links:
        bn.addArc(*link)

    bn = addProbabilities(bn)

    gum.saveBN(bn,"MI_bayes.bif")
    with open("MI_bayes.bif","r") as out:
        print(out.read())

    gumImage.export(bn,"MI_bayes.png")

    return bn

def build_model_decision():
    '''This is the method that makes the nodes and the arcs for the Decision Network.
    This method includes creating any links/nodes for influence/decision diagram functionality.
    We add decision nodes and utility blocks 
    The method returns the model and exports the model as both a BIF and image files'''
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


    dn.utility('Utility')[{'Call_Ambulance':0,'MI':0}]= 0   #You fine and nothing bad happens + no wasted medical resources

    dn.utility('Utility')[{'Call_Ambulance':0,'MI':1}]= -100   #You dead af

    dn.utility('Utility')[{'Call_Ambulance':1,'MI':0}]= -50     #Wasted medical resources (and your own money)

    dn.utility('Utility')[{'Call_Ambulance':1,'MI':1}]= 100     #Your life will most likely be saved

    dn.saveBIFXML("MI_decision.bifxml")
    with open("MI_decision.bifxml","r") as out:
        print(out.read())

    gumImage.export(dn,"MI_decision.png")

    return dn

def addProbabilities(model):
    '''This is the method that represents the raw probability values that get fed into either the bayesian network 
    or the decision network. We load all the conditional probability tables that are used in both the decision
    network and the standrad bayesian network all in this method'''
    #Prior Probabilties
    ##########

    #P(Smoking):
    model.cpt("Smokes").fillWith([0.735,0.265])

    #P(Alcohol_Abuse):
    model.cpt("Alcohol_Abuse").fillWith([0.777,0.223])

    #P(Physical_Activity):
    model.cpt("Physical_Activity").fillWith([0.275,0.725])

    #P(Cholestrol):
    model.cpt('Cholesterol').fillWith([0.61,0.39])

    #P(Chest_Pain):
    model.cpt('Chest_Pain').fillWith([0.7,0.3])
    #CPTs
    ##########

    #P(Hypertension | Smoking, Alcohol_Abuse):

    model.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':0}] = [0.701, 0.299]
    model.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':0}] = [0.635, 0.365]
    model.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':1}] = [0.615, 0.385]
    model.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':1}] = [0.608, 0.392]

    #P(Heart_Disease | Hypertension, Cholesterol, Smokes, Alcohol_Abuse, Physical_Activity)
    heart_disease_cpt = heart_disease_table.get_conditional_dictionary()
    for key, value in heart_disease_cpt.items():
        key_dict = {'Hypertension':key[0], 'Cholesterol':key[1], 'Smokes': key[2], 'Alcohol_Abuse':key[3], 'Physical_Activity':key[4]}
        model.cpt("Heart_Disease")[key_dict] = [value[0], value[1]]


    #P(ST_Segment_Elevation | MI):

    model.cpt("ST_Segment_Elevation")[{'MI':0}] = [0.89,0.11]
    model.cpt("ST_Segment_Elevation")[{'MI':1}] = [0.175,0.825]

    #P(MI | Heart_Disease, Chest_Pain):

    model.cpt("MI")[{'Heart_Disease':0, 'Chest_Pain':0}] = [0.9996, 0.0004]
    model.cpt("MI")[{'Heart_Disease':0, 'Chest_Pain':1}] = [0.85,0.15]
    model.cpt("MI")[{'Heart_Disease':1, 'Chest_Pain':0}] = [0.8489,0.1511]
    model.cpt("MI")[{'Heart_Disease':1, 'Chest_Pain':1}] = [0.28,0.72]

    return model

def inference_decision(model, evidence = None):
    '''This is a method that can be used from the wrapper.py that can run inferences for the decision network. 
    It will print out a table that gives the optimal decision given the evidence the network has seen 
    Its arguments are as follows:
    model: this is the pyAgrum model that either is the decision network or bayesian model
    evidence: this is a dictionary that represents evidence i.e. {'Cholesterol':1, 'Hypertension':1}'''

    ie = gum.ShaferShenoyLIMIDInference(model)
    if ((evidence is not None) and (isinstance(evidence,dict))):
        ie.setEvidence(evidence)
    ie.makeInference()
    for node in nodes:
        print (ie.posterior(node))
    print("Optimal Decision:")
    print(ie.optimalDecision("Call_Ambulance"))

def inference(model, evidence = None):
    '''This is a method that can be used from the wrapper.py that can run inferences for the standard bayesian network. 
    Its arguments are as follows:
    model: this is the pyAgrum model that either is the decision network or bayesian model
    evidence: this is a dictionary that represents evidence i.e. {'Cholesterol':1, 'Hypertension':1}'''
    ie=gum.LazyPropagation(model)
    if ( (evidence is not None) and (isinstance(evidence,dict))):
        ie.setEvidence(evidence)
    
    ie.makeInference()
    for node in nodes:
        print (ie.posterior(node))


def testIndep(model,x,y,knowing):
    '''This method tests conditional independence in a model'''
    res="" if model.isIndependent(x,y,knowing) else " NOT"
    giv="." if len(knowing)==0 else f" given {knowing}."
    print(f"{x} and {y} are{res} independent{giv}")