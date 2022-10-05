import pyAgrum as gum
import pyAgrum.lib.image as gumImage
import heart_disease_table


nodes=['Smokes', 'Hypertension', 'Heart_Disease', 'Cholesterol', 'Alcohol_Abuse', 'Physical_Activity', 'MI', 'ST_Segment_Elevation']

links = [('Smokes','Hypertension'),('Hypertension','Heart_Disease'),('Smokes','Heart_Disease'),
('Cholesterol','Heart_Disease'),('Alcohol_Abuse','Hypertension'),('Alcohol_Abuse','Heart_Disease'),('Physical_Activity','Heart_Disease'),
('Heart_Disease','MI'),('MI','ST_Segment_Elevation')]


def build_model():

    #Topology:
    ##########
    
    bn=gum.BayesNet('MI_bayes')
    for node in nodes:
        bn.add(node,2)
    
    
    for link in links:
        bn.addArc(*link)

    #Prior Probabilties
    ##########

    #P(Smoking):
    bn.cpt("Smokes").fillWith([0.735,0.265])

    #P(Alcohol_Abuse):
    bn.cpt("Alcohol_Abuse").fillWith([0.777,0.223])
    
    #P(Physical_Activity):
    bn.cpt("Physical_Activity").fillWith([0.275,0.725])
    
    #P(Cholestrol):
    bn.cpt('Cholesterol').fillWith([0.59,0.41])
    
    
    #CPTs
    ##########

    #P(Hypertension | Smoking, Alcohol_Abuse):

    bn.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':0}] = [0.701, 0.299]
    bn.cpt("Hypertension")[{'Smokes':0, 'Alcohol_Abuse':1}] = [0.665, 0.335]
    bn.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':0}] = [0.715, 0.285]
    bn.cpt("Hypertension")[{'Smokes':1, 'Alcohol_Abuse':1}] = [0.68, 0.32]

    #P(Heart_Disease | Hypertension, Cholesterol, Smokes, Alcohol_Abuse, Physical_Activity)
    heart_disease_cpt = heart_disease_table.get_conditional_dictionary()
    for key, value in heart_disease_cpt.items():
        key_dict = {'Hypertension':key[0], 'Cholesterol':key[1], 'Smokes': key[2], 'Alcohol_Abuse':key[3], 'Physical_Activity':key[4]}
        bn.cpt("Heart_Disease")[key_dict] = [value[0], value[1]]


    #P(MI | Heart_Disease):

    bn.cpt('MI')[{'Heart_Disease':0}] = [0.999609,0.000391]
    bn.cpt('MI')[{'Heart_Disease':1}] = [0.5489,0.4511]

    #P(ST_Segment_Elevation | MI):

    bn.cpt("ST_Segment_Elevation")[{'MI':0}] = [0.49,0.51]
    bn.cpt("ST_Segment_Elevation")[{'MI':1}] = [0.375,0.625]

    gum.saveBN(bn,"./output/MI_bayes.bif")
    # with open("MI_bayes.bif","r") as out:
    #     print(out.read())

    gumImage.export(bn,"./output/MI_bayes.png")

    return bn

def inference(bn, evidence = None):
    
    ie=gum.LazyPropagation(bn)
    if ( (evidence is not None) and (isinstance(evidence,dict))):
        ie.setEvidence(evidence)
    
    ie.makeInference()
    for node in nodes:
        print (ie.posterior(node))


def testIndep(bn,x,y,knowing):
    res="" if bn.isIndependent(x,y,knowing) else " NOT"
    giv="." if len(knowing)==0 else f" given {knowing}."
    print(f"{x} and {y} are{res} independent{giv}")